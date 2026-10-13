"""
bartz_integral_correction.py

Replaces the closed-form Bartz correlation (+/-30-50 % scatter) with an
integral boundary-layer method, station by station > Reports the
remaining uncertainty as a band with stated causes.

USAGE
    python bartz_integral_correction.py --self-test
    python bartz_integral_correction.py --conditions operating_conditions.json \\
        --geometry nozzle_geometry.csv --out integral_results
    Optional "integral" section in operating_conditions.json (see IntegralConfig).

    Then, in operating_conditions.json "gas", add
        "h_correction_csv": "integral_results/integral_correction_factors.csv",
        "h_correction_column": "F_high"      (or F_nominal / F_low)
    and run nozzle_wall_temperature.py as usual. Regenerate the CSV whenever
    the geometry, chamber pressure or gas properties change.

OUTPUTS
    integral_correction_factors.csv  F = h_int / h_Bartz (low/nominal/high) vs x
    integral_station_results.csv     per station: h (Bartz/integral/band), F, Kp, wall T
    integral_wall_history.csv        throat/peak wall T vs time, both models
    bartz_integral_correction.png    summary plots

WHY BARTZ IS OFF, AND THE FIX
    Bartz is a fully-developed pipe-flow correlation. In a short nozzle:
    1. No BL history: h depends on upstream growth. -> Integral energy
       equation carries enthalpy thickness along x.
    2. Relaminarization: when Kp = (mu/rho u^2) du/dx > ~2-3.5e-6 (small,
       low-pc motors), heat transfer drops. -> St blended part-way to
       laminar (w_max = 0.5), lagging over ~2 throat radii.
    3. Property variation: Bartz's sigma fit -> Eckert reference
       temperature T* = Te + 0.5(Tw - Te) + 0.22(Taw - Te).

METHOD (Ambrok closed form)
    With Y = r G* dH Delta_2, dH = cp(Taw - Tw), G* = rho* u,
    and St = C Pr^-n Re_D2^-m (turbulent 0.0125, 1/4, 1/2; laminar 0.2204, 1, 4/3):
        Y^(1+m) = Y0^(1+m) + (1+m) C Pr^-n Int G* (r dH)^(1+m) mu*^m dx
        h = St G* cp
    Y0 from an equivalent upstream run length (default 1 inlet diameter).
    F(x) is iterated with the wall solver at the burn-mean wall temperature
    (or the steady one: integral.F_reference = "steady").

"""

import argparse
import copy
import dataclasses
import math
import os

import numpy as np
import pandas as pd

import nozzle_wall_temperature as nwt


# Configuration

@dataclasses.dataclass
class IntegralConfig:
    C_turb: float = 0.0125               # turbulent St-Re_D2 constant (Kays & Crawford)
    upstream_run_length_m: float | None = None  # equivalent BL run before inlet (None -> inlet diameter)
    laminarization: bool = True
    Kp_onset: float = 2.0e-6             # Kp where turbulent -> laminar blending starts
    Kp_full: float = 3.5e-6              # Kp where the laminar fraction reaches w_max
    w_max: float = 0.5                   # max reversion: 0 = stays turbulent, 1 = fully laminar St
    relaxation_length_Rt: float = 2.0    # distance (throat radii) for the BL to respond to Kp
    reference_temperature: bool = True   # Eckert T* properties (False: static-temperature)
    F_reference: str = "transient"       # wall temperature F is evaluated at: "transient" | "steady"
    coupling_iterations: int = 4

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in (d or {}).items() if not k.startswith("_")})


C_LAM, M_LAM, N_LAM = 0.2204, 1.0, 4.0 / 3.0
M_TURB, N_TURB = 0.25, 0.5


# Integral boundary-layer heat transfer

def ambrok_stanton(x, r, Gs, mus, dH, Pr, C, m, n, run_length):
    """Closed-form integral energy solution. Returns (St, Re_D2)."""
    f = Gs * (r * dH) ** (1.0 + m) * mus ** m
    k = (1.0 + m) * C * Pr ** (-n)
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(x))])
    Y = (k * (f[0] * run_length + integral)) ** (1.0 / (1.0 + m))
    ReD = Y / (r * dH * mus)
    return C * Pr ** (-n) * ReD ** (-m), ReD


def gas_props(gas):
    g, cp = gas["gamma"], gas["cp_J_kgK"]
    return g, cp, cp * (g - 1.0) / g, gas["viscosity_Pa_s"], gas["Prandtl"]


def acceleration_parameter(x, gas, flow):
    g, cp, R, mu0, _ = gas_props(gas)
    Te, p = flow["T"], flow["p"]
    u = flow["M"] * np.sqrt(g * R * Te)
    rho = p / (R * Te)
    mu = mu0 * (Te / flow["T0"]) ** nwt.OMEGA
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(u > 1e-6, mu / (rho * u ** 2) * np.gradient(u, x), 0.0)


def laminar_fraction(x, Kp, ic: IntegralConfig, r_throat):
    """Fraction of the way from turbulent to laminar St. w_eq rises (smoothstep)
    from 0 at Kp_onset to w_max at Kp_full; w relaxes towards it over
    relaxation_length_Rt throat radii. The BL enters turbulent (w = 0)."""
    if not ic.laminarization or ic.w_max <= 0:
        return np.zeros_like(Kp)
    s = np.clip((Kp - ic.Kp_onset) / max(ic.Kp_full - ic.Kp_onset, 1e-30), 0.0, 1.0)
    w_eq = ic.w_max * s * s * (3.0 - 2.0 * s)
    L = ic.relaxation_length_Rt * r_throat
    if L <= 0:
        return w_eq
    w = np.zeros_like(w_eq)
    for i in range(1, len(w)):
        a = math.exp(-(x[i] - x[i - 1]) / L)
        w[i] = w_eq[i] + (w[i - 1] - w_eq[i]) * a
    return w


def integral_h(geo, gas, flow, Twg, ic: IntegralConfig, Kp=None):
    """Gas-side h [W/m2K] from the integral method at wall temperature Twg(x)."""
    g, cp, R, mu0, Pr = gas_props(gas)
    x, r = geo["x_m"].to_numpy(), geo["r_inner_m"].to_numpy()
    Te, Taw, T0 = flow["T"], flow["Taw"], flow["T0"]
    G = gas["chamber_pressure_Pa"] / gas["c_star_m_s"] / flow["area_ratio"]   # rho u
    Tw = np.minimum(np.asarray(Twg, float), Taw - 1.0)
    Tref = Te + 0.5 * (Tw - Te) + 0.22 * (Taw - Te) if ic.reference_temperature else Te
    Gs = G * Te / Tref
    mus = mu0 * (Tref / T0) ** nwt.OMEGA
    dH = cp * (Taw - Tw)
    L0 = ic.upstream_run_length_m if ic.upstream_run_length_m is not None else 2.0 * r[0]

    St_t, ReD_t = ambrok_stanton(x, r, Gs, mus, dH, Pr, ic.C_turb, M_TURB, N_TURB, L0)
    St_l, _ = ambrok_stanton(x, r, Gs, mus, dH, Pr, C_LAM, M_LAM, N_LAM, L0)
    if Kp is None:
        Kp = acceleration_parameter(x, gas, flow)
    w = laminar_fraction(x, Kp, ic, r[flow["i_throat"]])
    St = (1.0 - w) * St_t + w * St_l
    h = gas.get("bartz_multiplier", 1.0) * St * Gs * cp
    return {"h": h, "Kp": Kp, "w_lam": w, "ReD2_turb": ReD_t, "run_length_m": L0}


def bartz_h(flow, Twg):
    return flow["h0"] * nwt.bartz_sigma(Twg, flow["T0"], flow["X"])


def with_F(flow, F):
    f = dict(flow)
    f["h0"] = flow["h0"] * F
    return f


# Coupling to the wall solvers

def couple_steady(geo, gas, flow, wall, ext, ic):
    base = nwt.solve_steady(geo, flow, wall, ext)
    Twg = base["T_inner"].copy()
    for _ in range(20):
        F = integral_h(geo, gas, flow, Twg, ic)["h"] / bartz_h(flow, Twg)
        res = nwt.solve_steady(geo, with_F(flow, F), wall, ext)
        change = float(np.max(np.abs(res["T_inner"] - Twg)))
        Twg = res["T_inner"]
        if change < 0.5:
            break
    F = integral_h(geo, gas, flow, Twg, ic)["h"] / bartz_h(flow, Twg)
    return base, res, F, Twg


def burn_mean(tr):
    return tr["T_inner_hist"][1:].mean(axis=0)


def couple_transient(geo, gas, flow, wall, ext, tr_cfg, ic):
    Twg = 0.5 * (wall["initial_temperature_K"] + flow["Taw"])
    F_prev = None
    for _ in range(ic.coupling_iterations):
        F = integral_h(geo, gas, flow, Twg, ic)["h"] / bartz_h(flow, Twg)
        tr = nwt.solve_transient(geo, with_F(flow, F), wall, ext, tr_cfg)
        Twg = burn_mean(tr)
        if F_prev is not None and np.max(np.abs(F / F_prev - 1.0)) < 0.005:
            break
        F_prev = F
    F = integral_h(geo, gas, flow, Twg, ic)["h"] / bartz_h(flow, Twg)
    return tr, F, Twg


# Uncertainty ensemble (structural: which assumption is wrong?)

def ensemble_variants(ic: IntegralConfig, L0):
    rep = lambda **kw: dataclasses.replace(ic, **kw)
    return {
        "nominal": ic,
        "short upstream run (x0.25)": rep(upstream_run_length_m=0.25 * L0),
        "long upstream run (x4)": rep(upstream_run_length_m=4.0 * L0),
        "no laminarization (stays turbulent)": rep(laminarization=False),
        "full laminarization (w_max = 1)": rep(w_max=1.0),
        "early onset (Kp 1.5-2.5e-6)": rep(Kp_onset=1.5e-6, Kp_full=2.5e-6),
        "late onset (Kp 3-5e-6)": rep(Kp_onset=3.0e-6, Kp_full=5.0e-6),
        "fast BL response (0.5 Rt)": rep(relaxation_length_Rt=0.5),
        "slow BL response (5 Rt)": rep(relaxation_length_Rt=5.0),
        "static-T properties": rep(reference_temperature=False),
    }


def ensemble(geo, gas, flow, Twg, ic, L0):
    hb = bartz_h(flow, Twg)
    return {name: integral_h(geo, gas, flow, Twg, v)["h"] / hb
            for name, v in ensemble_variants(ic, L0).items()}



# Self-test

def example_case(x_origin=-0.05):
    """In-memory case in nozzle_wall_temperature.py format: small hybrid,
    conical nozzle (Dt 25 mm, 40/15 deg), graphite-like wall, pc 2.5 MPa.
    Synthetic: for testing only, not your motor."""
    Rt, Ru, Rd, Rch, Re_ = 0.0125, 1.5 * 0.0125, 0.382 * 0.0125, 0.040, 0.027
    tc, td = math.radians(40), math.radians(15)
    phi = np.linspace(tc, 0, 400); xu, ru = -Ru * np.sin(phi), Rt + Ru * (1 - np.cos(phi))
    xc0 = xu[0] - (Rch - ru[0]) / math.tan(tc)
    xc = np.linspace(xc0, xu[0], 400)[:-1]; rc = ru[0] + (xu[0] - xc) * math.tan(tc)
    phi = np.linspace(0, td, 400)[1:]; xd, rd = Rd * np.sin(phi), Rt + Rd * (1 - np.cos(phi))
    xdv = np.linspace(xd[-1], xd[-1] + (Re_ - rd[-1]) / math.tan(td), 400)[1:]
    rdv = rd[-1] + (xdv - xd[-1]) * math.tan(td)
    xa, ra = np.concatenate([xc, xu, xd, xdv]), np.concatenate([rc, ru, rd, rdv])
    x = np.union1d(np.linspace(xa.min(), xa.max(), 121), [0.0])
    r = np.interp(x, xa, ra)
    geo = pd.DataFrame({"x_m": x - x_origin, "r_inner_m": r, "wall_thickness_m": 0.045 - r})
    g, MW, T0 = 1.24, 26.5, 2600.0
    R = 8314.462618 / MW
    cstar = math.sqrt(g * R * T0) / (g * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))
    cfg = {
        "case_label": "synthetic test case (not your motor)",
        "gas": {"chamber_pressure_Pa": 2.5e6, "stagnation_temperature_K": T0, "gamma": g,
                "cp_J_kgK": g * R / (g - 1), "viscosity_Pa_s": 8.5e-5, "Prandtl": 4 * g / (9 * g - 5),
                "c_star_m_s": cstar, "throat_curvature_radius_m": 0.94 * Rt,
                "back_pressure_Pa": 1.013e5, "bartz_multiplier": 1.0},
        "wall": {"conductivity_W_mK": 60.0, "density_kg_m3": 1800.0, "cp_J_kgK": 1500.0,
                 "initial_temperature_K": 293.15},
        "external": {"h_W_m2K": 10.0, "bulk_temperature_K": 293.15, "emissivity": 0.8,
                     "surroundings_temperature_K": 293.15},
        "transient": {"burn_duration_s": 5.0, "radial_cells": 24, "output_times": 101,
                      "rtol": 1e-5, "atol_K": 1e-3},
    }
    return cfg, geo


def self_test():
    print("Self-test")
    x = np.linspace(1e-4, 1.0, 20001)
    r, ones = np.ones_like(x), np.ones_like(x)
    G, mu, Pr, dH = 50.0, 5e-5, 0.7, 1e6
    Rex = G * x / mu
    ok = True
    for name, C, m, n, ref in [
            ("turbulent", 0.0125, 0.25, 0.5, 0.0287 * Rex ** -0.2 * Pr ** -0.4),
            ("laminar", C_LAM, 1.0, 4 / 3, 0.332 * Rex ** -0.5 * Pr ** (-2 / 3))]:
        St, _ = ambrok_stanton(x, r, G * ones, mu * ones, dH * ones, Pr, C, m, n, x[0])
        err = np.max(np.abs(St[100:] / ref[100:] - 1.0))
        ok &= err < 5e-3
        print(f"  flat plate {name:9s}: max |St/St_exact - 1| = {err:.1e}  {'PASS' if err < 5e-3 else 'FAIL'}")

    Tw = None
    Fs = []
    for origin in (0.0, -0.123):      # x origin of the geometry file must not change F
        cfg, geo = example_case(x_origin=origin)
        fl = nwt.compute_flow(geo, cfg["gas"])
        Tw = np.full(len(geo), 1000.0)
        Fs.append(integral_h(geo, cfg["gas"], fl, Tw, IntegralConfig())["h"] / bartz_h(fl, Tw))
    d = float(np.max(np.abs(Fs[0] - Fs[1])))
    ok &= d < 1e-9
    print(f"  F independent of geometry x origin:  max diff {d:.1e}  {'PASS' if d < 1e-9 else 'FAIL'}")

    cfg_m = copy.deepcopy(cfg); cfg_m["gas"]["bartz_multiplier"] = 0.7   # must cancel out of F
    fl2 = nwt.compute_flow(geo, cfg_m["gas"])
    F2 = integral_h(geo, cfg_m["gas"], fl2, Tw, IntegralConfig())["h"] / bartz_h(fl2, Tw)
    d = float(np.max(np.abs(Fs[1] - F2)))
    ok &= d < 1e-9
    print(f"  F independent of bartz_multiplier:   max diff {d:.1e}  {'PASS' if d < 1e-9 else 'FAIL'}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


# Output

def plot(out_png, geo, flow, hi, ic, Fn, Fl, Fh, hb, tr_b, tr_n, tr_l, tr_h, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = geo["x_m"].to_numpy()
    xmm = (x - x[flow["i_throat"]]) * 1e3
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    a.fill_between(xmm, 0.5 * hb / 1e3, 1.5 * hb / 1e3, color="tab:red", alpha=.06,
                   label="Bartz ±50 % (old uncertainty)")
    a.fill_between(xmm, Fl * hb / 1e3, Fh * hb / 1e3, color="tab:blue", alpha=.2, label="Integral band")
    a.plot(xmm, hb / 1e3, color="tab:red", ls="--", lw=1.5, label="Bartz")
    a.plot(xmm, Fn * hb / 1e3, color="tab:blue", lw=2, label="Integral, nominal")
    a.set_xlabel("x from throat (mm)"); a.set_ylabel("h_g (kW/m²K)")
    a.set_title("Gas-side heat transfer coefficient"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[0, 1]
    a.fill_between(xmm, Fl, Fh, color="tab:blue", alpha=.2)
    a.plot(xmm, Fn, color="tab:blue", lw=2, label="F = h_integral / h_Bartz")
    a.axhline(1.0, color="tab:red", ls="--", lw=1)
    a.set_xlabel("x from throat (mm)"); a.set_ylabel("Correction factor F")
    a.set_title("Per-station correction (integral_correction_factors.csv)"); a.grid(alpha=.3)
    b = a.twinx()
    b.semilogy(xmm, np.clip(hi["Kp"], 1e-9, None), color="tab:purple", lw=1, label="Kp")
    for k in (ic.Kp_onset, ic.Kp_full):
        b.axhline(k, color="tab:purple", ls=":", lw=1)
    b.set_ylabel("Acceleration parameter Kp", color="tab:purple")
    l1, n1 = a.get_legend_handles_labels(); l2, n2 = b.get_legend_handles_labels()
    a.legend(l1 + l2, n1 + n2, fontsize=8, loc="lower right")

    a = ax[1, 0]
    a.plot(xmm, flow["Taw"], "k:", lw=1, label="Adiabatic wall T_aw")
    a.fill_between(xmm, tr_l["T_inner"], tr_h["T_inner"], color="tab:blue", alpha=.2, label="Integral band")
    a.plot(xmm, tr_b["T_inner"], color="tab:red", ls="--", label="Bartz")
    a.plot(xmm, tr_n["T_inner"], color="tab:blue", lw=2, label="Integral, nominal")
    a.set_xlabel("x from throat (mm)"); a.set_ylabel("Temperature (K)")
    a.set_title(f"Inner-wall temperature at end of burn ({tr_b['t'][-1]:.1f} s)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 1]
    i = flow["i_throat"]
    a.fill_between(tr_b["t"], tr_l["T_inner_hist"][:, i], tr_h["T_inner_hist"][:, i],
                   color="tab:blue", alpha=.2, label="Integral band")
    a.plot(tr_b["t"], tr_b["T_inner_hist"][:, i], color="tab:red", ls="--", label="Bartz")
    a.plot(tr_b["t"], tr_n["T_inner_hist"][:, i], color="tab:blue", lw=2, label="Integral, nominal")
    a.set_xlabel("Time (s)"); a.set_ylabel("Throat inner-wall T (K)")
    a.set_title("Throat temperature history"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def run(cfg, geo, out_dir="integral_results"):
    cfg = copy.deepcopy(cfg)
    gas, wall, ext, tr_cfg = cfg["gas"], cfg["wall"], cfg["external"], cfg["transient"]
    if gas.pop("h_correction_csv", None):            # F is always relative to plain Bartz
        print("note: ignoring gas.h_correction_csv here (F is computed against plain Bartz)")
    gas.pop("h_correction_column", None)
    ic = IntegralConfig.from_dict(cfg.get("integral"))
    os.makedirs(out_dir, exist_ok=True)

    flow = nwt.compute_flow(geo, gas)
    for w in flow["warnings"]:
        print("WARNING:", w)
    i = flow["i_throat"]
    x = geo["x_m"].to_numpy()

    st_b, st_n, F_st, Tw_st = couple_steady(geo, gas, flow, wall, ext, ic)
    tr_b = nwt.solve_transient(geo, flow, wall, ext, tr_cfg)
    tr_n, F_tr, Tw_tr = couple_transient(geo, gas, flow, wall, ext, tr_cfg, ic)
    use_tr = ic.F_reference.lower() != "steady"
    Fn, Tw_ref = (F_tr, Tw_tr) if use_tr else (F_st, Tw_st)

    hi = integral_h(geo, gas, flow, Tw_ref, ic)
    ens = ensemble(geo, gas, flow, Tw_ref, ic, hi["run_length_m"])
    Fs = np.array(list(ens.values()))
    Fl, Fh = Fs.min(axis=0), Fs.max(axis=0)
    tr_l = nwt.solve_transient(geo, with_F(flow, Fl), wall, ext, tr_cfg)
    tr_h = nwt.solve_transient(geo, with_F(flow, Fh), wall, ext, tr_cfg)
    hb = bartz_h(flow, Tw_ref)

    # ---- files ----
    xt = x - x[i]
    pd.DataFrame({"x_m": x, "x_from_throat_m": xt, "r_inner_m": geo["r_inner_m"],
                  "F_low": Fl, "F_nominal": Fn, "F_high": Fh}).to_csv(
        os.path.join(out_dir, "integral_correction_factors.csv"), index=False)
    df = pd.DataFrame({
        "x_m": x, "x_from_throat_m": xt, "r_inner_m": geo["r_inner_m"], "Mach": flow["M"],
        "T_adiabatic_wall_K": flow["Taw"], "T_wall_reference_K": Tw_ref,
        "Kp": hi["Kp"], "laminar_fraction": hi["w_lam"], "Re_Delta2_turb": hi["ReD2_turb"],
        "h_bartz_W_m2K": hb, "h_integral_W_m2K": Fn * hb,
        "h_integral_low_W_m2K": Fl * hb, "h_integral_high_W_m2K": Fh * hb,
        "F_low": Fl, "F_nominal": Fn, "F_high": Fh, "F_steady_reference": F_st,
        "T_inner_K_steady_bartz": st_b["T_inner"], "T_inner_K_steady_integral": st_n["T_inner"],
        "T_inner_K_burn_end_bartz": tr_b["T_inner"], "T_inner_K_burn_end_integral": tr_n["T_inner"],
        "T_inner_K_burn_end_integral_low": tr_l["T_inner"],
        "T_inner_K_burn_end_integral_high": tr_h["T_inner"]})
    for name, Fv in ens.items():
        df[f"F[{name}]"] = Fv
    df.to_csv(os.path.join(out_dir, "integral_station_results.csv"), index=False)
    pd.DataFrame({"t_s": tr_b["t"],
                  "T_inner_throat_bartz_K": tr_b["T_inner_hist"][:, i],
                  "T_inner_throat_integral_K": tr_n["T_inner_hist"][:, i],
                  "T_inner_throat_integral_low_K": tr_l["T_inner_hist"][:, i],
                  "T_inner_throat_integral_high_K": tr_h["T_inner_hist"][:, i],
                  "T_inner_peak_bartz_K": tr_b["T_inner_hist"].max(axis=1),
                  "T_inner_peak_integral_K": tr_n["T_inner_hist"].max(axis=1)}).to_csv(
        os.path.join(out_dir, "integral_wall_history.csv"), index=False)
    plot(os.path.join(out_dir, "bartz_integral_correction.png"), geo, flow, hi, ic,
         Fn, Fl, Fh, hb, tr_b, tr_n, tr_l, tr_h, cfg.get("case_label", ""))

    # ---- summary ----
    print("=" * 72)
    print(f"BARTZ vs INTEGRAL BOUNDARY-LAYER METHOD   {cfg.get('case_label', '')}")
    print("=" * 72)
    print(f"  Upstream run length {hi['run_length_m']*1e3:.1f} mm | Kp at throat {hi['Kp'][i]:.2e} "
          f"(band {ic.Kp_onset:.1e}-{ic.Kp_full:.1e}) | laminar fraction at throat {hi['w_lam'][i]:.2f}")
    print(f"  F evaluated at the {'burn-mean' if use_tr else 'steady'} wall temperature; "
          f"throat F steady-ref {F_st[i]:.2f} vs transient-ref {F_tr[i]:.2f}")
    print(f"  THROAT h_g: Bartz {hb[i]/1e3:.2f} kW/m2K -> integral {Fn[i]*hb[i]/1e3:.2f} kW/m2K "
          f"(F = {Fn[i]:.2f}, band {Fl[i]:.2f}-{Fh[i]:.2f})")
    j = int(np.argmin(np.abs(xt - 0.5 * xt[-1])))
    print(f"  Divergent (x = {xt[j]*1e3:.0f} mm from throat): F = {Fn[j]:.2f}, band {Fl[j]:.2f}-{Fh[j]:.2f}")
    print("  Spread at the throat by assumption:")
    for name, Fv in sorted(ens.items(), key=lambda kv: -abs(kv[1][i] - Fn[i])):
        if name != "nominal":
            print(f"    {name:40s} F = {Fv[i]:.3f}  ({100*(Fv[i]/Fn[i]-1):+5.1f} %)")
    print(f"  END OF BURN throat inner wall: Bartz {tr_b['T_inner'][i]:.0f} K, integral "
          f"{tr_n['T_inner'][i]:.0f} K [{tr_l['T_inner'][i]:.0f}-{tr_h['T_inner'][i]:.0f} K]")
    print(f"  STEADY throat inner wall:      Bartz {st_b['T_inner'][i]:.0f} K, integral "
          f"{st_n['T_inner'][i]:.0f} K")
    print(f"  Wrote {os.path.join(out_dir, 'integral_correction_factors.csv')}")
    print("  -> set gas.h_correction_csv to this file in operating_conditions.json and run "
          "nozzle_wall_temperature.py")
    return {"F": Fn, "F_low": Fl, "F_high": Fh, "flow": flow, "transient_integral": tr_n,
            "transient_low": tr_l, "transient_high": tr_h, "steady_integral": st_n}


def main():
    ap = argparse.ArgumentParser(description="Integral boundary-layer correction to Bartz")
    ap.add_argument("--conditions", default="operating_conditions.json")
    ap.add_argument("--geometry", default="nozzle_geometry.csv")
    ap.add_argument("--out", default="integral_results")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    cfg, geo = nwt.load_inputs(args.conditions, args.geometry)
    run(cfg, geo, args.out)


if __name__ == "__main__":
    main()