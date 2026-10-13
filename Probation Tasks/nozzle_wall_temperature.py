"""
nozzle_wall_temperature.py
==========================

Nozzle wall temperature along the length of a rocket nozzle, using the
Bartz (1957) correlation for the gas-side convective heat transfer
coefficient.

INPUTS
------
operating_conditions.json
    gas       chamber pressure, stagnation temperature, gamma, cp, viscosity,
              Prandtl number, c*, throat radius of curvature, back pressure,
              Bartz multiplier (empirical correction, 1.0 = plain Bartz),
              optional h_correction_csv / h_correction_column (per-station
              correction factors from bartz_integral_correction.py)
    wall      conductivity, density, specific heat, initial temperature
    external  back-face cooling: effective h, bulk temperature,
              radiation emissivity and surroundings temperature
    transient burn duration, radial cells, number of output times, solver tolerances

nozzle_geometry.csv
    x_m               axial position [m] (any origin; throat = minimum radius)
    r_inner_m         hot-gas-side wall radius [m]
    wall_thickness_m  local wall thickness [m]

OUTPUTS (in --out directory)
-------
wall_temperature_profile.csv  per station: flow state, Reynolds number, and
                              inner/outer wall temperature, h_g and heat flux
                              for (a) steady state and (b) end of burn
wall_temperature_history.csv  throat and peak inner-wall temperature vs time
nozzle_wall_temperature.png   summary plots
verification.json             (with --verify) grid / tolerance / energy checks


    python nozzle_wall_temperature.py --conditions operating_conditions.json \\
        --geometry nozzle_geometry.csv --out results
    # optional: numerical verification and comparison with a reference profile
    python nozzle_wall_temperature.py ... --verify --reference wall_temperature_profile.csv

METHOD > Refer to any academic literature on the readme
1. Isentropic 1-D flow.  Local Mach number from the area-Mach relation
   (subsonic branch upstream of the throat, supersonic downstream):
       A/At = (1/M) * [ (2/(g+1)) (1 + (g-1)/2 M^2) ]^((g+1)/(2(g-1)))
   Static T = T0/X, p = p0 X^(-g/(g-1)), with X = 1 + (g-1)/2 M^2.

2. Adiabatic wall (recovery) temperature, turbulent recovery factor r = Pr^(1/3):
       T_aw = T0 (1 + r (g-1)/2 M^2) / X

3. Bartz correlation, SI form (the g_c of the original US-unit form drops out):
       h_g = C_B * F(x) * 0.026/Dt^0.2 * (mu^0.2 cp / Pr^0.6)_0 * (p0/c*)^0.8
             * (Dt/Rc)^0.1 * (At/A)^0.9 * sigma
       sigma = [0.5 (Twg/T0) X + 0.5]^-(0.8 - w/5) * X^-(w/5),   w = 0.6
   where C_B is the "bartz_multiplier" and F(x) the optional per-station
   correction (1.0 if no h_correction_csv). Gas properties are taken at the
   stagnation condition, as Bartz specifies. sigma depends on the hot-wall
   temperature Twg, so h_g and Twg are always solved together.

4. Wall heat conduction: 1-D radial (cylindrical shell) at every axial
   station; axial conduction along the wall is neglected.
   Hot face :  q_in  = h_g (T_aw - Twg)
   Back face:  q_out = h_ext (Two - T_bulk) + eps sigma_SB (Two^4 - T_surr^4)
   - Steady state:  q_in * r_in = k (Twg - Two) / ln(r_out/r_in) = q_out * r_out
   - Transient:     finite-volume method of lines (equal-thickness radial
                    cells, exact cylindrical conductances, half-cell
                    resistance to each surface), integrated with an implicit
                    BDF solver.  The heat in and out are carried as extra ODE
                    states so the energy audit is exact to round-off.

ASSUMPTIONS / LIMITATIONS
- Steady, frozen, calorically perfect gas; constant properties at T0.
- Flow attached and shock-free (a warning is printed if the back pressure
  is high enough to push a normal shock inside the nozzle).
- Plain Bartz is an empirical turbulent correlation; scatter against test
  data of +/-30-50 % is typical, and it tends to over-predict in the
  convergent section and on small low-pressure nozzles. 
   FIX: Use bartz_integral_correction.py to generate h_correction_csv, and bartz_multiplier to calibrate against hot-fire thermocouple data.
- No gas radiation, film cooling, deposition or ablation.
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann constant [W/m^2 K^4]
OMEGA = 0.6                 # viscosity-temperature exponent in Bartz sigma


# Input

def load_inputs(conditions_path, geometry_path, axial_refine=1):
    with open(conditions_path) as f:
        cfg = json.load(f)
    geo = pd.read_csv(geometry_path).sort_values("x_m").reset_index(drop=True)
    for col in ("x_m", "r_inner_m", "wall_thickness_m"):
        if col not in geo:
            raise ValueError(f"geometry file is missing column '{col}'")
    if axial_refine > 1:
        geo = refine_geometry(geo, axial_refine)
    return cfg, geo


def refine_geometry(geo, factor):
    """Insert (factor-1) linearly interpolated stations between each pair of
    input stations (factor 2: 121 -> 241 stations). Used for the axial grid study."""
    x = geo["x_m"].to_numpy()
    xf = np.concatenate([np.linspace(x[i], x[i + 1], factor + 1)[:-1]
                         for i in range(len(x) - 1)] + [x[-1:]])
    return pd.DataFrame({c: np.interp(xf, x, geo[c].to_numpy()) for c in geo.columns})


def load_h_correction(path, x_from_throat, r_inner, column="F_nominal"):
    """Per-station multiplier F(x) from bartz_integral_correction.py, interpolated
    onto this geometry (x measured from the throat). Refuses a file made for a
    different nozzle: it must cover the stations and match the contour."""
    c = pd.read_csv(path)
    for col in ("x_from_throat_m", "r_inner_m", column):
        if col not in c:
            raise ValueError(f"{path}: missing column '{col}' "
                             "(regenerate it with bartz_integral_correction.py)")
    xc = c["x_from_throat_m"].to_numpy()
    tol = 1e-6 + 1e-3 * (xc.max() - xc.min())
    if x_from_throat.min() < xc.min() - tol or x_from_throat.max() > xc.max() + tol:
        raise ValueError(f"{path} covers x = {xc.min()*1e3:.1f}..{xc.max()*1e3:.1f} mm from the "
                         f"throat but the geometry spans {x_from_throat.min()*1e3:.1f}.."
                         f"{x_from_throat.max()*1e3:.1f} mm: it was made for a different nozzle")
    r_c = np.interp(x_from_throat, xc, c["r_inner_m"].to_numpy())
    if np.max(np.abs(r_c / r_inner - 1.0)) > 0.02:
        raise ValueError(f"{path} was generated for a different contour (radius mismatch > 2 %); "
                         "regenerate it with bartz_integral_correction.py")
    return np.interp(x_from_throat, xc, c[column].to_numpy())



# Gas dynamics

def area_ratio_from_mach(M, g):
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** (
        (g + 1.0) / (2.0 * (g - 1.0)))


def mach_from_area_ratio(AR, g, supersonic):
    if AR <= 1.0 + 1e-12:
        return 1.0
    f = lambda M: area_ratio_from_mach(M, g) - AR
    return brentq(f, 1.0 + 1e-12, 50.0, xtol=1e-14) if supersonic \
        else brentq(f, 1e-9, 1.0 - 1e-12, xtol=1e-14)


def compute_flow(geo, gas):
    """Isentropic flow state and Bartz h_g (without sigma) at every station."""
    g = gas["gamma"]
    p0 = gas["chamber_pressure_Pa"]
    T0 = gas["stagnation_temperature_K"]
    cp, mu, Pr = gas["cp_J_kgK"], gas["viscosity_Pa_s"], gas["Prandtl"]
    cstar = gas["c_star_m_s"]
    Rc = gas["throat_curvature_radius_m"]
    R = cp * (g - 1.0) / g

    r = geo["r_inner_m"].to_numpy()
    i_t = int(np.argmin(r))
    r_t = r[i_t]
    Dt = 2.0 * r_t
    AR = (r / r_t) ** 2
    M = np.array([mach_from_area_ratio(a, g, supersonic=(i > i_t)) for i, a in enumerate(AR)])
    X = 1.0 + 0.5 * (g - 1.0) * M ** 2
    T = T0 / X
    p = p0 * X ** (-g / (g - 1.0))
    rf = Pr ** (1.0 / 3.0)
    Taw = T0 * (1.0 + rf * 0.5 * (g - 1.0) * M ** 2) / X

    # Reynolds number on local diameter, mass flux from the choked throat
    mdot = p0 * math.pi * r_t ** 2 / cstar
    Re = 4.0 * mdot / (math.pi * 2.0 * r * mu)

    h0 = (gas.get("bartz_multiplier", 1.0) * 0.026 / Dt ** 0.2
          * (mu ** 0.2 * cp / Pr ** 0.6) * (p0 / cstar) ** 0.8
          * (Dt / Rc) ** 0.1 * (1.0 / AR) ** 0.9)

    # optional per-station correction from bartz_integral_correction.py
    h_corr = None
    if gas.get("h_correction_csv"):
        x = geo["x_m"].to_numpy()
        h_corr = gas.get("h_correction_column", "F_nominal")
        h0 = h0 * load_h_correction(gas["h_correction_csv"], x - x[i_t], r, h_corr)

    # consistency checks on the supplied gas data
    cstar_ideal = math.sqrt(g * R * T0) / (g * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))
    warnings = []
    if abs(cstar / cstar_ideal - 1.0) > 0.02:
        warnings.append(f"c* = {cstar:.1f} m/s differs from ideal {cstar_ideal:.1f} m/s "
                        f"(R = cp(g-1)/g = {R:.1f} J/kgK); check gas inputs")
    pb = gas.get("back_pressure_Pa", 0.0)
    if pb > 0:
        Me, pe = M[-1], p[-1]
        p_shock = pe * (1.0 + 2.0 * g / (g + 1.0) * (Me ** 2 - 1.0))
        if pb > p_shock:
            warnings.append(f"back pressure {pb:.0f} Pa exceeds the pressure behind a normal "
                            f"shock at the exit ({p_shock:.0f} Pa): a shock sits inside the "
                            "nozzle and the isentropic results downstream of it are invalid")
        elif pb > pe:
            warnings.append(f"nozzle is over-expanded (pe = {pe:.0f} Pa < pb = {pb:.0f} Pa); "
                            "flow separation near the exit is possible")

    return {"i_throat": i_t, "Dt": Dt, "area_ratio": AR, "M": M, "X": X, "T": T,
            "p": p, "Taw": Taw, "Re": Re, "h0": h0, "T0": T0, "warnings": warnings,
            "h_correction": h_corr}


def bartz_sigma(Twg, T0, X, omega=OMEGA):
    return 1.0 / ((0.5 * Twg / T0 * X + 0.5) ** (0.8 - omega / 5.0) * X ** (omega / 5.0))


def dsigma_dTwg(Twg, T0, X, omega=OMEGA):
    a = 0.8 - omega / 5.0
    A = 0.5 * Twg / T0 * X + 0.5
    return -a * bartz_sigma(Twg, T0, X, omega) / A * 0.5 * X / T0


# Steady state

def solve_steady(geo, flow, wall, ext):
    """Per station: h_g(Twg) (Taw - Twg) r_in = k (Twg - Two)/ln(ro/ri) = q_out(Two) r_out."""
    k = wall["conductivity_W_mK"]
    hb, Tb = ext["h_W_m2K"], ext["bulk_temperature_K"]
    eps, Ts = ext["emissivity"], ext["surroundings_temperature_K"]
    ri = geo["r_inner_m"].to_numpy()
    ro = ri + geo["wall_thickness_m"].to_numpy()
    T_lo = min(Tb, Ts)
    n = len(ri)
    Twg, Two = np.zeros(n), np.zeros(n)

    for i in range(n):
        Kw = k / math.log(ro[i] / ri[i])            # wall conductance per radian

        def outer(Tg):
            f = lambda To: Kw * (Tg - To) - ro[i] * (hb * (To - Tb) + eps * SIGMA_SB * (To ** 4 - Ts ** 4))
            return brentq(f, T_lo, Tg, xtol=1e-12)

        def resid(Tg):
            To = outer(Tg)
            h = flow["h0"][i] * bartz_sigma(Tg, flow["T0"], flow["X"][i])
            return h * (flow["Taw"][i] - Tg) * ri[i] - Kw * (Tg - To)

        Twg[i] = brentq(resid, T_lo + 1e-9, flow["Taw"][i], xtol=1e-12)
        Two[i] = outer(Twg[i])

    h = flow["h0"] * bartz_sigma(Twg, flow["T0"], flow["X"])
    return {"T_inner": Twg, "T_outer": Two, "h": h, "q": h * (flow["Taw"] - Twg)}


# Transient (method of lines, radial finite volume)

class TransientModel:
    def __init__(self, geo, flow, wall, ext, n_cells):
        self.flow, self.ext = flow, ext
        self.k = wall["conductivity_W_mK"]
        self.rho_c = wall["density_kg_m3"] * wall["cp_J_kgK"]
        self.ri = geo["r_inner_m"].to_numpy()
        self.ro = self.ri + geo["wall_thickness_m"].to_numpy()
        self.ns, self.N = len(self.ri), n_cells

        faces = self.ri[:, None] + (self.ro - self.ri)[:, None] * np.linspace(0, 1, n_cells + 1)[None, :]
        self.rc = 0.5 * (faces[:, :-1] + faces[:, 1:])                 # cell centres
        self.V = 0.5 * (faces[:, 1:] ** 2 - faces[:, :-1] ** 2)        # volume per radian per metre
        self.G = self.k / np.log(self.rc[:, 1:] / self.rc[:, :-1])     # centre-to-centre conductance
        self.Ki = self.k / np.log(self.rc[:, 0] / self.ri)             # inner half cell
        self.Ko = self.k / np.log(self.ro / self.rc[:, -1])            # outer half cell
        self.Ts_in = flow["Taw"].copy() * 0.5                          # Newton warm starts
        self.Ts_out = np.full(self.ns, wall["initial_temperature_K"])

    def inner_surface(self, Tc):
        """Solve h_g(Ts) (Taw - Ts) r_in = Ki (Ts - Tc) for the hot-face temperature Ts."""
        f0, T0, X, Taw, ri, Ki = self.flow["h0"], self.flow["T0"], self.flow["X"], self.flow["Taw"], self.ri, self.Ki
        Ts = np.clip(self.Ts_in, Tc, Taw)
        for _ in range(30):
            s = bartz_sigma(Ts, T0, X)
            F = f0 * s * (Taw - Ts) * ri - Ki * (Ts - Tc)
            dF = f0 * ri * (dsigma_dTwg(Ts, T0, X) * (Taw - Ts) - s) - Ki
            step = F / dF
            Ts = Ts - step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.Ts_in = Ts
        h = f0 * bartz_sigma(Ts, T0, X)
        return Ts, h

    def outer_surface(self, Tc):
        """Solve Ko (Tc - To) = r_out [h_ext (To - Tb) + eps sigma (To^4 - Tsur^4)]."""
        hb, Tb = self.ext["h_W_m2K"], self.ext["bulk_temperature_K"]
        eps, Tsur = self.ext["emissivity"], self.ext["surroundings_temperature_K"]
        To = self.Ts_out.copy()
        for _ in range(30):
            F = self.Ko * (Tc - To) - self.ro * (hb * (To - Tb) + eps * SIGMA_SB * (To ** 4 - Tsur ** 4))
            dF = -self.Ko - self.ro * (hb + 4 * eps * SIGMA_SB * To ** 3)
            step = F / dF
            To = To - step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.Ts_out = To
        return To

    def fluxes(self, T):
        """Heat flows per radian per metre: into the inner cell and out of the outer cell."""
        Ts, h = self.inner_surface(T[:, 0])
        To = self.outer_surface(T[:, -1])
        Q_in = self.Ki * (Ts - T[:, 0])
        Q_out = self.Ko * (T[:, -1] - To)
        return Q_in, Q_out, Ts, To, h

    def rhs(self, t, y):
        T = y[:-2].reshape(self.ns, self.N)
        Q_in, Q_out, *_ = self.fluxes(T)
        Q = np.zeros((self.ns, self.N + 1))
        Q[:, 0] = Q_in
        Q[:, 1:-1] = self.G * (T[:, :-1] - T[:, 1:])
        Q[:, -1] = Q_out
        dT = (Q[:, :-1] - Q[:, 1:]) / (self.rho_c * self.V)
        return np.concatenate([dT.ravel(), [Q_in.sum(), Q_out.sum()]])

    def sparsity(self):
        n = self.ns * self.N
        S = lil_matrix((n + 2, n + 2), dtype=int)
        for s in range(self.ns):
            for j in range(self.N):
                i = s * self.N + j
                for jj in (j - 1, j, j + 1):
                    if 0 <= jj < self.N:
                        S[i, s * self.N + jj] = 1
            S[n, s * self.N] = 1
            S[n + 1, s * self.N + self.N - 1] = 1
        return S.tocsr()

    def stored_energy(self, T, T_ref):
        return float(np.sum(self.rho_c * self.V * (T - T_ref)))


def solve_transient(geo, flow, wall, ext, tr, n_cells=None, rtol=None, atol=None):
    n_cells = n_cells or tr["radial_cells"]
    model = TransientModel(geo, flow, wall, ext, n_cells)
    T_init = wall["initial_temperature_K"]
    y0 = np.concatenate([np.full(model.ns * n_cells, T_init), [0.0, 0.0]])
    t_end = tr["burn_duration_s"]
    t_eval = np.linspace(0.0, t_end, tr["output_times"])
    sol = solve_ivp(model.rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval,
                    rtol=rtol or tr["rtol"], atol=atol or tr["atol_K"],
                    jac_sparsity=model.sparsity())
    if not sol.success:
        raise RuntimeError(f"transient solve failed: {sol.message}")

    # post-process every output time (surface temperatures from the cell field)
    T_in_hist, T_out_hist = [], []
    for y in sol.y.T:
        _, _, Ts, To, _ = model.fluxes(y[:-2].reshape(model.ns, n_cells))
        T_in_hist.append(Ts)
        T_out_hist.append(To)
    T_in_hist, T_out_hist = np.array(T_in_hist), np.array(T_out_hist)

    T_end = sol.y[:-2, -1].reshape(model.ns, n_cells)
    _, _, Ts, To, h = model.fluxes(T_end)
    E_in, E_out = sol.y[-2, -1], sol.y[-1, -1]
    audit = abs(model.stored_energy(T_end, T_init) - (E_in - E_out)) / E_in
    return {"t": sol.t, "T_inner_hist": T_in_hist, "T_outer_hist": T_out_hist,
            "T_inner": Ts, "T_outer": To, "h": h, "q": h * (flow["Taw"] - Ts),
            "energy_audit_relative_error": audit, "nfev": sol.nfev}


# Outputs

def build_profile(geo, flow, steady, trans):
    return pd.DataFrame({
        "x_m": geo["x_m"], "r_inner_m": geo["r_inner_m"], "wall_thickness_m": geo["wall_thickness_m"],
        "area_ratio": flow["area_ratio"], "Mach": flow["M"], "pressure_Pa": flow["p"],
        "T_gas_static_K": flow["T"], "T_adiabatic_wall_K": flow["Taw"], "Reynolds": flow["Re"],
        "T_inner_K_steady": steady["T_inner"], "T_outer_K_steady": steady["T_outer"],
        "h_gas_W_m2K_steady": steady["h"], "heat_flux_inner_W_m2_steady": steady["q"],
        "T_inner_K_burn_end": trans["T_inner"], "T_outer_K_burn_end": trans["T_outer"],
        "h_gas_W_m2K_burn_end": trans["h"], "heat_flux_inner_W_m2_burn_end": trans["q"],
    })


def make_plots(df, flow, trans, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df["x_m"] * 1e3
    xt = x.iloc[flow["i_throat"]]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    a = ax[0, 0]
    ri, ro = df["r_inner_m"] * 1e3, (df["r_inner_m"] + df["wall_thickness_m"]) * 1e3
    a.fill_between(x, ri, ro, color="0.75", label="wall")
    a.fill_between(x, -ri, -ro, color="0.75")
    a.plot(x, ri, "k", x, -ri, "k", lw=1)
    a.set_aspect("equal"); a.set_xlabel("x (mm)"); a.set_ylabel("r (mm)")
    a.set_title("Nozzle contour")
    b = a.twinx(); b.plot(x, df["Mach"], color="tab:purple"); b.set_ylabel("Mach", color="tab:purple")

    a = ax[0, 1]
    a.plot(x, df["h_gas_W_m2K_steady"] / 1e3, "tab:red", label="h_g steady")
    a.plot(x, df["h_gas_W_m2K_burn_end"] / 1e3, "tab:orange", ls="--", label="h_g end of burn")
    a.set_xlabel("x (mm)"); a.set_ylabel("h_g (kW/m²K)"); a.grid(alpha=.3)
    b = a.twinx()
    b.plot(x, df["heat_flux_inner_W_m2_steady"] / 1e6, "tab:blue", label="q steady")
    b.plot(x, df["heat_flux_inner_W_m2_burn_end"] / 1e6, "tab:cyan", ls="--", label="q end of burn")
    b.set_ylabel("q (MW/m²)", color="tab:blue")
    a.set_title("Gas-side heat transfer coefficient and heat flux"
                + (f" (corrected, {flow['h_correction']})" if flow.get("h_correction") else " (Bartz)"))
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8)

    a = ax[1, 0]
    a.plot(x, df["T_adiabatic_wall_K"], "k:", label="Adiabatic wall T_aw")
    a.plot(x, df["T_inner_K_steady"], "tab:red", lw=2, label="Inner wall, steady")
    a.plot(x, df["T_outer_K_steady"], "tab:red", ls="--", label="Outer wall, steady")
    a.plot(x, df["T_inner_K_burn_end"], "tab:blue", lw=2, label="Inner wall, end of burn")
    a.plot(x, df["T_outer_K_burn_end"], "tab:blue", ls="--", label="Outer wall, end of burn")
    a.axvline(xt, color="0.6", lw=0.8)
    a.set_xlabel("x (mm)"); a.set_ylabel("Temperature (K)"); a.grid(alpha=.3)
    a.set_title("Wall temperature along the nozzle"); a.legend(fontsize=8)

    a = ax[1, 1]
    i = flow["i_throat"]
    a.plot(trans["t"], trans["T_inner_hist"][:, i], "tab:red", label="Throat inner wall")
    a.plot(trans["t"], trans["T_outer_hist"][:, i], "tab:blue", label="Throat outer wall")
    a.plot(trans["t"], trans["T_inner_hist"].max(axis=1), "tab:red", ls=":", label="Peak inner wall")
    a.axhline(df["T_inner_K_steady"].iloc[i], color="tab:red", lw=0.8, ls="--", label="Throat steady")
    a.set_xlabel("Time (s)"); a.set_ylabel("Temperature (K)"); a.grid(alpha=.3)
    a.set_title("Transient wall temperature"); a.legend(fontsize=8)

    fig.suptitle(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Verification

def run_verification(args, cfg, geo, flow, trans):
    gas, wall, ext, tr = cfg["gas"], cfg["wall"], cfg["external"], cfg["transient"]
    i = flow["i_throat"]
    out = {"radial_grid_study": []}
    for nc in (12, 24, 48):
        r = trans if nc == tr["radial_cells"] else solve_transient(geo, flow, wall, ext, tr, n_cells=nc)
        out["radial_grid_study"].append({
            "radial_cells": nc, "throat_inner_K": float(r["T_inner"][i]),
            "throat_outer_K": float(r["T_outer"][i]),
            "energy_audit_relative_error": float(r["energy_audit_relative_error"])})
    rs = {d["radial_cells"]: d["throat_inner_K"] for d in out["radial_grid_study"]}
    out["radial_24_vs_48_throat_difference_K"] = abs(rs[24] - rs[48])

    tight = solve_transient(geo, flow, wall, ext, tr, rtol=tr["rtol"] / 100, atol=tr["atol_K"] / 100)
    out["temporal_tolerance_difference_K"] = float(abs(tight["T_inner"][i] - trans["T_inner"][i]))

    out["axial_grid_study"] = []
    for f in (1, 2, 4):
        _, g2 = load_inputs(args.conditions, args.geometry, axial_refine=f)
        fl2 = compute_flow(g2, gas)
        st2 = solve_steady(g2, fl2, wall, ext)
        out["axial_grid_study"].append({
            "axial_stations": len(g2), "peak_inner_K": float(st2["T_inner"].max()),
            "peak_heat_flux_W_m2": float(st2["q"].max())})
    out["note"] = "Numerical verification only; no experimental validation"
    return out


def compare_reference(df, ref_path):
    ref = pd.read_csv(ref_path)
    print("\nComparison with reference profile (max |difference| over all stations):")
    for col in ref.columns:
        if col in df and len(ref) == len(df):
            d = np.max(np.abs(df[col].to_numpy() - ref[col].to_numpy()))
            rel = d / max(np.max(np.abs(ref[col].to_numpy())), 1e-30)
            print(f"  {col:32s} {d:12.4g}   ({rel:.2e} relative)")


def main():
    ap = argparse.ArgumentParser(description="Bartz nozzle wall temperature calculator")
    ap.add_argument("--conditions", default="operating_conditions.json")
    ap.add_argument("--geometry", default="nozzle_geometry.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--verify", action="store_true", help="run grid/tolerance/energy checks")
    ap.add_argument("--reference", help="reference wall_temperature_profile.csv to compare against")
    args = ap.parse_args()

    cfg, geo = load_inputs(args.conditions, args.geometry)
    gas, wall, ext, tr = cfg["gas"], cfg["wall"], cfg["external"], cfg["transient"]
    os.makedirs(args.out, exist_ok=True)

    flow = compute_flow(geo, gas)
    for w in flow["warnings"]:
        print("WARNING:", w)
    if flow["h_correction"]:
        print(f"Applying per-station h_g correction '{flow['h_correction']}' "
              f"from {gas['h_correction_csv']}")
    steady = solve_steady(geo, flow, wall, ext)
    trans = solve_transient(geo, flow, wall, ext, tr)

    df = build_profile(geo, flow, steady, trans)
    df.to_csv(os.path.join(args.out, "wall_temperature_profile.csv"), index=False)
    hist = pd.DataFrame({
        "t_s": trans["t"],
        "T_inner_throat_K": trans["T_inner_hist"][:, flow["i_throat"]],
        "T_outer_throat_K": trans["T_outer_hist"][:, flow["i_throat"]],
        "T_inner_peak_K": trans["T_inner_hist"].max(axis=1),
        "x_peak_m": geo["x_m"].to_numpy()[trans["T_inner_hist"].argmax(axis=1)]})
    hist.to_csv(os.path.join(args.out, "wall_temperature_history.csv"), index=False)
    make_plots(df, flow, trans, os.path.join(args.out, "nozzle_wall_temperature.png"),
               cfg.get("case_label", ""))

    i = flow["i_throat"]
    js, jt = int(np.argmax(steady["T_inner"])), int(np.argmax(trans["T_inner"]))
    print(f"Case: {cfg.get('case_label', '')}")
    print(f"  Throat at x = {geo['x_m'][i]*1e3:.1f} mm, Dt = {flow['Dt']*1e3:.2f} mm, "
          f"exit A/At = {flow['area_ratio'][-1]:.2f}, exit Mach = {flow['M'][-1]:.3f}")
    print(f"  Throat: Taw {flow['Taw'][i]:.0f} K, h_g steady {steady['h'][i]:.0f} W/m2K")
    print(f"  STEADY       peak inner wall {steady['T_inner'][js]:.1f} K at x = {geo['x_m'][js]*1e3:.1f} mm, "
          f"peak q {steady['q'].max()/1e6:.3f} MW/m2")
    print(f"  END OF BURN  ({tr['burn_duration_s']} s) peak inner wall {trans['T_inner'][jt]:.1f} K "
          f"at x = {geo['x_m'][jt]*1e3:.1f} mm; throat inner/outer "
          f"{trans['T_inner'][i]:.1f} / {trans['T_outer'][i]:.1f} K")
    print(f"  Energy audit relative error: {trans['energy_audit_relative_error']:.2e}")

    if args.verify:
        ver = run_verification(args, cfg, geo, flow, trans)
        with open(os.path.join(args.out, "verification.json"), "w") as f:
            json.dump(ver, f, indent=2)
        print("\nVerification:\n" + json.dumps(ver, indent=2))
    if args.reference:
        compare_reference(df, args.reference)
    print(f"\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()