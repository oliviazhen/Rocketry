"""Run Matthew's bartz_nozzle.NozzleModel at each Back, Massier & Gier test condition.

Bartz h is evaluated at the measured wall temperature at each plug, so the
comparison isolates the gas-side correlation (no wall-conduction model involved).

Geometry: wall area ratios from JPL TR 32-415 Table B-2 (32 static taps), plus
the throat (z/L = 0.603) and the inlet/exit area ratios (7.75, 2.68) from the
report text. Throat diameter 1.803 in, nozzle length L = 5.925 in.
Throat radius of curvature: 0.625 in (rc/rt ~ 0.69) as commonly used for this
nozzle in CFD validation studies; sensitivity is tested in calibrate.py.

Gas: methanol-heated air, treated as air (as in the report). gamma from the
report (1.380 at 1030 R, 1.345 at 1500 R), cp, mu, Pr from CoolProp at Tt.
"""
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
from bartz_nozzle import Geometry, NozzleModel

IN = 0.0254
L = 5.925 * IN
RT = 1.803 / 2 * IN
RC_DEFAULT = 0.625 * IN

# Table B-2 wall area ratios (z/L, A/A*), with inlet, throat and exit added.
B2 = np.array([
    (0.0, 7.75), (0.107, 7.001), (0.222, 4.691), (0.3396, 2.939), (0.3401, 2.924),
    (0.395, 2.255), (0.455, 1.631), (0.460, 1.561), (0.506, 1.235), (0.529, 1.135),
    (0.554, 1.054), (0.587, 1.0078), (0.6018, 1.0012), (0.603, 1.0), (0.642, 1.028),
    (0.691, 1.176), (0.706, 1.231), (0.722, 1.297), (0.741, 1.372), (0.753, 1.423),
    (0.771, 1.505), (0.789, 1.582), (0.802, 1.642), (0.819, 1.716), (0.838, 1.815),
    (0.852, 1.866), (0.869, 1.960), (0.887, 2.049), (0.900, 2.112), (0.921, 2.210),
    (0.932, 2.283), (0.949, 2.379), (0.965, 2.471), (0.985, 2.574), (1.0, 2.68)])


def area_ratio_at(zl):
    """Interpolate radius linearly between taps (area ratio = (r/rt)^2)."""
    return np.interp(zl, B2[:, 0], np.sqrt(B2[:, 1])) ** 2


def gamma_at(Tt_K):
    T = np.array([1030, 1500, 2000]) * 5 / 9
    return float(np.interp(Tt_K, T, [1.380, 1.345, 1.328]))


def air_props(Tt_K):
    p = 101325.0
    return dict(cp=CP.PropsSI("C", "T", Tt_K, "P", p, "Air"),
                mu=CP.PropsSI("V", "T", Tt_K, "P", p, "Air"),
                Pr=CP.PropsSI("Prandtl", "T", Tt_K, "P", p, "Air"))


def ideal_cstar(Tt_K, gamma, R=287.05):
    return np.sqrt(R * Tt_K / gamma) * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))


def build_model(pt, Tt, rc=RC_DEFAULT, zl=None):
    zl = np.unique(zl)
    ratio = area_ratio_at(zl)
    geometry = Geometry(x=zl * L, ri=RT * np.sqrt(ratio), thickness=np.full(zl.size, 0.01))
    gam, props = gamma_at(Tt), air_props(Tt)
    config = {
        "gas": {"chamber_pressure_Pa": pt, "stagnation_temperature_K": Tt, "gamma": gam,
                "cp_J_kgK": props["cp"], "viscosity_Pa_s": props["mu"], "Prandtl": props["Pr"],
                "c_star_m_s": ideal_cstar(Tt, gam), "throat_curvature_radius_m": rc},
        # Wall/external entries are required by the constructor but unused here.
        "wall": {"conductivity_W_mK": 20, "density_kg_m3": 7800, "cp_J_kgK": 500,
                 "initial_temperature_K": 300},
        "external": {"h_W_m2K": 1000, "bulk_temperature_K": 300,
                     "surroundings_temperature_K": 300, "emissivity": 0},
    }
    return NozzleModel(geometry, config), zl


def predict(df, rc=RC_DEFAULT):
    out = df.copy()
    for col in ["area_ratio", "Mach", "h_bartz_W_m2K", "Taw_model_K", "cstar_ideal_m_s",
                "cstar_measured_m_s", "Re_D", "accel_K"]:
        out[col] = np.nan
    stations = np.array(sorted(set(df.z_over_L)))
    for test, g in df.groupby("test"):
        pt, Tt = g.pt_Pa.iloc[0], g.Tt_K.iloc[0]
        model, zl = build_model(pt, Tt, rc, stations)
        # Acceleration parameter K = (nu/u^2) du/dx with static properties (1-D flow).
        R = model.cp_gas * (model.gamma - 1) / model.gamma
        T = model.Tstatic
        u = model.M * np.sqrt(model.gamma * R * T)
        rho = model.pstatic / (R * T)
        mu_s = np.array([CP.PropsSI("V", "T", t, "P", 101325.0, "Air") for t in T])
        K = mu_s / rho / u ** 2 * np.gradient(u, model.geometry.x)
        for idx, row in g.iterrows():
            i = int(np.searchsorted(zl, row.z_over_L))
            h = model.bartz_h(np.full(zl.size, row.Tw_K))[i]
            out.loc[idx, ["area_ratio", "Mach", "h_bartz_W_m2K", "Taw_model_K", "Re_D", "accel_K"]] = [
                model.ratio[i], model.M[i], h, model.Taw[i], model.Re[i], K[i]]
            out.loc[idx, "cstar_ideal_m_s"] = model.cstar
            out.loc[idx, "cstar_measured_m_s"] = pt * model.At / row.mdot_kg_s
    out["supersonic"] = out.z_over_L > 0.603
    out["ratio_h"] = out.h_exp_W_m2K / out.h_bartz_W_m2K
    # Heat-flux ratio using the model's own Taw (what the code would predict at that Tw)
    out["q_bartz_W_m2"] = out.h_bartz_W_m2K * (out.Taw_model_K - out.Tw_K)
    out["ratio_q"] = out.qw_W_m2 / out.q_bartz_W_m2
    return out


if __name__ == "__main__":
    df = pd.read_csv("bmg_heat_transfer_data.csv")
    res = predict(df)
    res.to_csv("bmg_vs_bartz.csv", index=False)
    print(res.groupby("test")[["cstar_ideal_m_s", "cstar_measured_m_s"]].first().round(1).to_string())
    print(res.groupby("test").ratio_h.describe()[["mean", "min", "max"]].round(3).to_string())
