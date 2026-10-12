"""Fit Bartz correction factors to the Back, Massier & Gier data.

Response: y = ln(h_measured / h_Bartz), evaluated at the measured wall temperature.
Correction C = exp(fitted y), applied as h_corrected = C * h_Bartz
(i.e. a bartz_multiplier, optionally varying along the nozzle).

Training set: turbulent tests (pt >= 100 psia) whose rows pass the transcription
consistency check. Tests 234 (75 psia) and 301 (45 psia) are held out as an
out-of-regime check, since Back et al. report reduced heat transfer below ~75 psia.

Uncertainty: points from one test share conditions, instrumentation and inlet
boundary layer, so they are not independent. Confidence intervals therefore
come from a cluster bootstrap (resampling whole tests), and model choice from
leave-one-test-out cross-validation.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from predict_bartz import predict, RC_DEFAULT, IN, area_ratio_at

rng = np.random.default_rng(20261013)
B = 4000


def features(df, name):
    s = np.where(df.supersonic, 1, -1) * np.log(df.area_ratio)  # signed log area ratio
    one = np.ones(len(df))
    if name == "M0 uncorrected":
        return np.empty((len(df), 0))
    if name == "M1 constant":
        return one[:, None]
    if name == "M2 piecewise-linear in ln(A/A*)":
        return np.column_stack([one, np.minimum(s, 0), np.maximum(s, 0)])
    if name == "M3 quadratic in signed ln(A/A*)":
        return np.column_stack([one, s, s ** 2])
    if name == "M4 = M2 + inlet BL (l=0)":
        return np.column_stack([one, np.minimum(s, 0), np.maximum(s, 0),
                                (df.approach_length_in == 0).astype(float)])
    if name == "M5 = M2 + ln(pt)":
        return np.column_stack([one, np.minimum(s, 0), np.maximum(s, 0),
                                np.log(df.pt_Pa / 1e6)])
    raise KeyError(name)


MODELS = ["M0 uncorrected", "M1 constant", "M2 piecewise-linear in ln(A/A*)",
          "M3 quadratic in signed ln(A/A*)", "M4 = M2 + inlet BL (l=0)", "M5 = M2 + ln(pt)"]


def fit(df, name):
    X = features(df, name)
    y = np.log(df.ratio_h.values)
    if X.shape[1] == 0:
        return np.empty(0)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def pred(df, name, beta):
    X = features(df, name)
    return X @ beta if X.shape[1] else np.zeros(len(df))


def loto_cv(df, name):
    """Leave-one-test-out: RMSE of ln(h_meas/h_pred), pooled over held-out points."""
    errs = []
    for t in df.test.unique():
        tr, te = df[df.test != t], df[df.test == t]
        beta = fit(tr, name)
        errs.append(np.log(te.ratio_h.values) - pred(te, name, beta))
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))


def cluster_bootstrap(df, name, Xgrid_df=None):
    tests = df.test.unique()
    groups = {t: df[df.test == t] for t in tests}
    betas, curves = [], []
    for _ in range(B):
        pick = rng.choice(tests, size=len(tests), replace=True)
        boot = pd.concat([groups[t] for t in pick])
        beta = fit(boot, name)
        betas.append(beta)
        if Xgrid_df is not None:
            curves.append(np.exp(pred(Xgrid_df, name, beta)))
    return np.array(betas), (np.array(curves) if curves else None)


def grid_frame(n=300):
    zl = np.linspace(0.12, 0.95, n)
    return pd.DataFrame(dict(z_over_L=zl, area_ratio=area_ratio_at(zl), supersonic=zl > 0.603,
                             approach_length_in=18, pt_Pa=1e6))


def main():
    raw = pd.read_csv("bmg_heat_transfer_data.csv")
    raw = raw[raw.qa_pass]
    res = predict(raw)
    res.to_csv("bmg_vs_bartz.csv", index=False)
    train = res[res.pt_Pa >= 99 * 6894.757].copy()
    held = res[res.pt_Pa < 99 * 6894.757].copy()

    summary = {"n_points_train": int(len(train)), "n_tests_train": int(train.test.nunique()),
               "tests_train": sorted(map(int, train.test.unique())),
               "tests_heldout": sorted(map(int, held.test.unique())), "models": {}}

    # Raw discrepancy
    r = train.ratio_h
    summary["raw"] = {
        "h_meas_over_bartz_median": float(r.median()),
        "h_meas_over_bartz_min": float(r.min()), "h_meas_over_bartz_max": float(r.max()),
        "bartz_overprediction_median_pct": float((1 / r.median() - 1) * 100),
        "q_meas_over_bartz_median": float(train.ratio_q.median()),
        "by_region_median_ratio": {
            "convergent (A/A*>1.5)": float(train[(~train.supersonic) & (train.area_ratio > 1.5)].ratio_h.median()),
            "throat region (A/A*<=1.5)": float(train[train.area_ratio <= 1.5].ratio_h.median()),
            "divergent (A/A*>1.5)": float(train[(train.supersonic) & (train.area_ratio > 1.5)].ratio_h.median()),
        },
    }

    for name in MODELS:
        rmse, mae = loto_cv(train, name)
        beta = fit(train, name)
        entry = {"loto_rmse_log": rmse, "loto_mae_log": mae,
                 "loto_rmse_pct_approx": (np.exp(rmse) - 1) * 100,
                 "coefficients": beta.tolist()}
        if len(beta):
            betas, _ = cluster_bootstrap(train, name)
            lo, hi = np.percentile(betas, [2.5, 97.5], axis=0)
            entry["coef_ci95"] = list(zip(lo.tolist(), hi.tolist()))
        summary["models"][name] = entry

    # Global multiplier with CI
    b1, _ = cluster_bootstrap(train, "M1 constant")
    c = float(np.exp(fit(train, "M1 constant")[0]))
    summary["global_multiplier"] = {"C": c, "ci95": np.exp(np.percentile(b1[:, 0], [2.5, 97.5])).tolist()}

    # Chosen position-dependent model: M2 (see CV table). Curve + band on a grid.
    chosen = "M2 piecewise-linear in ln(A/A*)"
    grid = grid_frame()
    beta2 = fit(train, chosen)
    curve = np.exp(pred(grid, chosen, beta2))
    _, curves = cluster_bootstrap(train, chosen, grid)
    band = np.percentile(curves, [2.5, 97.5], axis=0)
    # Throat value and CI
    throat = pd.DataFrame(dict(area_ratio=[1.0], supersonic=[False]))
    cth = np.exp(pred(throat, chosen, beta2))[0]
    b2, thr = cluster_bootstrap(train, chosen, throat)
    summary["position_model"] = {
        "name": chosen,
        "form": "ln C = a + b_sub*min(s,0) + b_sup*max(s,0),  s = -ln(A/A*) subsonic, +ln(A/A*) supersonic",
        "a": beta2[0], "b_sub": beta2[1], "b_sup": beta2[2],
        "C_throat": float(cth), "C_throat_ci95": np.percentile(thr[:, 0], [2.5, 97.5]).tolist(),
    }
    for A, sup in [(4.0, False), (2.0, False), (2.0, True), (2.5, True)]:
        f = pd.DataFrame(dict(area_ratio=[A], supersonic=[sup]))
        _, cc = cluster_bootstrap(train, chosen, f)
        summary["position_model"][f"C_at_A={A}_{'sup' if sup else 'sub'}"] = {
            "C": float(np.exp(pred(f, chosen, beta2))[0]),
            "ci95": np.percentile(cc[:, 0], [2.5, 97.5]).tolist()}

    # Held-out (low pressure) check: how far below the turbulent calibration do they fall?
    held_pred = np.exp(pred(held, chosen, beta2))
    held["ratio_to_calibrated"] = held.ratio_h / held_pred
    summary["heldout"] = held.groupby("test").ratio_to_calibrated.median().to_dict()
    summary["heldout"] = {int(k): float(v) for k, v in summary["heldout"].items()}

    # Residuals vs acceleration parameter K, after the position model
    train["resid_M2"] = np.log(train.ratio_h) - pred(train, chosen, beta2)
    summary["corr_resid_vs_logK_convergent"] = float(np.corrcoef(
        np.log(train.loc[~train.supersonic, "accel_K"]), train.loc[~train.supersonic, "resid_M2"])[0, 1])
    summary["accel_K_max"] = float(res.accel_K.max())

    # Throat-curvature sensitivity: h_Bartz ~ rc^-0.1, so C scales by (rc/rc0)^0.1
    sens = {}
    for rc_in in [0.45, 0.625, 0.9, 1.8]:
        rr = predict(train, rc=rc_in * IN)
        sens[f"rc={rc_in}in (rc/rt={rc_in / 0.9015:.2f})"] = float(np.exp(np.mean(np.log(rr.ratio_h))))
    summary["rc_sensitivity_global_C"] = sens

    with open("calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    plot(train, held, grid, curve, band, c, summary)
    return summary


def plot(train, held, grid, curve, band, c, summary):
    ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    blue, orange = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "figure.facecolor": surf, "axes.facecolor": surf})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    ax.fill_between(grid.z_over_L, band[0], band[1], color=ink2, alpha=0.15, lw=0,
                    label="95% CI (cluster bootstrap)")
    ax.plot(grid.z_over_L, curve, color=ink, lw=2, label="Fitted correction C(A/A*)")
    ax.axhline(c, color=ink2, lw=1, ls="--", label=f"Single multiplier C = {c:.2f}")
    ax.axhline(1, color=ink2, lw=0.8, ls=":")
    ax.text(0.125, 1.01, "Uncorrected Bartz (C = 1)", color=ink2, fontsize=8, va="bottom")
    for grp, col, lab in [(train.approach_length_in == 18, blue, "Thick inlet BL (18-in approach), 100–254 psia"),
                          (train.approach_length_in == 0, orange, "Thin inlet BL (no approach), 101–202 psia")]:
        d = train[grp]
        ax.scatter(d.z_over_L, d.ratio_h, s=18, color=col, edgecolor=surf, lw=0.8, label=lab, zorder=3)
    d = held
    ax.scatter(d.z_over_L, d.ratio_h, s=22, marker="x", color=ink2, lw=1,
               label="Held out: 45 & 75 psia (not fitted)", zorder=3)
    ax.axvline(0.603, color=ink2, lw=0.8, ls=":")
    ax.text(0.607, 1.12, "throat", color=ink2, fontsize=8)
    ax.set(xlabel="Axial position z/L", ylabel="h measured / h Bartz", ylim=(0.3, 1.2), xlim=(0.1, 0.96))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    ax.set_title("Bartz overpredicts h most around and after the throat", loc="left", fontsize=10, color=ink)

    ax = axes[1]
    names = [m for m in summary["models"]]
    vals = [(np.exp(summary["models"][m]["loto_rmse_log"]) - 1) * 100 for m in names]
    short = ["Uncorrected", "Single multiplier", "Piecewise in A/A*  (chosen)",
             "Quadratic in A/A*", "Piecewise + inlet BL", "Piecewise + pressure"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=[ink2 if i == 0 else blue for i in range(len(vals))], height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=8, color=ink)
    ax.set_yticks(y, short)
    ax.set(xlabel="Typical error on unseen tests (%)", xlim=(0, max(vals) * 1.2))
    ax.grid(alpha=0.25, axis="x")
    ax.set_title("Leave-one-test-out cross-validation", loc="left", fontsize=10, color=ink)
    fig.savefig("bartz_calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k != "models"}, indent=1, default=float))
    for m, e in s["models"].items():
        print(f"{m:38s} LOTO RMSE(log)={e['loto_rmse_log']:.3f}  coef={np.round(e['coefficients'], 3)}")
