"""Uncertainty quantification and global sensitivity analysis for bartz_nozzle.py.

Turns "the peak inner wall is 1786.54 K" into "median X K, 90% interval A-B K,
P(T > limit) = p, and this input drives most of the spread".

Pipeline
1. Input uncertainty is declared in a JSON spec (``uq_inputs.json``): one
   distribution per uncertain config field, each with its rationale.
2. Steady Monte Carlo (Latin hypercube): distribution of peak wall temperature,
   probability of exceeding material limits, axial uncertainty band.
3. Steady Sobol indices (Saltelli sampling + SALib): first-order (S1) and total
   (ST) contribution of each input to the variance of the peak temperature.
4. Transient: the burn-end solve is ~50x slower, so a Gaussian-process
   surrogate is trained on a few hundred solves, checked on a held-out set,
   then used for the same Monte Carlo and Sobol analysis.

Run from the example folder (same layout as the solver):
    python ../uq_nozzle.py --geometry nozzle_geometry.csv --config operating_conditions.json \
        --spec uq_inputs.json --output uq_results

All sampling is done in the unit hypercube and mapped to physical values with
each distribution's inverse CDF, so LHS, Sobol and the GP share one input space.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bartz_nozzle import NozzleModel, RadialWall, load_geometry  # noqa: E402
from nozzle_config import NozzleConfig, load_config  # noqa: E402

U_CLIP = 1e-6  # keeps unbounded inverse CDFs finite at Sobol-sequence endpoints



# Input-uncertainty specification
class _Parameter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    path: str = Field(pattern=r"^(gas|wall|external)\.[A-Za-z_][A-Za-z0-9_]*$",
                      description="section.key in operating_conditions.json")
    rationale: str = ""

    @property
    def section_key(self) -> tuple[str, str]:
        section, key = self.path.split(".")
        return section, key


class Uniform(_Parameter):
    distribution: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def _order(self):
        if not self.high > self.low:
            raise ValueError(f"{self.name}: high must exceed low")
        return self

    def ppf(self, u):
        return self.low + (self.high - self.low) * np.asarray(u)


class LogNormal(_Parameter):
    distribution: Literal["lognormal"]
    median: float = Field(gt=0)
    sigma_ln: float = Field(gt=0, le=2, description="standard deviation of ln(value)")

    def ppf(self, u):
        return self.median * np.exp(self.sigma_ln * stats.norm.ppf(u))


class Normal(_Parameter):
    distribution: Literal["normal"]
    mean: float
    sd: float = Field(gt=0)

    def ppf(self, u):
        return self.mean + self.sd * stats.norm.ppf(u)


class Triangular(_Parameter):
    distribution: Literal["triangular"]
    low: float
    mode: float
    high: float

    @model_validator(mode="after")
    def _order(self):
        if not self.low <= self.mode <= self.high or self.low == self.high:
            raise ValueError(f"{self.name}: need low <= mode <= high and low < high")
        return self

    def ppf(self, u):
        width = self.high - self.low
        return stats.triang(c=(self.mode - self.low) / width, loc=self.low, scale=width).ppf(u)


Parameter = Annotated[Union[Uniform, LogNormal, Normal, Triangular], Field(discriminator="distribution")]


class UQSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    description: str = ""
    parameters: list[Parameter] = Field(min_length=1)
    material_limits_K: list[Annotated[float, Field(gt=0)]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self):
        for attr in ("name", "path"):
            values = [getattr(p, attr) for p in self.parameters]
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate parameter {attr}s: {values}")
        return self

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def dim(self) -> int:
        return len(self.parameters)

    def to_physical(self, u: np.ndarray) -> np.ndarray:
        """Map unit-hypercube samples (n, d) to physical input values (n, d)."""
        u = np.clip(np.atleast_2d(u), U_CLIP, 1 - U_CLIP)
        return np.column_stack([p.ppf(u[:, i]) for i, p in enumerate(self.parameters)])

    def to_features(self, u: np.ndarray) -> np.ndarray:
        """Surrogate input coordinates in which the response is smooth.

        Unbounded inputs (normal, lognormal) use the standard-normal score z,
        which is linear in the value (or its logarithm), scaled so +/-3 sd maps
        to [0, 1]. Bounded inputs keep u. Training a GP directly on u puts a
        near-vertical cliff at the tails of every unbounded input (held-out max
        error 69 K vs 1.3 K on this case).
        """
        u = np.clip(np.atleast_2d(u), U_CLIP, 1 - U_CLIP)
        cols = [stats.norm.ppf(u[:, i]) / 6 + 0.5 if p.distribution in ("normal", "lognormal") else u[:, i]
                for i, p in enumerate(self.parameters)]
        return np.column_stack(cols)


def load_spec(path) -> UQSpec:
    return UQSpec.model_validate_json(Path(path).read_text())


# Model wrapper

STEADY_OUTPUTS = ["peak_inner_K", "peak_inner_x_m", "peak_outer_K", "peak_heat_flux_MW_m2", "throat_inner_K"]
TRANSIENT_OUTPUTS = ["burn_end_peak_inner_K", "burn_end_peak_outer_K", "burn_end_throat_inner_K"]


class NozzleUQ:
    """Evaluates the unchanged solver for one vector of uncertain inputs."""

    def __init__(self, geometry, base: NozzleConfig, spec: UQSpec):
        self.geometry, self.base, self.spec = geometry, base, spec
        data = base.model_dump()
        for p in spec.parameters:
            section, key = p.section_key
            if key not in data[section]:
                raise ValueError(f"{p.path} is not a field of the configuration")
            if isinstance(data[section][key], list):
                raise ValueError(f"{p.path} is a per-station profile; only scalar fields can be sampled")
        self.x_m = geometry.x

    def config_for(self, values) -> dict:
        updates: dict[str, dict] = {}
        for p, value in zip(self.spec.parameters, values):
            section, key = p.section_key
            updates.setdefault(section, {})[key] = float(value)
        return self.base.with_updates(**updates).to_solver_dict()  # re-validated every time

    def steady(self, values):
        model = NozzleModel(self.geometry, self.config_for(values))
        s = model.steady()
        i = int(np.argmax(s["T_inner_K"]))
        scalars = [s["T_inner_K"][i], self.x_m[i], np.max(s["T_outer_K"]),
                   np.max(s["heat_flux_inner_W_m2"]) / 1e6, s["T_inner_K"][model.throat]]
        return np.array(scalars), s["T_inner_K"]

    def transient(self, values):
        config = self.config_for(values)
        model = NozzleModel(self.geometry, config)
        t = config["transient"]
        run = RadialWall(model, t["radial_cells"]).transient(
            t["burn_duration_s"], outputs=2, rtol=t["rtol"], atol=t["atol_K"])
        inner, outer = run["T_inner_K"][-1], run["T_outer_K"][-1]
        return np.array([inner.max(), outer.max(), inner[model.throat]])


_WORKER: Optional[NozzleUQ] = None


def _init_worker(problem):
    global _WORKER
    _WORKER = problem


def _steady_task(values):
    return _WORKER.steady(values)


def _transient_task(values):
    return _WORKER.transient(values)


def evaluate(problem: NozzleUQ, X: np.ndarray, kind: str, workers: int):
    """Run the solver for every row of X, in parallel when workers > 1."""
    task = _steady_task if kind == "steady" else _transient_task
    if workers <= 1:
        _init_worker(problem)
        results = [task(row) for row in X]
    else:
        chunk = max(1, len(X) // (workers * 8))
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(problem,)) as pool:
            results = list(pool.map(task, X, chunksize=chunk))
    if kind == "steady":
        return np.array([r[0] for r in results]), np.array([r[1] for r in results])
    return np.array(results)



# Statistics
def lhs(spec: UQSpec, n: int, seed: int) -> np.ndarray:
    return qmc.LatinHypercube(d=spec.dim, seed=seed).random(n)


def distribution_summary(y, limits, seed, n_boot=2000):
    """Quantiles with bootstrap 95% CIs, and exceedance probabilities with exact binomial CIs."""
    y = np.asarray(y, dtype=float)
    probs = [0.05, 0.5, 0.95]
    rng = np.random.default_rng(seed)
    boot = np.quantile(rng.choice(y, size=(n_boot, len(y)), replace=True), probs, axis=1)
    quantiles = {f"p{int(q * 100):02d}_K": {"value": float(np.quantile(y, q)),
                                         "ci95": [float(v) for v in np.percentile(boot[k], [2.5, 97.5])]}
                 for k, q in enumerate(probs)}
    exceed = {}
    for limit in limits:
        k, n = int(np.sum(y > limit)), len(y)
        low = 0.0 if k == 0 else float(stats.beta.ppf(0.025, k, n - k + 1))
        high = 1.0 if k == n else float(stats.beta.ppf(0.975, k + 1, n - k))
        exceed[f"{limit:g}"] = {"probability": k / n, "ci95": [low, high], "count": k}
    return {"n": int(len(y)), "mean_K": float(y.mean()), "sd_K": float(y.std(ddof=1)),
            "quantiles": quantiles, "exceedance": exceed}


def sobol_indices(spec: UQSpec, model_fn, n_base: int, seed: int):
    """Saltelli sampling + SALib Sobol analysis. model_fn maps unit samples (m, d) -> outputs (m, k)."""
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    problem = {"num_vars": spec.dim, "names": spec.names, "bounds": [[0.0, 1.0]] * spec.dim}
    U = sobol_sample.sample(problem, n_base, calc_second_order=False, seed=seed)
    Y = np.atleast_2d(model_fn(U).T).T
    results = []
    for column in range(Y.shape[1]):
        si = sobol_analyze.analyze(problem, Y[:, column], calc_second_order=False,
                                   seed=seed, print_to_console=False)
        results.append({name: {"S1": float(si["S1"][i]), "S1_ci95": float(si["S1_conf"][i]),
                               "ST": float(si["ST"][i]), "ST_ci95": float(si["ST_conf"][i])}
                        for i, name in enumerate(spec.names)})
    return results, U, Y


def ranking(indices: dict) -> list[str]:
    return sorted(indices, key=lambda name: indices[name]["ST"], reverse=True)


# Gaussian-process surrogate for the transient solve
class TransientSurrogate:
    """GP surrogate taking unit-hypercube inputs; one independent GP per output.

    ``features`` maps unit samples to the coordinates the GP is trained in
    (use ``UQSpec.to_features``); identity if omitted.
    """

    def __init__(self, seed=0, restarts=4, features=None):
        self.seed, self.restarts, self.models = seed, restarts, []
        self.features = features or (lambda U: np.atleast_2d(U))

    def fit(self, U, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        U = self.features(U)
        d = U.shape[1]
        self.models = []
        for column in np.atleast_2d(Y.T):
            kernel = (ConstantKernel(1.0, (1e-3, 1e5))
                      * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                      + WhiteKernel(1e-6, (1e-12, 1e-1)))
            gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=self.restarts,
                                          random_state=self.seed)
            self.models.append(gp.fit(U, column))
        return self

    def predict(self, U, return_std=False):
        U = self.features(U)
        out = [gp.predict(U, return_std=True) for gp in self.models]
        mean = np.column_stack([m for m, _ in out])
        return (mean, np.column_stack([s for _, s in out])) if return_std else mean

    def validate(self, U, Y):
        mean, sd = self.predict(U, return_std=True)
        metrics = []
        for k in range(Y.shape[1]):
            err = mean[:, k] - Y[:, k]
            metrics.append({"rmse_K": float(np.sqrt(np.mean(err ** 2))),
                            "max_abs_error_K": float(np.max(np.abs(err))),
                            "r2": float(1 - np.sum(err ** 2) / np.sum((Y[:, k] - Y[:, k].mean()) ** 2)),
                            "coverage_95": float(np.mean(np.abs(err) <= 1.96 * sd[:, k])),
                            "kernel": str(self.models[k].kernel_)})
        return metrics, mean, sd


# Plots
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#86b6ef", "#eb6834"


def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def make_plots(folder, spec, x_m, profiles, steady_peak, transient_peak, baseline, sobol_steady,
               sobol_transient, gp_test, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    folder = Path(folder)

    # 1. Peak-temperature distribution and exceedance curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    series = [("Steady peak", steady_peak, BLUE, baseline["steady_peak_inner_K"])]
    if transient_peak is not None:
        series.append(("Burn-end peak (GP)", transient_peak, ORANGE, baseline["transient_peak_inner_K"]))
    lo = min(np.min(s[1]) for s in series)
    hi = max(np.max(s[1]) for s in series)
    bins = np.linspace(lo, hi, 70)
    for label, y, color, base in series:
        p5, p50, p95 = np.quantile(y, [0.05, 0.5, 0.95])
        ax1.hist(y, bins=bins, color=color, alpha=0.55, density=True, edgecolor=SURFACE, linewidth=0.4,
                 label=f"{label}: median {p50:.0f} K, 90% {p5:.0f}-{p95:.0f} K")
        ax1.axvline(p50, color=color, linewidth=1.6)
        ax1.axvline(base, color=color, linewidth=1.2, linestyle=":")
        grid = np.linspace(lo - 50, hi + 50, 400)
        ax2.plot(grid, [np.mean(y > g) for g in grid], color=color, linewidth=2, label=label)
    for limit in spec.material_limits_K:
        ax2.axvline(limit, color=INK2, linewidth=0.8, linestyle="--")
        ax2.text(limit, 1.015, f"{limit:g} K", ha="center", va="bottom", fontsize=7, color=INK2,
                 transform=ax2.get_xaxis_transform())
    ax1.set(xlabel="Peak inner-wall temperature [K]", ylabel="Probability density")
    ax1.set_title("Distribution (solid = median, dotted = deterministic run)", pad=16)
    ax1.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.25)
    ax2.set(xlabel="Material limit [K]", ylabel="P(peak inner wall > limit)", ylim=(-0.02, 1.02))
    ax2.set_title("Exceedance probability", pad=16)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=10, color=INK)
    fig.savefig(folder / "uq_peak_temperature.png", dpi=180)
    plt.close(fig)

    # 2. Sobol indices (same parameter order in every panel, sorted by steady ST)
    panels = [("Steady peak inner wall", sobol_steady)]
    if sobol_transient is not None:
        panels.append(("Burn-end peak inner wall (GP)", sobol_transient))
    order = ranking(sobol_steady)[::-1]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.4), layout="constrained", sharey=True)
    for ax, (label, idx) in zip(np.atleast_1d(axes), panels):
        ypos = np.arange(len(order))
        st = [idx[n]["ST"] for n in order]
        s1 = [idx[n]["S1"] for n in order]
        ax.barh(ypos + 0.19, st, height=0.36, color=BLUE, xerr=[idx[n]["ST_ci95"] for n in order],
                error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2}, label="Total effect ST")
        ax.barh(ypos - 0.19, s1, height=0.36, color=BLUE_LIGHT, xerr=[idx[n]["S1_ci95"] for n in order],
                error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 2}, label="First order S1")
        for y, name in zip(ypos, order):
            edge = max(idx[name]["ST"], 0) + idx[name]["ST_ci95"]
            ax.text(edge + 0.02, y + 0.19, f"{idx[name]['ST']:.2f}", va="center", fontsize=7, color=INK)
        ax.set_yticks(ypos, order)
        ax.set(xlim=(-0.02, 1.05), xlabel="Share of output variance", title=label)
        ax.grid(axis="y", visible=False)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(folder / "uq_sobol_indices.png", dpi=180)
    plt.close(fig)

    # 3. Axial uncertainty band (steady)
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    x = np.asarray(x_m) * 1000
    p5, p25, p50, p75, p95 = np.quantile(profiles, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    ax.fill_between(x, p5, p95, color=BLUE_LIGHT, alpha=0.45, linewidth=0, label="5-95%")
    ax.fill_between(x, p25, p75, color=BLUE_LIGHT, alpha=0.9, linewidth=0, label="25-75%")
    ax.plot(x, p50, color=BLUE, linewidth=2, label="Median")
    ax.plot(x, baseline["steady_profile_K"], color=INK, linewidth=1, linestyle=":", label="Deterministic run")
    for limit in spec.material_limits_K:
        ax.axhline(limit, color=INK2, linewidth=0.6, linestyle="--")
        ax.text(x[0], limit + 8, f"{limit:g} K limit", fontsize=7, color=INK2, va="bottom")
    ax.set(xlabel="Axial position [mm]", ylabel="Steady inner-wall temperature [K]",
           title="Steady inner-wall temperature: Monte Carlo band")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.savefig(folder / "uq_axial_band.png", dpi=180)
    plt.close(fig)

    # 4. Surrogate validation
    if gp_test is not None:
        y_true, mean, sd, metrics = gp_test
        fig, ax = plt.subplots(figsize=(4.2, 4), layout="constrained")
        ax.errorbar(y_true, mean, yerr=1.96 * sd, fmt="o", markersize=3.5, color=ORANGE,
                    ecolor=INK2, elinewidth=0.6, capsize=0)
        span = [min(y_true.min(), mean.min()) - 20, max(y_true.max(), mean.max()) + 20]
        ax.plot(span, span, color=INK2, linewidth=0.8)
        ax.text(0.04, 0.96, f"held-out n = {len(y_true)}\nRMSE {metrics['rmse_K']:.2f} K\n"
                f"max |err| {metrics['max_abs_error_K']:.2f} K\nR² {metrics['r2']:.4f}\n"
                f"95% interval coverage {metrics['coverage_95']:.0%}", transform=ax.transAxes,
                va="top", fontsize=8, color=INK)
        ax.set(xlabel="Solver burn-end peak [K]", ylabel="GP prediction [K]", title="Transient surrogate check",
               xlim=span, ylim=span)
        fig.savefig(folder / "uq_gp_validation.png", dpi=180)
        plt.close(fig)



# Driver
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_samples(path, spec, X, Y, output_names):
    header = ",".join(spec.names + output_names)
    np.savetxt(path, np.column_stack([X, Y]), delimiter=",", header=header, comments="", fmt="%.10g")


def run(args):
    t_start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    geometry = load_geometry(args.geometry)
    base = load_config(args.config)
    base.check_geometry(geometry.x, geometry.ri)
    spec = load_spec(args.spec)
    problem = NozzleUQ(geometry, base, spec)
    limits = spec.material_limits_K
    log = (lambda *a: print(*a, flush=True)) if not args.quiet else (lambda *a: None)

    # Deterministic reference (inputs exactly as in the config file)
    model = NozzleModel(geometry, base.to_solver_dict())
    s = model.steady()
    t = base.transient
    det_transient = RadialWall(model, t.radial_cells).transient(t.burn_duration_s, outputs=2, rtol=t.rtol, atol=t.atol_K)
    baseline = {"steady_peak_inner_K": float(s["T_inner_K"].max()), "steady_profile_K": s["T_inner_K"],
                "transient_peak_inner_K": float(det_transient["T_inner_K"][-1].max())}

    summary = {"case_label": base.case_label,
               "deterministic": {k: v for k, v in baseline.items() if k != "steady_profile_K"},
               "inputs": [p.model_dump() for p in spec.parameters]}

    # 1. Steady Monte Carlo
    log(f"[steady MC] {args.n_mc} Latin-hypercube samples")
    U = lhs(spec, args.n_mc, args.seed)
    X = spec.to_physical(U)
    Y, profiles = evaluate(problem, X, "steady", args.workers)
    write_samples(out / "steady_mc_samples.csv", spec, X, Y, STEADY_OUTPUTS)
    np.savez_compressed(out / "steady_mc_profiles.npz", x_m=geometry.x, T_inner_K=profiles)
    summary["steady_monte_carlo"] = {
        "peak_inner_wall": distribution_summary(Y[:, 0], limits, args.seed),
        "peak_outer_wall": distribution_summary(Y[:, 2], [], args.seed),
        "peak_location_x_m": {"p05": float(np.quantile(Y[:, 1], 0.05)), "median": float(np.median(Y[:, 1])),
                              "p95": float(np.quantile(Y[:, 1], 0.95))}}

    # 2. Steady Sobol
    log(f"[steady Sobol] N = {args.n_sobol} -> {args.n_sobol * (spec.dim + 2)} solves")
    sobol_steady_all, _, _ = sobol_indices(
        spec, lambda Us: evaluate(problem, spec.to_physical(Us), "steady", args.workers)[0][:, [0, 2, 3]],
        args.n_sobol, args.seed)
    summary["steady_sobol"] = {"peak_inner_K": sobol_steady_all[0], "peak_outer_K": sobol_steady_all[1],
                               "peak_heat_flux": sobol_steady_all[2],
                               "ranking_peak_inner_by_ST": ranking(sobol_steady_all[0])}

    # 3. Transient surrogate
    transient_peak = sobol_transient = gp_test = None
    if not args.skip_transient:
        log(f"[transient] {args.n_train} training + {args.n_test} held-out solver runs")
        U_train, U_test = lhs(spec, args.n_train, args.seed + 1), lhs(spec, args.n_test, args.seed + 2)
        Y_train = evaluate(problem, spec.to_physical(U_train), "transient", args.workers)
        Y_test = evaluate(problem, spec.to_physical(U_test), "transient", args.workers)
        write_samples(out / "transient_training_runs.csv", spec, spec.to_physical(U_train), Y_train, TRANSIENT_OUTPUTS)
        write_samples(out / "transient_heldout_runs.csv", spec, spec.to_physical(U_test), Y_test, TRANSIENT_OUTPUTS)
        log("[transient] fitting Gaussian process")
        gp = TransientSurrogate(seed=args.seed, features=spec.to_features).fit(U_train, Y_train[:, :2])
        metrics, mean, sd = gp.validate(U_test, Y_test[:, :2])
        gp_test = (Y_test[:, 0], mean[:, 0], sd[:, 0], metrics[0])

        U_big = lhs(spec, args.n_surrogate, args.seed + 3)
        mu, sigma = gp.predict(U_big, return_std=True)
        rng = np.random.default_rng(args.seed)
        draws = mu + sigma * rng.standard_normal(mu.shape)  # propagate surrogate error too
        transient_peak = draws[:, 0]
        sobol_tr, _, _ = sobol_indices(spec, gp.predict, args.n_sobol_surrogate, args.seed)
        sobol_transient = sobol_tr[0]
        summary["transient_surrogate"] = {
            "training_runs": args.n_train, "heldout_runs": args.n_test,
            "validation": {"burn_end_peak_inner_K": metrics[0], "burn_end_peak_outer_K": metrics[1]},
            "surrogate_samples": args.n_surrogate,
            "burn_end_peak_inner_wall": distribution_summary(transient_peak, limits, args.seed),
            "burn_end_peak_outer_wall": distribution_summary(draws[:, 1], [], args.seed),
            "sobol_peak_inner_K": sobol_tr[0], "sobol_peak_outer_K": sobol_tr[1],
            "ranking_peak_inner_by_ST": ranking(sobol_tr[0]),
            "note": "Distribution includes GP predictive uncertainty; Sobol indices use the GP mean."}

    import SALib
    import sklearn
    import scipy
    from importlib.metadata import version
    summary["provenance"] = {
        "config_sha256": base.config_hash(), "spec_sha256": _sha(args.spec), "geometry_sha256": _sha(args.geometry),
        "solver_sha256": _sha(Path(__file__).resolve().parent / "bartz_nozzle.py"), "uq_sha256": _sha(__file__),
        "seed": args.seed, "settings": {k: v for k, v in vars(args).items() if k.startswith("n_")},
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "SALib": version("SALib"), "runtime_s": round(time.time() - t_start, 1),
        "statistics_note": ("Exceedance CIs are exact binomial (Clopper-Pearson) treating samples as independent; "
                            "LHS is at least as accurate, so they are conservative. Results are only as good as "
                            "the input distributions in the spec file.")}
    (out / "uq_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_plots(out, spec, geometry.x, profiles, Y[:, 0], transient_peak, baseline, sobol_steady_all[0],
               sobol_transient, gp_test, f"{base.case_label}: input-uncertainty propagation")
    log(f"done in {time.time() - t_start:.0f} s -> {out}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--output", default=Path("uq_results"), type=Path)
    p.add_argument("--n-mc", type=int, default=4000, help="steady Latin-hypercube samples")
    p.add_argument("--n-sobol", type=int, default=1024, help="Sobol base sample size (power of 2); solves = N(d+2)")
    p.add_argument("--n-train", type=int, default=200, help="transient solver runs for GP training")
    p.add_argument("--n-test", type=int, default=50, help="held-out transient solver runs for GP validation")
    p.add_argument("--n-surrogate", type=int, default=20000, help="surrogate Monte Carlo samples")
    p.add_argument("--n-sobol-surrogate", type=int, default=4096, help="Sobol base size on the surrogate")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-transient", action="store_true")
    p.add_argument("--quiet", action="store_true")
    summary = run(p.parse_args(argv))
    st = summary["steady_monte_carlo"]["peak_inner_wall"]["quantiles"]
    print(f"Steady peak inner wall: median {st['p50_K']['value']:.0f} K, "
          f"90% interval {st['p05_K']['value']:.0f}-{st['p95_K']['value']:.0f} K; "
          f"most influential: {summary['steady_sobol']['ranking_peak_inner_by_ST'][:2]}")


if __name__ == "__main__":
    main()