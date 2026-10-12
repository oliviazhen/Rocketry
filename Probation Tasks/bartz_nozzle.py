"""Bartz nozzle heat transfer and wall temperature, in SI units.

CLI (from the supporting-files folder):
     python ../bartz_nozzle.py --geometry nozzle_geometry.csv \
         --config operating_conditions.json --output . --mode both

Model: choked quasi-1D isentropic gas flow; independent cylindrical wall rings;
constant material and chamber gas properties; prescribed external convection
and radiation. Transient rings use conservative radial finite volumes.
References, input definitions and model limitations are in the accompanying report.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import block_diag, lil_matrix

STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4


@dataclass(frozen=True)
class Geometry:
    x: np.ndarray
    ri: np.ndarray
    thickness: np.ndarray

    @property
    def ro(self):
        return self.ri + self.thickness

    @property
    def throat_index(self):
        return int(np.argmin(self.ri))

    def validate(self):
        if len(self.x) < 3 or self.x.shape != self.ri.shape or self.x.shape != self.thickness.shape:
            raise ValueError("Geometry must contain at least three matching 1D columns")
        if not all(np.all(np.isfinite(a)) for a in (self.x, self.ri, self.thickness)):
            raise ValueError("Geometry values must be finite")
        if np.any(np.diff(self.x) <= 0) or np.any(self.ri <= 0) or np.any(self.thickness <= 0):
            raise ValueError("x must increase strictly; radius and radial wall thickness must be positive")
        j = self.throat_index
        if j == 0 or j == len(self.x) - 1:
            raise ValueError("Include converging and diverging sections around an interior throat")
        if np.count_nonzero(np.isclose(self.ri, self.ri[j], rtol=1e-10, atol=1e-14)) != 1:
            raise ValueError("The contour must have one sampled minimum-radius throat station")
        if np.any(np.diff(self.ri[:j + 1]) > 1e-12) or np.any(np.diff(self.ri[j:]) < -1e-12):
            raise ValueError("Radius must be nonincreasing before, and nondecreasing after, the throat")


def load_geometry(path):
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    required = {"x_m", "r_inner_m", "wall_thickness_m"}
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError("CSV requires x_m,r_inner_m,wall_thickness_m (all metres)")
    g = Geometry(np.atleast_1d(data["x_m"]), np.atleast_1d(data["r_inner_m"]),
                 np.atleast_1d(data["wall_thickness_m"]))
    g.validate()
    return g


def area_mach(M, gamma):
    """Isentropic A/A* on either Mach branch."""
    M = np.asarray(M, dtype=float)
    if gamma <= 1 or np.any(M <= 0):
        raise ValueError("gamma > 1 and M > 0 are required")
    return (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M ** 2)) ** (
        (gamma + 1) / (2 * (gamma - 1))) / M


def mach_from_area(ratio, gamma, supersonic):
    ratio = float(ratio)
    if not np.isfinite(ratio) or ratio < 1 - 1e-12:
        raise ValueError("A/A* must be at least one")
    if abs(ratio - 1) <= 1e-12:
        return 1.0
    if not supersonic:
        return brentq(lambda M: area_mach(M, gamma) - ratio, 1e-12, 1.0, xtol=1e-13)
    upper = 2.0
    while area_mach(upper, gamma) < ratio:
        upper *= 2
        if upper > 256:
            raise ValueError("Area ratio exceeds the supported Mach bracket")
    return brentq(lambda M: area_mach(M, gamma) - ratio, 1.0, upper, xtol=1e-13)


def _positive(section, key):
    value = float(section[key])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be finite and positive")
    return value


class NozzleModel:
    def __init__(self, geometry, config):
        geometry.validate()
        self.geometry, self.config = geometry, config
        gas, wall, external = config["gas"], config["wall"], config["external"]
        self.pc = _positive(gas, "chamber_pressure_Pa")  # ABSOLUTE stagnation pressure
        self.T0 = _positive(gas, "stagnation_temperature_K")
        self.gamma = _positive(gas, "gamma")
        if not 1 < self.gamma < 2:
            raise ValueError("This perfect-gas model requires 1 < gamma < 2")
        self.cp_gas = _positive(gas, "cp_J_kgK")
        self.mu = _positive(gas, "viscosity_Pa_s")
        self.Pr = _positive(gas, "Prandtl")
        self.cstar = _positive(gas, "c_star_m_s")
        self.rc = _positive(gas, "throat_curvature_radius_m")
        self.factor = float(gas.get("bartz_multiplier", 1.0))
        if not np.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("bartz_multiplier must be positive")
        self.recovery = float(gas.get("recovery_factor", self.Pr ** (1 / 3)))
        if not np.isfinite(self.recovery) or self.recovery <= 0:
            raise ValueError("Recovery factor must be positive")
        self.k = _positive(wall, "conductivity_W_mK")
        self.rho = _positive(wall, "density_kg_m3")
        self.cp_wall = _positive(wall, "cp_J_kgK")
        self.initial = _positive(wall, "initial_temperature_K")
        n = len(geometry.x)
        def profile(name, positive=False):
            values = np.asarray(external[name], dtype=float)
            if values.ndim == 0:
                values = np.full(n, float(values))
            if values.shape != (n,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be a finite scalar or {n}-entry list")
            if np.any(values <= 0 if positive else values < 0):
                raise ValueError(f"Invalid {name}")
            return values
        self.hout = profile("h_W_m2K")
        self.Tinf = profile("bulk_temperature_K", positive=True)
        self.Tsur = profile("surroundings_temperature_K", positive=True)
        self.emissivity = profile("emissivity")
        if np.any(self.emissivity > 1):
            raise ValueError("Emissivity must be in [0,1]")
        self.throat = geometry.throat_index
        self.Dt = 2 * geometry.ri[self.throat]
        self.At = np.pi * geometry.ri[self.throat] ** 2
        self.area = np.pi * geometry.ri ** 2
        self.ratio = self.area / self.At
        self.M = np.array([mach_from_area(a, self.gamma, i > self.throat)
                           for i, a in enumerate(self.ratio)])
        self.chi = 1 + (self.gamma - 1) / 2 * self.M ** 2
        self.Tstatic = self.T0 / self.chi
        self.pstatic = self.pc / self.chi ** (self.gamma / (self.gamma - 1))
        self.Taw = self.Tstatic * (1 + self.recovery * (self.gamma - 1) / 2 * self.M ** 2)
        if np.any(np.maximum(self.Tinf, self.Tsur) > self.Taw):
            raise ValueError("This heating-case solver requires external temperatures <= local Taw")
        if self.initial > np.min(self.Taw):
            raise ValueError("Initial wall temperature must not exceed the smallest local Taw")
        self.mdot = self.pc * self.At / self.cstar
        self.Re = self.mdot / self.area * (2 * geometry.ri) / self.mu
        self.h_base = (self.factor * 0.026 / self.Dt ** 0.2
                       * self.mu ** 0.2 * self.cp_gas / self.Pr ** 0.6
                       * (self.pc / self.cstar) ** 0.8
                       * (self.Dt / self.rc) ** 0.1 * self.ratio ** -0.9)
        self.inner_area_per_length = 2 * np.pi * geometry.ri
        self.outer_area_per_length = 2 * np.pi * geometry.ro
        self.Rwall = np.log(geometry.ro / geometry.ri) / (2 * np.pi * self.k)
        self.notes = self._assumption_checks()

    def _assumption_checks(self):
        notes = []
        slope = np.max(np.abs(np.gradient(self.geometry.ri, self.geometry.x)))
        if slope > 0.35:
            notes.append("Steep contour: local cylindrical rings neglect slope-dependent wall geometry")
        if np.min(self.Re) < 1e4:
            notes.append("Low local Reynolds number: turbulent Bartz validity needs review")
        Rgas = self.cp_gas * (self.gamma - 1) / self.gamma
        cstar_ideal = np.sqrt(Rgas * self.T0 / self.gamma) * (
            (self.gamma + 1) / 2) ** ((self.gamma + 1) / (2 * (self.gamma - 1)))
        if abs(self.cstar / cstar_ideal - 1) > 0.05:
            notes.append("Input c* differs >5% from the constant-cp perfect-gas c*; check gas consistency")
        back = self.config["gas"].get("back_pressure_Pa")
        if back is not None:
            back = float(back)
            if not np.isfinite(back) or back < 0:
                raise ValueError("back_pressure_Pa must be finite and nonnegative")
            exit_subsonic_M = mach_from_area(self.ratio[-1], self.gamma, False)
            choking_back_limit = self.pc / (1 + (self.gamma - 1) / 2 * exit_subsonic_M ** 2) ** (
                self.gamma / (self.gamma - 1))
            if back >= choking_back_limit:
                raise ValueError("Back pressure is too high for the imposed choked-flow assumption")
            # Pressure immediately behind a normal shock located at the exit.
            postshock_exit_p = self.pstatic[-1] * (1 + 2 * self.gamma / (self.gamma + 1)
                                                   * (self.M[-1] ** 2 - 1))
            if back > postshock_exit_p:
                notes.append("Back pressure permits an internal normal shock; isentropic supersonic branch may fail")
            elif back > self.pstatic[-1]:
                notes.append("Overexpanded exit: external shocks/separation require independent assessment")
        return notes

    def bartz_h(self, Tw):
        Tw = np.asarray(Tw, dtype=float)
        if np.any(Tw <= 0) or not np.all(np.isfinite(Tw)):
            raise ValueError("Wall temperatures must be finite Kelvin values > 0")
        sigma = (0.5 * Tw / self.T0 * self.chi + 0.5) ** -0.68 * self.chi ** -0.12
        return self.h_base * sigma

    def hot_heat(self, Tw):
        """Hot gas heat input in W per metre of axial nozzle length."""
        return self.inner_area_per_length * self.bartz_h(Tw) * (self.Taw - Tw)

    def hot_derivative(self, Tw):
        h = self.bartz_h(Tw)
        dh = -0.68 * h * (0.5 * self.chi / self.T0) / (0.5 * Tw / self.T0 * self.chi + 0.5)
        return self.inner_area_per_length * (dh * (self.Taw - Tw) - h)

    def outer_heat(self, Tw):
        return self.outer_area_per_length * (self.hout * (Tw - self.Tinf)
                + self.emissivity * STEFAN_BOLTZMANN * (Tw ** 4 - self.Tsur ** 4))

    def outer_derivative(self, Tw):
        return self.outer_area_per_length * (self.hout + 4 * self.emissivity * STEFAN_BOLTZMANN * Tw ** 3)

    def steady(self):
        """Coupled gas film, cylindrical wall conduction, and outer heat loss."""
        n = len(self.M)
        Twg, Twc, heat = np.empty(n), np.empty(n), np.empty(n)
        for i in range(n):
            def qout(T):
                return self.outer_area_per_length[i] * (self.hout[i] * (T - self.Tinf[i])
                        + self.emissivity[i] * STEFAN_BOLTZMANN * (T ** 4 - self.Tsur[i] ** 4))
            if self.hout[i] == 0 and self.emissivity[i] == 0:
                Twg[i] = Twc[i] = self.Taw[i]
                heat[i] = 0
                continue
            low, high = min(self.Tinf[i], self.Tsur[i]), max(self.Tinf[i], self.Tsur[i])
            equilibrium = low if low == high else brentq(qout, low, high, xtol=1e-10)
            def residual(Tcold):
                Thot = Tcold + self.Rwall[i] * qout(Tcold)
                sigma = (0.5 * Thot / self.T0 * self.chi[i] + 0.5) ** -0.68 * self.chi[i] ** -0.12
                qgas = self.inner_area_per_length[i] * self.h_base[i] * sigma * (self.Taw[i] - Thot)
                return qgas - qout(Tcold)
            Twc[i] = brentq(residual, equilibrium, self.Taw[i], xtol=1e-9)
            heat[i] = qout(Twc[i])
            Twg[i] = Twc[i] + self.Rwall[i] * heat[i]
        return dict(T_inner_K=Twg, T_outer_K=Twc, h_gas_W_m2K=self.bartz_h(Twg),
                    heat_per_length_W_m=heat, heat_flux_inner_W_m2=heat / self.inner_area_per_length)


def _bounded_newton(function, derivative, lower, upper):
    """Vectorized safeguarded Newton for independent monotone scalar equations."""
    lower, upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
    x = (lower + upper) / 2
    for _ in range(70):
        residual = function(x)
        lower = np.where(residual < 0, x, lower)
        upper = np.where(residual >= 0, x, upper)
        if np.max(upper - lower) < 1e-8:
            return (lower + upper) / 2
        trial = x - residual / derivative(x)
        midpoint = (lower + upper) / 2
        x = np.where((trial > lower) & (trial < upper) & np.isfinite(trial), trial, midpoint)
    raise RuntimeError("Boundary surface-temperature solve did not converge")


class RadialWall:
    """Conservative cylindrical finite-volume wall at every axial station."""
    def __init__(self, model, cells):
        if cells < 3:
            raise ValueError("At least three radial cells are required")
        self.model, self.cells = model, int(cells)
        g = model.geometry
        self.faces = g.ri[:, None] + g.thickness[:, None] * np.linspace(0, 1, cells + 1)
        self.centers = (self.faces[:, 1:] + self.faces[:, :-1]) / 2
        self.capacity = model.rho * model.cp_wall * np.pi * (
            self.faces[:, 1:] ** 2 - self.faces[:, :-1] ** 2)
        self.G = 2 * np.pi * model.k / np.log(self.centers[:, 1:] / self.centers[:, :-1])
        self.Gin = 2 * np.pi * model.k / np.log(self.centers[:, 0] / g.ri)
        self.Gout = 2 * np.pi * model.k / np.log(g.ro / self.centers[:, -1])

    def surface_temperatures(self, T):
        m = self.model
        hot_node, cold_node = T[:, 0], T[:, -1]
        Thot = _bounded_newton(lambda Tw: self.Gin * (Tw - hot_node) - m.hot_heat(Tw),
                              lambda Tw: self.Gin - m.hot_derivative(Tw),
                              np.minimum(hot_node, m.Taw), np.maximum(hot_node, m.Taw))
        lower = np.minimum(cold_node, np.minimum(m.Tinf, m.Tsur))
        upper = np.maximum(cold_node, np.maximum(m.Tinf, m.Tsur))
        Tcold = _bounded_newton(lambda Tw: self.Gout * (Tw - cold_node) + m.outer_heat(Tw),
                               lambda Tw: self.Gout + m.outer_derivative(Tw), lower, upper)
        return Thot, Tcold

    def fluxes(self, T):
        Thot, Tcold = self.surface_temperatures(T)
        return self.model.hot_heat(Thot), self.model.outer_heat(Tcold), Thot, Tcold

    def rhs(self, time, flattened):
        # Last variable at each station accumulates net boundary heat in J/m.
        state = flattened.reshape(len(self.model.M), self.cells + 1)
        T = state[:, :self.cells]
        incoming, outgoing, _, _ = self.fluxes(T)
        interfaces = self.G * (T[:, :-1] - T[:, 1:])
        net = np.zeros_like(T)
        net[:, 0] = incoming
        net[:, -1] = -outgoing
        net[:, :-1] -= interfaces
        net[:, 1:] += interfaces
        result = np.zeros_like(state)
        result[:, :self.cells] = net / self.capacity
        result[:, -1] = incoming - outgoing
        return result.ravel()

    def transient(self, duration, outputs=101, rtol=1e-6, atol=1e-5):
        if not np.isfinite(duration) or duration <= 0 or outputs < 2 or rtol <= 0 or atol <= 0:
            raise ValueError("Invalid duration, output count or integration tolerances")
        n, m = len(self.model.M), self.cells
        state0 = np.zeros((n, m + 1))
        state0[:, :m] = self.model.initial
        pattern = lil_matrix((m + 1, m + 1), dtype=int)
        for i in range(m):
            pattern[i, max(0, i - 1):min(m, i + 2)] = 1
        pattern[m, 0] = pattern[m, m - 1] = 1
        sparsity = block_diag([pattern.tocsr()] * n, format="csr")
        atol_vector = np.full((n, m + 1), atol)
        atol_vector[:, -1] = max(atol, 1e-3)
        solution = solve_ivp(self.rhs, (0.0, duration), state0.ravel(), method="BDF",
                             t_eval=np.linspace(0, duration, outputs), rtol=rtol,
                             atol=atol_vector.ravel(), jac_sparsity=sparsity)
        if not solution.success:
            raise RuntimeError(solution.message)
        state = solution.y.T.reshape(outputs, n, m + 1)
        T = state[:, :, :m]
        Thot, Tcold = [], []
        for field in T:
            h, c = self.surface_temperatures(field)
            Thot.append(h)
            Tcold.append(c)
        Thot, Tcold = np.array(Thot), np.array(Tcold)
        stored = np.sum(self.capacity[None, :, :] * (T - self.model.initial), axis=2)
        integrated = state[:, :, -1]
        energy_error = float(np.max(np.abs(stored - integrated)))
        energy_scale = max(1.0, float(np.max(np.abs(integrated))))
        return dict(time_s=solution.t, cell_temperature_K=T,
                    radial_cell_centers_m=self.centers,
                    T_inner_K=Thot, T_outer_K=Tcold,
                    h_gas_W_m2K=self.model.bartz_h(Thot),
                    heat_flux_inner_W_m2=self.model.hot_heat(Thot) / self.model.inner_area_per_length,
                    stored_energy_J_m=stored, integrated_net_heat_J_m=integrated,
                    energy_audit_relative_error=energy_error / energy_scale,
                    energy_audit_absolute_error_J_m=energy_error, rhs_evaluations=solution.nfev)


def write_csv(path, columns):
    keys = list(columns)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(keys)
        writer.writerows(zip(*(np.asarray(columns[k]).ravel() for k in keys)))


def export_results(model, steady, transient, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    g = model.geometry
    (folder / "input_conditions.json").write_text(json.dumps(model.config, indent=2) + "\n", encoding="utf-8")
    write_csv(folder / "input_geometry.csv", dict(x_m=g.x, r_inner_m=g.ri,
                                                wall_thickness_m=g.thickness))
    columns = dict(x_m=g.x, r_inner_m=g.ri, wall_thickness_m=g.thickness,
                   area_ratio=model.ratio, Mach=model.M, pressure_Pa=model.pstatic,
                   T_gas_static_K=model.Tstatic, T_adiabatic_wall_K=model.Taw,
                   Reynolds=model.Re)
    for suffix, values in [("steady", steady), ("burn_end", transient)]:
        if values is None:
            continue
        for key in ["T_inner_K", "T_outer_K", "h_gas_W_m2K", "heat_flux_inner_W_m2"]:
            value = values[key] if suffix == "steady" else values[key][-1]
            columns[key + "_" + suffix] = value
    write_csv(folder / "wall_temperature_profile.csv", columns)
    summary = {"case_label": model.config.get("case_label", "User input"),
               "throat_x_m": float(g.x[model.throat]), "throat_diameter_m": model.Dt,
               "throat_curvature_radius_m": model.rc, "mass_flow_kg_s": model.mdot,
               "exit_Mach": float(model.M[-1]), "exit_static_pressure_Pa": float(model.pstatic[-1]),
               "assumption_notes": model.notes,
               "model": "Choked isentropic gas; independent cylindrical radial wall rings",
               "experimental_validation": False}
    if steady is not None:
        i = int(np.argmax(steady["T_inner_K"]))
        summary["steady"] = {"peak_inner_wall_K": float(steady["T_inner_K"][i]),
                              "peak_inner_wall_x_m": float(g.x[i]),
                              "peak_outer_wall_K": float(np.max(steady["T_outer_K"])),
                              "peak_h_gas_W_m2K": float(np.max(steady["h_gas_W_m2K"])),
                              "peak_inner_heat_flux_W_m2": float(np.max(steady["heat_flux_inner_W_m2"])),
                              "gas_heat_input_W_local_ring_integral": float(np.trapezoid(
                                  steady["heat_per_length_W_m"], g.x))}
    if transient is not None:
        time = transient["time_s"]
        ntime, nx = transient["T_inner_K"].shape
        i, j = np.unravel_index(np.argmax(transient["T_inner_K"]), (ntime, nx))
        summary["transient"] = {"burn_duration_s": float(time[-1]),
                                "peak_inner_wall_K": float(transient["T_inner_K"][i, j]),
                                "peak_inner_wall_time_s": float(time[i]),
                                "peak_inner_wall_x_m": float(g.x[j]),
                                "peak_outer_wall_K": float(np.max(transient["T_outer_K"])),
                                "energy_audit_relative_error": transient["energy_audit_relative_error"],
                                "energy_audit_absolute_error_J_m": transient["energy_audit_absolute_error_J_m"],
                                "rhs_evaluations": transient["rhs_evaluations"]}
        write_csv(folder / "transient_surface_temperatures.csv", {
            "time_s": np.repeat(time, nx), "x_m": np.tile(g.x, ntime),
            "T_inner_K": transient["T_inner_K"], "T_outer_K": transient["T_outer_K"],
            "h_gas_W_m2K": transient["h_gas_W_m2K"],
            "heat_flux_inner_W_m2": transient["heat_flux_inner_W_m2"]})
        np.savez_compressed(folder / "radial_temperature_field.npz", time_s=time, x_m=g.x,
                            radial_cell_centers_m=transient["radial_cell_centers_m"],
                            cell_temperature_K=transient["cell_temperature_K"],
                            stored_energy_J_m=transient["stored_energy_J_m"],
                            integrated_net_heat_J_m=transient["integrated_net_heat_J_m"])
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_results(model, steady, transient, folder)
    return summary


def plot_results(model, steady, transient, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    x = model.geometry.x * 1000
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), layout="constrained")
    axes[0, 0].plot(x, model.geometry.ri * 1000, "k-", label="Inner radius")
    axes[0, 0].plot(x, model.geometry.ro * 1000, "k--", label="Outer radius")
    axes[0, 0].set(ylabel="Radius [mm]")
    axes[0, 1].plot(x, model.Tstatic, "k:", label="Static gas T")
    axes[0, 1].plot(x, model.Taw, "k-.", label="Recovery T")
    if steady is not None:
        axes[0, 1].plot(x, steady["T_inner_K"], "k-", label="Inner wall, steady")
        axes[0, 1].plot(x, steady["T_outer_K"], "k--", label="Outer wall, steady")
        axes[1, 0].plot(x, steady["h_gas_W_m2K"] / 1000, "k-", label="Steady")
        axes[1, 1].plot(x, steady["heat_flux_inner_W_m2"] / 1e6, "k-", label="Steady")
    if transient is not None:
        axes[0, 1].plot(x, transient["T_inner_K"][-1], color="0.5", label="Inner wall, burn end")
        axes[1, 0].plot(x, transient["h_gas_W_m2K"][-1] / 1000, "k--", label="Burn end")
        axes[1, 1].plot(x, transient["heat_flux_inner_W_m2"][-1] / 1e6, "k--", label="Burn end")
    axes[0, 1].set(ylabel="Temperature [K]")
    axes[1, 0].set(ylabel="Bartz h [kW m$^{-2}$ K$^{-1}$]")
    axes[1, 1].set(ylabel="Inner heat flux [MW m$^{-2}$]")
    for ax in axes.flat:
        ax.axvline(x[model.throat], color="0.6", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Axial position [mm]")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(model.config.get("case_label", "Nozzle thermal analysis"), fontsize=10)
    fig.savefig(folder / "axial_profiles.pdf")
    fig.savefig(folder / "axial_profiles.png", dpi=180)
    plt.close(fig)
    if transient is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
        j = model.throat
        axes[0].plot(transient["time_s"], transient["T_inner_K"][:, j], "k-", label="Throat inner wall")
        axes[0].plot(transient["time_s"], transient["T_outer_K"][:, j], "k--", label="Throat outer wall")
        if steady is not None:
            axes[0].axhline(steady["T_inner_K"][j], color="0.5", linestyle=":", label="Steady inner wall")
        axes[0].set(xlabel="Time [s]", ylabel="Temperature [K]")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)
        image = axes[1].pcolormesh(x, transient["time_s"], transient["T_inner_K"],
                                    shading="auto", cmap="Greys")
        axes[1].set(xlabel="Axial position [mm]", ylabel="Time [s]")
        fig.colorbar(image, ax=axes[1], label="Inner wall T [K]")
        fig.savefig(folder / "transient_wall.pdf")
        fig.savefig(folder / "transient_wall.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", default=Path("results"), type=Path)
    parser.add_argument("--mode", choices=["steady", "transient", "both"], default="both")
    args = parser.parse_args()
    geometry = load_geometry(args.geometry)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = NozzleModel(geometry, config)
    for note in model.notes:
        warnings.warn(note, stacklevel=1)
    steady = model.steady() if args.mode in ("steady", "both") else None
    transient = None
    if args.mode in ("transient", "both"):
        settings = config["transient"]
        transient = RadialWall(model, int(settings.get("radial_cells", 24))).transient(
            _positive(settings, "burn_duration_s"), int(settings.get("output_times", 101)),
            float(settings.get("rtol", 1e-6)), float(settings.get("atol_K", 1e-5)))
    summary = export_results(model, steady, transient, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
