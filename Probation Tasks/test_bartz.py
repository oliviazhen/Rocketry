"""Physical and numerical verification using standard-library unittest."""
import json
from pathlib import Path
import unittest
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bartz_nozzle import Geometry, NozzleModel, RadialWall, area_mach, mach_from_area
def example_geometry(points_before=41, points_after=81):
    x = np.r_[np.linspace(0, 0.080, points_before),
              np.linspace(0.080, 0.200, points_after)[1:]]
    rt, rc, xt = 0.012, 0.025, 0.080
    radius = np.empty_like(x)
    for i, coordinate in enumerate(x):
        distance = abs(coordinate - xt)
        slope = 0.25 if coordinate < xt else 0.20
        join = rc * slope / np.sqrt(1 + slope ** 2)
        at_join = rt + rc * (1 - 1 / np.sqrt(1 + slope ** 2))
        radius[i] = (rt + rc - np.sqrt(rc ** 2 - distance ** 2)
                     if distance <= join else at_join + slope * (distance - join))
    return np.column_stack([x, radius, np.full_like(x, 0.003)])


def example_config():
    gamma, R, T0 = 1.2, 400.0, 2800.0
    cstar = np.sqrt(R * T0 / gamma) * ((gamma + 1) / 2) ** (
        (gamma + 1) / (2 * (gamma - 1)))
    return {
        "case_label": "Synthetic example - NOT USYD nozzle data",
        "gas": {
            "chamber_pressure_Pa": 2000000.0,
            "stagnation_temperature_K": T0,
            "gamma": gamma,
            "cp_J_kgK": gamma * R / (gamma - 1),
            "viscosity_Pa_s": 8e-5,
            "Prandtl": 0.7,
            "c_star_m_s": float(cstar),
            "throat_curvature_radius_m": 0.025,
            "back_pressure_Pa": 0.0,
            "bartz_multiplier": 1.0
        },
        "wall": {
            "description": "Illustrative homogeneous wall; not a specified material grade",
            "conductivity_W_mK": 30.0,
            "density_kg_m3": 7800.0,
            "cp_J_kgK": 600.0,
            "initial_temperature_K": 293.15
        },
        "external": {
            "description": "Prescribed effective cooling film; no coolant circuit is modelled",
            "h_W_m2K": 8000.0,
            "bulk_temperature_K": 293.15,
            "surroundings_temperature_K": 293.15,
            "emissivity": 0.7
        },
        "transient": {
            "burn_duration_s": 1.0,
            "radial_cells": 24,
            "output_times": 101,
            "rtol": 1e-6,
            "atol_K": 1e-5
        }
    }



def make_model(points_before=7, points_after=9, config=None):
    data = example_geometry(points_before, points_after)
    g = Geometry(data[:, 0], data[:, 1], data[:, 2])
    return NozzleModel(g, example_config() if config is None else config)


class PhysicalTests(unittest.TestCase):
    def test_known_area_mach_and_two_branches(self):
        self.assertAlmostEqual(float(area_mach(2, 1.4)), 1.6875, places=12)
        self.assertAlmostEqual(mach_from_area(1.6875, 1.4, True), 2.0, places=11)
        self.assertLess(mach_from_area(1.6875, 1.4, False), 1)
        self.assertEqual(mach_from_area(1, 1.4, False), 1)

    def test_throat_and_isentropic_stagnation_relation(self):
        m = make_model()
        self.assertEqual(m.M[m.throat], 1)
        self.assertTrue(np.all(m.M[:m.throat] < 1))
        self.assertTrue(np.all(m.M[m.throat + 1:] > 1))
        np.testing.assert_allclose(area_mach(m.M, m.gamma), m.ratio, rtol=1e-11)
        np.testing.assert_allclose(m.pstatic / m.pc, (m.Tstatic / m.T0) ** (
            m.gamma / (m.gamma - 1)), rtol=1e-12)
        self.assertTrue(np.all(m.Taw >= m.Tstatic))
        self.assertTrue(np.all(m.Taw <= m.T0))

    def test_bartz_pressure_viscosity_and_cstar_scaling(self):
        base = make_model()
        for key, multiplier, exponent in [("chamber_pressure_Pa", 2, 0.8),
                                            ("viscosity_Pa_s", 2, 0.2),
                                            ("c_star_m_s", 2, -0.8),
                                            ("throat_curvature_radius_m", 2, -0.1)]:
            cfg = example_config()
            cfg["gas"][key] *= multiplier
            changed = make_model(config=cfg)
            np.testing.assert_allclose(changed.bartz_h(1000) / base.bartz_h(1000),
                                       multiplier ** exponent, rtol=1e-12)

    def test_wall_temperature_feedback_and_heat_derivative(self):
        m = make_model()
        self.assertTrue(np.all(m.bartz_h(1500) < m.bartz_h(500)))
        T = np.full(len(m.M), 1000.0)
        numeric = (m.hot_heat(T + 0.001) - m.hot_heat(T - 0.001)) / 0.002
        np.testing.assert_allclose(m.hot_derivative(T), numeric, rtol=1e-8)

    def test_steady_three_fluxes_and_temperature_order(self):
        m = make_model()
        s = m.steady()
        gas = m.hot_heat(s["T_inner_K"])
        wall = (s["T_inner_K"] - s["T_outer_K"]) / m.Rwall
        outer = m.outer_heat(s["T_outer_K"])
        np.testing.assert_allclose(gas, wall, rtol=1e-8, atol=1e-4)
        np.testing.assert_allclose(wall, outer, rtol=1e-10, atol=1e-4)
        self.assertTrue(np.all(s["T_inner_K"] >= s["T_outer_K"]))
        self.assertTrue(np.all(s["T_inner_K"] <= m.Taw))
        self.assertTrue(np.all(s["T_outer_K"] >= m.Tinf))

    def test_adiabatic_steady_limit(self):
        cfg = example_config()
        cfg["external"].update(h_W_m2K=0.0, emissivity=0.0)
        m = make_model(config=cfg)
        s = m.steady()
        np.testing.assert_allclose(s["T_inner_K"], m.Taw)
        np.testing.assert_allclose(s["T_outer_K"], m.Taw)
        np.testing.assert_allclose(s["heat_flux_inner_W_m2"], 0)

    def test_radiation_only_steady_balance(self):
        cfg = example_config()
        cfg["external"]["h_W_m2K"] = 0.0
        m = make_model(config=cfg)
        s = m.steady()
        np.testing.assert_allclose(m.hot_heat(s["T_inner_K"]), m.outer_heat(s["T_outer_K"]),
                                   rtol=1e-8, atol=1e-4)

    def test_exact_logarithmic_steady_wall_is_discrete_equilibrium(self):
        m = make_model()
        wall = RadialWall(m, 17)
        s = m.steady()
        T = s["T_inner_K"][:, None] - s["heat_per_length_W_m"][:, None] / (
            2 * np.pi * m.k) * np.log(wall.centers / m.geometry.ri[:, None])
        state = np.column_stack([T, np.zeros(len(m.M))])
        rhs = wall.rhs(0, state.ravel()).reshape(state.shape)
        self.assertLess(np.max(np.abs(rhs[:, :-1])), 2e-5)

    def test_instantaneous_energy_conservation(self):
        m = make_model()
        wall = RadialWall(m, 12)
        T = np.tile(np.linspace(900, 350, 12), (len(m.M), 1))
        state = np.column_stack([T, np.zeros(len(m.M))])
        rhs = wall.rhs(0, state.ravel()).reshape(state.shape)
        incoming, outgoing, _, _ = wall.fluxes(T)
        np.testing.assert_allclose(np.sum(wall.capacity * rhs[:, :-1], axis=1),
                                   incoming - outgoing, rtol=1e-11, atol=1e-5)

    def test_transient_energy_audit_and_temperature_bounds(self):
        m = make_model(3, 3)
        t = RadialWall(m, 16).transient(1.0, outputs=11)
        self.assertLess(t["energy_audit_relative_error"], 1e-7)
        self.assertTrue(np.all(t["cell_temperature_K"] >= m.initial - 1e-3))
        self.assertTrue(np.all(t["T_inner_K"] <= m.Taw[None, :] + 1e-3))
        np.testing.assert_allclose(t["cell_temperature_K"][0], m.initial)

    def test_long_time_convergence_to_independent_steady_solve(self):
        m = make_model(3, 3)
        s = m.steady()
        t = RadialWall(m, 20).transient(20.0, outputs=3)
        np.testing.assert_allclose(t["T_inner_K"][-1], s["T_inner_K"], atol=0.05, rtol=0)
        np.testing.assert_allclose(t["T_outer_K"][-1], s["T_outer_K"], atol=0.05, rtol=0)

    def test_invalid_geometry_and_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            Geometry(np.array([0, 0, 1]), np.array([2, 1, 2]), np.ones(3)).validate()
        for section, key, value in [("gas", "gamma", 1.0),
                                    ("gas", "chamber_pressure_Pa", -1),
                                    ("external", "emissivity", 1.1),
                                    ("wall", "conductivity_W_mK", 0)]:
            cfg = example_config()
            cfg[section][key] = value
            with self.assertRaises(ValueError):
                make_model(config=cfg)

    def test_unchoked_back_pressure_rejected(self):
        cfg = example_config()
        cfg["gas"]["back_pressure_Pa"] = cfg["gas"]["chamber_pressure_Pa"]
        with self.assertRaises(ValueError):
            make_model(config=cfg)

    def test_variable_external_profiles(self):
        cfg = example_config()
        n = len(example_geometry(7, 9))
        cfg["external"]["h_W_m2K"] = np.linspace(5000, 10000, n).tolist()
        m = make_model(config=cfg)
        self.assertEqual(m.hout.shape, (n,))
        self.assertTrue(np.all(np.isfinite(m.steady()["T_inner_K"])))


def convergence_study():
    """Separate grid/time-tolerance studies at the throat and along x."""
    m = make_model(3, 3)
    j = m.throat
    runs, radial = {}, []
    for cells in [12, 24, 48]:
        run = RadialWall(m, cells).transient(1.0, outputs=3, rtol=2e-7, atol=2e-6)
        runs[cells] = run
        radial.append({"radial_cells": cells, "throat_inner_K": float(run["T_inner_K"][-1, j]),
                       "throat_outer_K": float(run["T_outer_K"][-1, j]),
                       "energy_audit_relative_error": run["energy_audit_relative_error"]})
    loose = RadialWall(m, 24).transient(1, outputs=3, rtol=1e-5, atol=1e-4)
    time_difference = float(np.max(np.abs(loose["T_inner_K"][-1] - runs[24]["T_inner_K"][-1])))
    axial = []
    for before, after in [(41, 81), (81, 161), (161, 321)]:
        mx = make_model(before, after)
        sx = mx.steady()
        axial.append({"axial_stations": len(mx.M),
                      "peak_inner_K": float(np.max(sx["T_inner_K"])),
                      "peak_heat_flux_W_m2": float(np.max(sx["heat_flux_inner_W_m2"]))})
    result = {"radial_grid_study": radial,
              "radial_24_vs_48_throat_difference_K": abs(radial[1]["throat_inner_K"] - radial[2]["throat_inner_K"]),
              "temporal_tolerance_difference_K": time_difference,
              "axial_grid_study": axial,
              "note": "Numerical verification only; no experimental validation"}
    assert result["radial_24_vs_48_throat_difference_K"] < 1.0
    assert time_difference < 0.1
    path = Path(__file__).resolve().parent / "verification.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PhysicalTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    target = Path(__file__).resolve().parent
    target.mkdir(exist_ok=True)
    (target / "unit_test_results.json").write_text(json.dumps({
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "successful": result.wasSuccessful(),
        "test_names": unittest.defaultTestLoader.getTestCaseNames(PhysicalTests)
    }, indent=2) + "\n")
    if not result.wasSuccessful():
        raise SystemExit(1)
    convergence_study()
