# Proposed changes to `bartz_nozzle.py`

This README summarises two issues found in the current code and how each is being addressed.

| # | Area | Status |
|---|---|---|
| 1 | Input validation and units | Proposed |
| 2 | Uncertainty in the Bartz correlation | Calibration done; uncertainty propagation proposed |

---

## 1. Input validation and units

### Issue
Input validation is scattered through `NozzleModel.__init__`:

- `_positive()` checks on individual values
- the `profile()` helper for external boundary conditions
- separate range checks on `gamma` and `emissivity`
- defaults buried in the code, e.g. `config["transient"].get("radial_cells", 24)`

Units only exist in key names (`chamber_pressure_Pa`, `cp_J_kgK`). The code also stops at the first bad value it finds, partway through setup, so a file with several mistakes has to be fixed and rerun one error at a time.

### Solution: a Pydantic input model
Define the configuration as a Pydantic model so that:

- **Everything is in one place.** Every input, its type, default, allowed range and unit is declared together, so the config documents itself.
- **Validation happens up front.** The whole file is checked before any calculation starts, and all errors are reported at once.
- **A JSON Schema is generated automatically.** Anyone writing an input file, including whoever produces the input JSON, can see exactly what's expected.
- **Sampling is easier.** Because valid ranges are declared explicitly, it's straightforward to draw valid inputs for sensitivity analysis or Monte Carlo runs.

---

## 2. Uncertainty in the Bartz correlation

### Issue
In `NozzleModel.__init__`, the line

```python
self.h_base = (self.factor * 0.026 / self.Dt ** 0.2 * ...)
```

is Bartz's simple nozzle equation (Eq. 1 in the report). `bartz_h()` adds the wall-temperature correction σ, and `hot_heat()` turns h into heat input. **Every wall temperature the solver reports, steady or transient, comes through that line.**

The only control on it is `bartz_multiplier` (`self.factor`), which defaults to 1.0. That means the code trusts Bartz exactly, with no correction and no uncertainty.

### Solution, part 1: calibrate against experimental data (done)
Fitting to Back, Massier & Gier runs the model backwards. Instead of predicting h from the inputs, we take their measured h and wall temperatures and ask what `bartz_multiplier` would make the code reproduce them.

- **Data:** Back, Massier & Gier (1965), JPL TR 32-415, transcribed from the report's printed tables. That gives 11 tests and 221 stations, with every row consistency-checked.
- **Finding:** at the median, Bartz overpredicts h by about 56%. The bias is largest at and after the throat.
- **Fitted correction:**
  - single multiplier: **0.64** (95% CI 0.63–0.66)
  - position-dependent: about **0.83** in the convergent section, **0.58** at the throat, **0.59** in the divergent section
- **Accuracy on tests left out of the fit:** typical error drops from 64% uncorrected to 19% with the position-dependent correction.
- **Limits:** the calibration doesn't hold below about 100 psia, where the boundary layer partly turns laminar. The throat radius of curvature used (0.625 in) still needs confirming from the report drawing.

Full method, code and data are in `bartz_calibration_package/`.

### Solution, part 2: uncertainty quantification as a risk measure (proposed)
The calibration gives a *range* for the multiplier, not just a best value. Running the solver over that range turns a single peak wall temperature into a distribution, which can then be used to assess risk.

Example with a 1700 K material limit:

| | Peak wall temp | 90% interval | Chance of exceeding 1700 K |
|---|---|---|---|
| **Design A** | 1600 K | 1570–1630 K | about 0%, comfortably safe |
| **Design B** | 1600 K | 1400–1800 K | about 1 in 5 |

Both designs have the same headline number, but only Design A is safe. The current code, which reports a single value, can't tell them apart.

**Proposed steps:**
1. Sample `bartz_multiplier` from the calibrated distribution, using the position-dependent version where possible.
2. Sample other uncertain inputs as well, such as wall conductivity, emissivity and external h.
3. Run the steady and/or transient solver for each sample and report peak temperatures as intervals and exceedance probabilities.
4. Use a sensitivity analysis (e.g. Sobol indices) to identify which inputs drive most of the spread.
