# Calibrating the Bartz correlation against Back, Massier & Gier (1965)

## Data
- Source: Back, Massier & Gier, *Convective Heat Transfer in a Convergent-Divergent Nozzle*, JPL Technical Report 32-415, Appendix B (NASA NTRS 19650010083). Water-cooled 30°/15° conical nozzle, 1.803 in throat diameter, air heated by methanol combustion.
- The values were transcribed from the report's printed tables (Tables B-2 and B-3), not digitised from plots. `transcribe_bmg.py` holds every value as printed.
- 11 tests and 221 stations. Each row is checked for internal consistency: the printed heat flux, wall temperature and h must imply the expected adiabatic wall temperature. 220 rows agree within 3% (median 0.5%), and the one that fails is excluded.
- **Training set:** 9 fully turbulent tests at 100–254 psia (184 points).
- **Held out:** tests 234 (75 psia) and 301 (45 psia). Back et al. report reduced heat transfer below about 75 psia, so these are used only as an out-of-regime check.

## Method
1. Run Matthew's `NozzleModel.bartz_h` (unchanged) at each test's conditions and the measured wall temperatures, using the nozzle contour from Table B-2.
2. Model the response y = ln(h_measured / h_Bartz). The correction factor C = exp(ŷ) acts as a `bartz_multiplier`.
3. Choose between models by leave-one-test-out cross-validation.
4. Compute 95% CIs with a cluster bootstrap that resamples whole tests (4,000 draws), because points within a test are correlated.

## Results
| | Value |
|---|---|
| Median h_measured / h_Bartz | **0.64** (range 0.42–1.00); Bartz overpredicts by about 56% at the median |
| Median q_measured / q_Bartz | 0.62 |
| Best single multiplier | **C = 0.64** (95% CI 0.63–0.66) |
| Position model (chosen) | ln C = −0.548 − 0.258·min(s,0) + 0.018·max(s,0), where s = −ln(A/A*) subsonic and +ln(A/A*) supersonic |
| C at convergent A/A* = 4 | 0.83 (0.78–0.87) |
| C at throat | **0.58** (0.57–0.59) |
| C at divergent A/A* = 2.5 | 0.59 (0.56–0.62) |
| Leave-one-test-out error | uncorrected 64% → single multiplier 26% → position model 19% |

- Adding an inlet boundary-layer term or a pressure term did not improve cross-validated error, so the simpler position model is preferred.
- The CIs describe the *mean* correction. Individual stations scatter about ±19% around it, from circumferential variation and plug-to-plug differences.
- The held-out low-pressure tests fall a further 22–24% below the turbulent calibration. This is consistent with Back et al.'s relaminarisation explanation: **do not apply this calibration below about 100 psia for a nozzle of this size.**

## Caveats
- **Throat radius of curvature** was taken as 0.625 in (rc/rt ≈ 0.69), a value used in CFD reproductions of this nozzle, but not confirmed in the report text. Bartz h scales with rc^0.1, so the single multiplier moves from 0.62 to 0.71 across rc/rt = 0.5–2. Confirm rc from the report's drawing before relying on the third decimal place.
- **Single facility.** This is one nozzle (air, about 600–850 K stagnation temperature). Rocket conditions are hotter, with combustion products and radiation, so treat the factors as evidence of the size and shape of Bartz's bias rather than universal constants.
- **Shape near the throat.** The measured ratio stays near 0.75–0.8 until just upstream of the throat and then drops sharply. A smooth function of A/A* can't capture that fully. Part of the drop is a 2-D effect that the 1-D model can't represent: Back et al. measured mass flux up to 15% below the 1-D value just downstream of the throat.
- Gas properties are for air from CoolProp at stagnation temperature; γ comes from the report.

## Files
- `transcribe_bmg.py` → `bmg_heat_transfer_data.csv`: transcribed data in SI units, with QA columns.
- `predict_bartz.py`: runs `bartz_nozzle.NozzleModel` at the test conditions.
- `calibrate.py` → `bmg_vs_bartz.csv`, `calibration_summary.json`, `bartz_calibration.png`.
- Run order: `python transcribe_bmg.py && python calibrate.py` (needs numpy, scipy, pandas, matplotlib, CoolProp, and `bartz_nozzle.py` in the same folder).
