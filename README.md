# Decode the Bioreactor: Team 04

SDSC STEM4Health Hackathon, Project 1. For 12 fed-batch test runs we predict **living cells (VCD), total cells (TCD), substrate and byproduct** at the 13 samples after day 0. Each test run gives us only its day-0 measurements, its reactor volumes and its full feed log. The score is the mean absolute percentage error (MAPE), averaged over the four quantities.

**Key idea: the feed log is the operators' diary.** The operators chose every feed *after* looking at the measurements we can't see, so the log still carries their trace. Three first-principles facts turn that trace back into predictions:

1. **Substrate obeys a mass balance.** Tomorrow's amount is today's amount, diluted, plus what the feeds bring in, minus what the cells eat. Fitted per day, it explains 57–94 % of the run-to-run variation of the next reading.
2. **Substrate boluses follow two fixed thresholds.** After the samples of days 7, 8 and 9, a bolus is given only below 1.95. On days 11 and 12 there is always a bolus, and it is large below 0.97. All 245 training decisions follow this rule.
3. **Unrecorded volume is base.** In the first days the volume rises more than the recorded feeds explain. This is base added against lactic acid, so it tracks the byproduct (r = 0.66).

## Results

| MAPE (%) | VCD | TCD | Substrate | Byproduct | **Mean** |
|---|---|---|---|---|---|
| Leaderboard, 12 test runs (22 Nov, 10:56 UTC) | 6.747 | 6.550 | 6.633 | **6.969** | **6.725** |
| Leave-one-run-out, 49 training runs | 6.28 | 6.21 | 5.97 | 8.51 | 6.75 ± 0.25 |
| Baseline: average training curve (leave-one-run-out) | 8.77 | 8.80 | 11.67 | 11.86 | 10.27 |

Leave-one-run-out means each training run is hidden once and predicted from the other 48, exactly like a test run. Everything that learns from data is refitted each time: shapes, regressions, mass balances, thresholds and clue links.

## The model

### VCD and TCD: one run factor, predicted twice

All runs follow the same recipe, so their log curves share one main pattern. A run is the typical curve plus a run-specific multiple of a fixed shape:

$$\ln y_{r,k} = \mu_k + z_r\, v_k + g_k\,(t_{r,k} - \bar t_k) + \varepsilon_{r,k}, \qquad k = 1,\dots,13$$

- **μ** (typical log curve), **v** (shape) and the training runs' **run factors** $z_r$ come from a singular value decomposition of the 49 × 13 matrix of log curves. One number per run captures most of what is predictable; the rest is mostly counting noise.
- **Sampling-time step** $g_k(t_{r,k} - \bar t_k)$. Samples of one "day" are taken at slightly different clock times. In the first days the culture roughly doubles per day, so a sample taken one hour early reads about 3 % lower. Every count is moved to the usual sampling time of its day before fitting, and every prediction is moved back to the run's own sampling times. $g_k$ is the slope of the typical log curve; nothing is fitted.
- **Predicting z for a test run** (its curve is hidden):
  - a **Gaussian process** (RBF kernel, marginal-likelihood fit) on the five standardised day-0 values: log VCD, viability, substrate, log byproduct, volume. It follows the curved seeding effect (flat below a day-0 VCD of about 1.0, a ramp to about 1.25, flat above).
  - a **ridge regression** on day-0 VCD, day-0 byproduct, day-0 volume, the mid-run Feed B level and whether a bolus was needed on days 7–9 (inputs held within ±3 SD). A straight line continues sensibly outside the training range, where the GP falls back to the average run; 5 test runs start outside that range.
- **Combination.** Each half gives a curve, multiplied per day by the factor in [0.75, 1.10] that minimises MAPE on the training runs (MAPE punishes over-prediction more). The two curves are averaged in log space (geometric mean). TCD is kept at or above VCD.
- **Why log space:** MAPE scores ratios (|ŷ/y − 1|), and a log difference is a ratio. The counting noise is also constant in log space, and cell growth is multiplicative.

### Substrate: a hidden-state model

The reading of a test run is hidden at 13 samples. The model keeps a *belief*, a probability for every value from 0.04 to 8, and updates it with two kinds of knowledge:

- **Moves (mass balance), fitted per day on the training runs:**

  $$G_{k+1} = a_k + b_k\,\frac{G_k V_k + 300\,F^S_k}{V_{k+1}} + \gamma_k(\Delta t_k - 1) + d_k\,\frac{F^B_k}{V_{k+1}} + \varepsilon_k,\qquad \varepsilon_k\sim N(0, s_k^2)$$

  The substrate feed holds about 300 units per litre, Feed B about 50 on most days, and from day 3 the culture eats 3–5 units per litre per day on most days.
- **Clues (operator decisions):** bolus decisions after days 7, 8, 9, 11 and 12 say on which side of the thresholds (1.95, 0.97) the reading was. On days 2–6, the Feed B rate was raised when substrate was low.
- **Forward–backward pass** (as in hidden Markov models) combines moves and clues, so a clue on day 12 also sharpens days 11 and 10. The prediction is the middle of the belief, tilted halfway towards low values for MAPE.

### Byproduct: one run factor from the missing volume

The same one-factor form in log space, with its own typical curve and a shape that grows over time (runs drift apart). The run factor comes from a single ridge regression on the five day-0 values and the **unexplained volume of the first five intervals** ($V_{k+1} - V_k$ minus all recorded feeds). Around day 2 the volume rises by up to about 40 mL more than the feeds explain: base added against lactic acid. Runs that needed more of it stay higher in byproduct to the end.

## Reproduce

**Data.** The challenge data is not in this repository (as the organisers asked). The code looks for `bioreaction_remaining.csv` and `bioreaction_template.csv` under `input-data/` next to or above the repository, which is where a hackathon Renku session mounts it (`input-data/Project 1 - DTB/`). Elsewhere, pass the folder with `--data-dir`.

**Environment.** Python 3.11 with the packages in `requirements.txt`; the hackathon's Renku session (`environment.yml`) already has them.

```bash
pip install -r requirements.txt        # not needed in the hackathon session
python src/bioreactor_model.py --cv    # about 15 s; prints the validation table, writes outputs/submission.csv
```

- `python src/bioreactor_model.py` alone only writes the submission file (about 4 s).
- The same model with explanations, validation and figures: `notebooks/bioreactor_final.ipynb` (Run All, about 30 s). It is saved with `SUBMIT = False`, so it never contacts the leaderboard unless you switch that on.
- The run is deterministic. Expected output: the validation table above and an `outputs/submission.csv` identical to the committed file, `sha256 b655ed8529cf43cbbcf7f830ed291b68eaa83b0986b7ef5b1a7a53233c6c5ff7`. This is the file that scored 6.725 on the leaderboard.

## Repository

```text
.
├── README.md                        this file
├── requirements.txt                 package versions the model was reproduced with
├── environment.yml                  hackathon session environment (from the organisers)
├── src/bioreactor_model.py          final model as a script (model code verbatim from the notebook)
├── notebooks/bioreactor_final.ipynb final model with explanations, validation and figures
├── outputs/submission.csv           our final leaderboard submission (12 test runs × 13 samples)
├── docs/
│   ├── project_summary.md           short project summary
│   └── hackathon_setup.md           the organisers' original setup guide
└── Brewfile, project.toml, init.d/  Renku session setup (from the organisers)
```

## What did not help

All were compared run by run in leave-one-run-out validation; many were also checked on the 20 training runs that resemble the test set, and some in an extrapolation test (the most extreme runs held out together).

- **Model form for VCD/TCD:** polynomials and linear models instead of the GP (+0.2), other kernels (Matérn, rational quadratic, per-input length-scales), stronger regularisation, per-day models (+0.36), a mechanistic growth ODE from the day-0 state alone (7.9 % instead of 6.3 %), tree ensembles, PLS, bagging.
- **Preprocessing:** outlier capping, denoising, other transforms of the inputs. The log scale and the one-factor projection already do the useful part.
- **More inputs for VCD/TCD:** most feed-log clues did not survive validation. A version with Feed C doses validated well but scored worse on the leaderboard (7.275).
- **A richer byproduct model** (two factors and more feed clues) was better in validation (7.49 % vs 8.51 %) but worse on the leaderboard (7.974 vs 6.969). We kept the simple one.

The limiting factor is information, not model complexity. Counting scatter alone sets a floor of about 5.5 % mean error, which no model can beat on these measurements.

## Limitations

- **The byproduct model was chosen on the 12 leaderboard runs**, against the validation result. A leaderboard score of 12 runs carries about ±0.5 of sampling noise.
- **The sampling-time step gained far more on the leaderboard (−0.16) than in validation (−0.02 to −0.03 ± 0.035).** Some test runs were sampled unusually early; part of the gain is probably luck.
- **The bolus rule is an observation**, assumed to hold for the test runs.
- **The model reads the whole feed log, including decisions made after the predicted day.** That suits this challenge (reconstructing finished runs). For real-time use, only the forward pass of the substrate model would apply.
- Units and chemical identities are not given; substrate behaves like glucose and byproduct like lactate, but the model does not depend on that.
