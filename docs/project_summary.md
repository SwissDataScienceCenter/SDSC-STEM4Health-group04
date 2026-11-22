### Decode the Bioreactor: Team 04

We predict VCD, TCD, substrate and byproduct at 13 samples for 12 fed-batch runs, using only day-0 measurements, volumes and the feed log. **Key idea: the feed log is the operators' diary.** Feeds were chosen after the hidden measurements were seen, so the log carries their trace.

- **Three facts:**
  - Substrate follows a mass balance.
  - Bolus decisions follow two fixed thresholds (245 of 245 training decisions).
  - Unexplained volume is base added against acid, and it tracks the byproduct.
- **VCD/TCD:** in log space, a typical curve plus one run factor times a shared shape. The run factor is the average of a Gaussian process (day-0 state) and a ridge regression. Counts are aligned to the usual sampling time.
- **Substrate:** a hidden-state model. The mass balance moves a probability distribution, operator decisions sharpen it, and a forward–backward pass combines them.
- **Byproduct:** a one-factor model, with the run factor from ridge on the day-0 state and the early unexplained volume.
- **Results:** leave-one-run-out **6.75 %** mean MAPE (baseline 10.27 %); leaderboard **6.725 %**, with the best byproduct score (6.97).
- **Lessons:** noise floor about 5.5 %. Information, not model complexity, drove accuracy: ODEs, other kernels and trees did not help.
- **Limitations:**
  - The byproduct model was chosen on 12 leaderboard runs.
  - The bolus rule is assumed for test runs.
  - Using later feed decisions makes this a retrospective reconstruction, not a real-time forecast.
