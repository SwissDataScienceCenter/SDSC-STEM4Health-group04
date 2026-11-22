"""Decode the Bioreactor (SDSC STEM4Health hackathon, Project 1), Team 04: final model.

Predicts VCD, TCD, substrate and byproduct at the 13 later samples of each test run from the day-0
measurements, the reactor volumes and the feed log, and writes the submission file.

The model code below is taken verbatim from notebooks/bioreactor_final.ipynb (sections 1-3 and 6); only the
command-line entry point at the end is new. This script never contacts the leaderboard.

Usage (from the repository root):
    python src/bioreactor_model.py                       # writes outputs/submission.csv
    python src/bioreactor_model.py --cv                  # also runs the leave-one-run-out validation (about 15 s)
    python src/bioreactor_model.py --data-dir "/path/to/Project 1 - DTB" --out my_submission.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
NAMES = {"vcd": "VCD", "tcd": "TCD", "substrate": "Substrate", "byproduct": "Byproduct"}
DAYS = list(range(1, 14))                              # the 13 samples to predict


# ======================================================================== 1. data and inputs
def load(path):
    df = pd.read_csv(path)
    df["to_predict"] = df["to_predict"].astype(str).str.strip().str.lower().eq("true")
    samples = df[df["volume"].notna()].copy()                    # sample rows carry a volume, feed rows do not
    samples["k"] = samples.groupby("batch_id").cumcount()        # sample index 0..13
    feeds = df[df["volume"].isna()].copy()
    runs = list(dict.fromkeys(df["batch_id"]))
    test = [r for r in runs if samples.loc[samples["batch_id"] == r, "to_predict"].any()]
    train = [r for r in runs if r not in test]
    assert (samples.groupby("batch_id").size() == 14).all(), "every run should have 14 samples"
    wide = lambda col: samples.pivot(index="batch_id", columns="k", values=col).loc[runs]
    Y, T, V = {c: wide(c) for c in TARGETS}, wide("elapsed_time"), wide("volume")
    F = {name: pd.DataFrame(0.0, index=runs, columns=range(13)) for name in ("feed_b", "feed_substrate", "feed_c", "feed_d")}
    for run, g in feeds.groupby("batch_id"):
        k = np.clip(np.searchsorted(T.loc[run].values, g["elapsed_time"].values, side="left") - 1, 0, 12)   # interval k = from sample k to sample k+1
        for name in F:
            F[name].loc[run] = np.bincount(k, weights=g[name].fillna(0.0).values, minlength=13)
    dV = V.diff(axis=1).iloc[:, 1:]; dV.columns = range(13)
    gap = dV - (F["feed_b"] + F["feed_substrate"] + F["feed_c"] + F["feed_d"])
    return dict(samples=samples, runs=runs, train=train, test=test, Y=Y, T=T, V=V, FB=F["feed_b"], FS=F["feed_substrate"], gap=gap)


def bolus_category(D):
    """0 = no substrate bolus after the sample, 1 = small (0.4 % of the volume), 2 = large (0.6 %)"""
    frac = D["FS"] / D["V"].iloc[:, :13].values
    return (frac > 0).astype(int) + (frac > 0.005).astype(int)


def features(D):
    Y, T, V, FB, gap = D["Y"], D["T"], D["V"], D["FB"], D["gap"]
    X = pd.DataFrame(index=D["runs"])
    X["lvcd0"], X["viab0"], X["sub0"] = np.log(Y["vcd"][0]), Y["vcd"][0] / Y["tcd"][0], Y["substrate"][0]
    X["lbyp0"], X["V0"] = np.log(Y["byproduct"][0]), V[0]
    rate = pd.DataFrame({k: np.log(FB[k] / V[k] / (T[k] - T[k - 1])) for k in range(1, 13)})     # Feed B per litre and per day (log)
    dev = rate - rate.mean()
    X["fb_mid"] = dev[[4, 5, 6, 7, 8]].mean(axis=1)
    X["inn1"] = dev[1]
    for k in range(2, 13):                                                                       # day-to-day change of the Feed B rate
        a, b = dev[k], dev[k - 1]
        X[f"inn{k}"] = a - (a * b).sum() / (b * b).sum() * b
    cat = bolus_category(D)
    for k in (7, 8, 9):
        X[f"fsX{k}"] = (cat[k] > 0).astype(float)
    X["fsXany"] = X[["fsX7", "fsX8", "fsX9"]].max(axis=1)
    for k in range(5):
        X[f"gap{k}"] = gap[k]
    assert not X.isna().any().any(), "an input is missing"
    return X


DAY0 = ["lvcd0", "viab0", "sub0", "lbyp0", "V0"]


# ======================================================================== 2. helpers
def standardise(A, B, clip=None):
    m, s = A.mean(0), A.std(0)
    keep = s > 1e-9                                    # an input that is constant in the training runs is ignored
    s = np.where(keep, s, 1.0)
    A, B = (A - m) / s, (B - m) / s * keep
    return (np.clip(A, -clip, clip), np.clip(B, -clip, clip)) if clip else (A, B)


def mape_factor(y, pred, lo=0.75, hi=1.10):
    f = np.linspace(lo, hi, int(round((hi - lo) / 0.005)) + 1)
    return f[int(np.argmin([np.mean(np.abs(y - c * pred) / y) for c in f]))]


# ======================================================================== 3.1 VCD and TCD: one run factor, GP + ridge, sampling-time step
from scipy.interpolate import PchipInterpolator

CELL_LINEAR = ["lvcd0", "lbyp0", "V0", "fb_mid", "fsXany"]


def growth_rate(c, D, train):
    """Slope of the typical log-curve (per day) at the usual sampling time of days 1 to 13, and those usual times."""
    usual = D["T"].loc[train].mean().values                        # usual sampling time of the 14 samples, day 0 included
    typical = np.log(D["Y"][c].loc[train]).mean().values
    return PchipInterpolator(usual, typical).derivative()(usual[1:]), usual[1:]


def cell_model(c, D, X, train, target, time_correction=True):
    A = D["Y"][c].loc[train, DAYS].values
    rate, usual = growth_rate(c, D, train)
    if not time_correction:                                        # the model of our 6.804 entry
        rate = np.zeros(13)
    late_train = D["T"].loc[train, DAYS].values - usual            # days a sample was taken later (+) or earlier (-) than usual
    late_target = D["T"].loc[target, DAYS].values - usual
    L = np.log(A) - rate * late_train                              # every training count moved to the usual sampling time
    typical = L.mean(0)
    U, S, Vt = np.linalg.svd(L - typical, full_matrices=False)
    z, shape = U[:, 0] * S[0], Vt[0]
    members = []
    Fa, Fb = standardise(X.loc[train, DAY0].values, X.loc[target, DAY0].values)
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(3.0, (0.5, 50)) + WhiteKernel(0.5, (1e-2, 5)), normalize_y=True).fit(Fa, z)
    members.append((gp.predict(Fa), gp.predict(Fb)))
    Fa, Fb = standardise(X.loc[train, CELL_LINEAR].values, X.loc[target, CELL_LINEAR].values, clip=3.0)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 31)).fit(Fa, z)
    members.append((ridge.predict(Fa), ridge.predict(Fb)))
    log_pred = 0.0
    for z_fit, z_new in members:
        fitted = np.exp(typical + np.outer(z_fit, shape) + rate * late_train)      # back at each run's own sampling times
        pred = np.exp(typical + np.outer(z_new, shape) + rate * late_target)
        log_pred = log_pred + np.log(pred * np.array([mape_factor(A[:, j], fitted[:, j]) for j in range(13)])) / len(members)
    return np.exp(log_pred)


# ======================================================================== 3.2 substrate: hidden-state model
GRID_G = np.arange(0.04, 8.0001, 0.02)     # possible substrate readings
C_BOLUS = 300.0                             # substrate per litre of substrate feed; the fitted value is 300 to 320


def substrate_fit(D, X, train):
    G, V, T, FB, FS = D["Y"]["substrate"], D["V"], D["T"], D["FB"], D["FS"]
    cat = bolus_category(D)
    M = {"move": {}, "clue": {}}
    for k in range(13):                     # mass balance from sample k to sample k+1
        g_after = ((G[k] * V[k] + C_BOLUS * FS[k]) / V[k + 1]).loc[train].values
        cols = [g_after, (T[k + 1] - T[k]).loc[train].values - 1.0]
        if k > 0:
            cols.append((FB[k] / V[k + 1]).loc[train].values)
        A = np.column_stack([np.ones(len(train))] + cols); y = G.loc[train, k + 1].values
        b = np.linalg.lstsq(A, y, rcond=None)[0]; res = y - A @ b
        leverage = np.einsum("ij,jk,ik->i", A, np.linalg.pinv(A.T @ A), A)
        M["move"][k] = (b, float(np.sqrt(np.mean((res / (1 - leverage)) ** 2))))                 # leave-one-out spread
    for k in range(2, 13):                  # Feed B was raised when substrate was low: change of rate ~ a + b * G[k]
        x, y = G.loc[train, k].values, X.loc[train, f"inn{k}"].values
        A = np.column_stack([np.ones(len(train)), x]); b = np.linalg.lstsq(A, y, rcond=None)[0]
        s = (y - A @ b).std(ddof=2)
        if b[1] / (s / np.sqrt(((x - x.mean()) ** 2).sum())) < -2.5:                             # keep a clue only where the link is clear
            M["clue"][k] = (b, float(s))
    low = [(G.loc[r, k], cat.loc[r, k]) for r in train for k in (11, 12)]
    high = [(G.loc[r, k], cat.loc[r, k]) for r in train for k in (7, 8, 9)]
    M["th_low"] = 0.5 * (max(g for g, c in low if c == 2) + min(g for g, c in low if c == 1))      # below: large bolus
    M["th_high"] = 0.5 * (max(g for g, c in high if c >= 1) + min(g for g, c in high if c == 0))   # below: a bolus at all
    return M


def substrate_belief(run, D, X, M, jitter=0.012):
    """Probability of every possible reading at each of the 14 samples of one run (rows = samples, columns = GRID_G)."""
    G, V, T, FB, FS = D["Y"]["substrate"], D["V"], D["T"], D["FB"], D["FS"]
    cat = bolus_category(D); g = GRID_G; n = len(g)

    def clue(k):
        e = np.ones(n)
        if k in (7, 8, 9, 11, 12):
            below_low = norm.cdf((np.log(M["th_low"]) - np.log(g)) / jitter)
            below_high = norm.cdf((np.log(M["th_high"]) - np.log(g)) / jitter)
            e = e * (1 - below_high, below_high - below_low, below_low)[int(cat.loc[run, k])]
        if k in M["clue"]:
            b, s = M["clue"][k]
            e = e * norm.pdf(X.loc[run, f"inn{k}"], b[0] + b[1] * g, s)
        return e + 1e-300

    P = []
    for k in range(13):
        b, s = M["move"][k]
        mu = b[0] + b[1] * (g * V.loc[run, k] + C_BOLUS * FS.loc[run, k]) / V.loc[run, k + 1] + b[2] * (T.loc[run, k + 1] - T.loc[run, k] - 1.0)
        if k > 0:
            mu = mu + b[3] * FB.loc[run, k] / V.loc[run, k + 1]
        K = norm.pdf(g[None, :], mu[:, None], s); K = K / (K.sum(1, keepdims=True) + 1e-300)
        P.append((1 - 1e-6) * K + 1e-6 / n)             # tiny floor: no reading is ever ruled out completely
    fwd = np.zeros((14, n)); fwd[0, np.argmin(np.abs(g - G.loc[run, 0]))] = 1.0      # day 0 is known
    for k in range(13):
        v = (fwd[k] @ P[k]) * clue(k + 1); fwd[k + 1] = v / v.sum()
    bwd = np.ones((14, n))
    for k in range(12, -1, -1):
        v = P[k] @ (bwd[k + 1] * clue(k + 1)); bwd[k] = v / v.sum()
    belief = fwd * bwd
    return belief / belief.sum(1, keepdims=True)


def belief_quantile(belief, q, tilt=0.0):
    w = belief / GRID_G[None, :] ** tilt
    c = np.cumsum(w, 1) / w.sum(1, keepdims=True)
    return np.array([np.interp(q, c[k], GRID_G) for k in range(belief.shape[0])])


def substrate_model(D, X, train, target, tilt=0.5):
    """tilt 0 = median of the belief, tilt 1 = the exact MAPE-optimal point if the belief were perfectly calibrated"""
    M = substrate_fit(D, X, train)
    return np.array([belief_quantile(substrate_belief(r, D, X, M), 0.5, tilt)[1:] for r in target])


# ======================================================================== 3.3 byproduct, and all four models together
BYPRODUCT_INPUTS = DAY0 + [f"gap{k}" for k in range(5)]


def byproduct_model(D, X, train, target):
    A = D["Y"]["byproduct"].loc[train, DAYS].values
    L = np.log(A); typical = L.mean(0)
    U, S, Vt = np.linalg.svd(L - typical, full_matrices=False)
    z, shape = U[:, 0] * S[0], Vt[0]
    Fa, Fb = standardise(X.loc[train, BYPRODUCT_INPUTS].values, X.loc[target, BYPRODUCT_INPUTS].values)
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(Fa, z)
    fitted, pred = np.exp(typical + np.outer(ridge.predict(Fa), shape)), np.exp(typical + np.outer(ridge.predict(Fb), shape))
    return pred * np.array([mape_factor(A[:, j], fitted[:, j]) for j in range(13)])


def predict(D, X, train, target):
    """Fit on the training runs, predict the target runs. Returns {variable: DataFrame(runs x days 1..13)}."""
    out = {"vcd": cell_model("vcd", D, X, train, target), "tcd": cell_model("tcd", D, X, train, target)}
    out["tcd"] = np.maximum(out["tcd"], out["vcd"])              # total cells cannot be fewer than viable cells
    out["substrate"] = substrate_model(D, X, train, target)
    out["byproduct"] = byproduct_model(D, X, train, target)
    return {c: pd.DataFrame(out[c], index=target, columns=DAYS) for c in TARGETS}


# ======================================================================== 4. leave-one-run-out validation
def cross_validate(D, X):
    train = D["train"]
    P = {c: pd.DataFrame(index=train, columns=DAYS, dtype=float) for c in TARGETS}
    for held in train:
        p = predict(D, X, [r for r in train if r != held], [held])
        for c in TARGETS:
            P[c].loc[held] = p[c].loc[held].values
    return P


def error_table(D, P):
    truth = lambda c: D["Y"][c].loc[P[c].index, DAYS].values
    return {c: np.abs(truth(c) - P[c].values) / truth(c) * 100 for c in TARGETS}


# ======================================================================== 6. submission file
def make_submission(P, D, template_path):
    template = pd.read_csv(template_path, dtype={"batch_id": str})
    assert list(template.columns) == ["batch_id", "elapsed_time"] + TARGETS, "unexpected template columns"
    assert set(template["batch_id"]) == set(P["vcd"].index), "template runs differ from the test runs"
    rows = []
    for run, t in zip(template["batch_id"], template["elapsed_time"]):
        times = D["T"].loc[run].values; k = int(np.argmin(np.abs(times - t)))
        assert k >= 1 and abs(times[k] - t) < 1e-4, f"timestamp {t} of {run} is not a sample to predict"
        rows.append([P[c].loc[run, k] for c in TARGETS])
    template[TARGETS] = np.array(rows)
    assert template[TARGETS].notna().all().all() and (template[TARGETS] > 0).all().all() and len(template) == 13 * len(P["vcd"])
    return template


# ======================================================================== command-line entry point (new)
REPO = Path(__file__).resolve().parent.parent


def find_file(pattern, data_dir=None):
    """Look in --data-dir, then in the usual places of a Renku session (input-data next to or above the repository)."""
    roots = [Path(data_dir)] if data_dir else [REPO / "input-data", REPO.parent / "input-data", Path("input-data"), Path("../input-data")]
    for root in roots:
        if root.exists():
            hits = sorted(root.rglob(pattern))
            if hits:
                return hits[0]
    raise FileNotFoundError(f"Could not find {pattern}. Pass the challenge data folder with --data-dir.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", help='folder with bioreaction_remaining.csv and bioreaction_template.csv (default: search input-data)')
    ap.add_argument("--out", default=str(REPO / "outputs" / "submission.csv"), help="submission file to write")
    ap.add_argument("--cv", action="store_true", help="also run the leave-one-run-out validation on the 49 training runs")
    args = ap.parse_args()

    data_path, template_path = find_file("*bioreaction_remaining*.csv", args.data_dir), find_file("*bioreaction_template*.csv", args.data_dir)
    print("data:    ", data_path)
    print("template:", template_path)
    D = load(data_path)
    X = features(D)
    train, test_ids = D["train"], D["test"]
    print(f"{len(train)} training runs, {len(test_ids)} test runs, {X.shape[1]} inputs")

    if args.cv:
        P_cv = cross_validate(D, X)
        P_base = {c: pd.DataFrame([D["Y"][c].loc[[r for r in train if r != held], DAYS].mean().values for held in train], index=train, columns=DAYS) for c in TARGETS}
        E, E_base = error_table(D, P_cv), error_table(D, P_base)
        table = pd.DataFrame({"baseline: average curve": {NAMES[c]: E_base[c].mean() for c in TARGETS}, "this model": {NAMES[c]: E[c].mean() for c in TARGETS}})
        table.loc["SCORE (mean of the four)"] = table.mean()
        per_run = np.mean([E[c].mean(axis=1) for c in TARGETS], axis=0)
        print("\nleave-one-run-out MAPE (%):\n" + table.round(2).to_string())
        print(f"standard error of the score over the {len(train)} runs: +/- {per_run.std(ddof=1) / np.sqrt(len(per_run)):.2f}\n")

    P_test = predict(D, X, train, test_ids)
    submission = make_submission(P_test, D, template_path)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(submission)} rows; TCD never below VCD: {bool((submission.tcd >= submission.vcd).all())}")


if __name__ == "__main__":
    main()
