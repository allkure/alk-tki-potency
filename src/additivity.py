"""Compound-mutation resistance predicted from component single mutants.

The utility demonstration in the data descriptor. For a compound variant C with components c_i,

    epsilon = lfc(C) - sum_i lfc(c_i),   lfc(v) = log10[IC50(v) / IC50(WT)]

with every lfc taken against the wild type of its own experimental context, and a compound used
only where all of its components were measured in that same context.

Gilteritinib is excluded throughout: it is a FLT3 inhibitor included in the dataset as an
off-target comparator, not an ALK-directed therapeutic, so it has no inhibitor generation and
does not belong in a generational analysis. Including it would add 10 observations and move the
fitted slope from 0.652 to 0.647.

    python3 -I src/additivity.py
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python3 -I too

import alkdata as A


def head(s):
    print(f"\n{s}\n{'=' * len(s)}")


def paired_diff(x, y_lo, y_hi, n=4000, seed=0):
    """Bootstrap the difference between two correlations measured on the same rows."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    obs = stats.spearmanr(x, y_hi).statistic - stats.spearmanr(x, y_lo).statistic
    b = []
    for _ in range(n):
        k = rng.choice(idx, len(idx), replace=True)
        if np.std(x[k]) == 0:
            continue
        b.append(stats.spearmanr(x[k], y_hi[k]).statistic
                 - stats.spearmanr(x[k], y_lo[k]).statistic)
    b = np.array(b)
    return obs, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), \
        max(2 * min((b <= 0).mean(), (b >= 0).mean()), 1 / len(b))


def table(d=None):
    """Decomposable compound observations, ALK-directed drugs only."""
    d = A.load() if d is None else d
    c = A.compound_table(d)
    return c[c["generation"].notna()].copy()


def calibrate(c):
    """Leave-one-drug-out calibration of observed ~ a + b * expected."""
    rows = []
    for drug in sorted(c["drug"].unique()):
        tr, te = c[c["drug"] != drug], c[c["drug"] == drug]
        lr = stats.linregress(tr["expected"], tr["observed"])
        pred = lr.intercept + lr.slope * te["expected"]
        rows.append({"drug": drug, "generation": int(te["generation"].iloc[0]),
                     "n": len(te), "a": lr.intercept, "b": lr.slope,
                     "mae_calibrated": float(np.mean(np.abs(pred - te["observed"]))),
                     "mae_naive": float(np.mean(np.abs(te["expected"] - te["observed"]))),
                     "rho": (stats.spearmanr(pred, te["observed"]).statistic
                             if len(te) > 3 else np.nan)})
    return pd.DataFrame(rows), stats.linregress(c["expected"], c["observed"])


def prioritisation_auc(d=None):
    """Can the model pick, out of all possible pairs, the compounds the field has studied?"""
    d = A.load() if d is None else d
    c = table(d)
    full = stats.linregress(c["expected"], c["observed"])
    lf = A.fold_change(d)
    studied = {"+".join(sorted(p)) for p in lf.loc[lf["n_mut"] == 2, "parts"]}

    rows = []
    for drug, s in lf[lf["is_single"]].groupby("drug_canonical"):
        if pd.isna(A.GENERATION.get(drug)):
            continue
        sing = s.groupby("variant")["lfc"].median()
        if len(sing) < 5:
            continue
        rec = []
        for a, b in itertools.combinations(sing.index, 2):
            if a[1:-1] == b[1:-1]:          # same position: cannot co-occur
                continue
            rec.append((full.intercept + full.slope * (sing[a] + sing[b]),
                        "+".join(sorted([a, b])) in studied))
        p = pd.DataFrame(rec, columns=["predicted", "studied"])
        if p["studied"].sum() < 3:
            continue
        rows.append({"drug": drug, "generation": A.GENERATION[drug], "pairs": len(p),
                     "studied": int(p["studied"].sum()),
                     "auc": A.auc(p["predicted"], p["studied"])})
    return pd.DataFrame(rows)


def main():
    d = A.load()
    c = table(d)
    head("Decomposable compound observations")
    print(f"  {len(c)} observations, {c['compound'].nunique()} unique compounds, "
          f"{c['drug'].nunique()} ALK-directed drugs, {c['paper'].nunique()} laboratories")

    head("1. Is resistance additive?")
    lo, hi = A.bootstrap_ci(c["epsilon"])
    print(f"  mean epsilon, all observations   {c['epsilon'].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    per = c.groupby("compound")["epsilon"].mean()
    lo, hi = A.bootstrap_ci(per)
    print(f"  mean epsilon, per compound       {per.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"(n = {len(per)})")
    print(f"  Wilcoxon signed-rank vs 0        P = {stats.wilcoxon(per).pvalue:.3g}")

    head("2. Compression: the diagnostic that must precede any cross-drug claim")
    r = stats.spearmanr(c["expected"], c["epsilon"])
    lr = stats.linregress(c["expected"], c["observed"])
    print(f"  epsilon vs expected   rho = {r.statistic:+.3f}  P = {r.pvalue:.2g}")
    print(f"  observed ~ expected   slope = {lr.slope:.3f} (1.0 would be additive), "
          f"r = {lr.rvalue:.3f}")
    print("\n  per laboratory, to show it is not one assay")
    for lab, s in c.groupby("paper"):
        if len(s) < 6:
            continue
        print(f"    {lab:<16}n = {len(s):<4} slope {stats.linregress(s.expected, s.observed).slope:.2f}")
    for assay, s in c.groupby("assay"):
        print(f"    {assay:<16}n = {len(s):<4} slope {stats.linregress(s.expected, s.observed).slope:.2f}")

    head("3. Is there a generational trend in epistasis?")
    print(f"  {'generation':<12}{'n':>5}{'mean eps':>11}{'mean expected':>15}")
    for g, s in c.groupby("generation"):
        print(f"  gen {int(g):<8}{len(s):>5}{s['epsilon'].mean():>+11.3f}"
              f"{s['expected'].mean():>+15.2f}")
    X = np.column_stack([np.ones(len(c)), c["generation"].to_numpy(float),
                         c["expected"].to_numpy(float)])
    y = c["epsilon"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = np.sum(resid ** 2) / dof * np.linalg.inv(X.T @ X)
    print("\n  regression of epsilon on generation AND expected sum:")
    for i, nm in enumerate(["intercept", "generation", "expected sum"]):
        t = beta[i] / np.sqrt(cov[i, i])
        print(f"    {nm:<14}{beta[i]:>+9.3f}  t = {t:+.2f}  P = {2 * stats.t.sf(abs(t), dof):.3g}")
    print("  -> the raw generational trend is compression, not epistasis.")

    head("4. Prediction: leave-one-drug-out")
    loo, full = calibrate(c)
    print(f"  full fit: observed = {full.intercept:+.3f} + {full.slope:.3f} x expected")
    print(f"\n  {'held-out drug':<14}{'gen':>4}{'n':>4}{'naive MAE':>11}{'calibrated':>12}{'rho':>7}")
    for _, r_ in loo.sort_values("mae_calibrated").iterrows():
        print(f"  {r_['drug']:<14}{r_['generation']:>4}{r_['n']:>4}"
              f"{r_['mae_naive']:>11.2f}{r_['mae_calibrated']:>12.2f}{r_['rho']:>7.2f}")
    print(f"\n  overall naive      {np.mean(np.abs(c.expected - c.observed)):.2f} log10 "
          f"({10 ** np.mean(np.abs(c.expected - c.observed)):.1f}-fold)")
    cal = np.abs(full.intercept + full.slope * c.expected - c.observed).mean()
    print(f"  overall calibrated {cal:.2f} log10 ({10 ** cal:.1f}-fold)")

    head("5. Where it fails: ranking which compounds matter")
    pa = prioritisation_auc(d)
    print(f"  {'drug':<14}{'gen':>4}{'pairs':>7}{'studied':>9}{'AUC':>7}")
    for _, r_ in pa.sort_values(["generation", "auc"]).iterrows():
        print(f"  {r_['drug']:<14}{r_['generation']:>4}{r_['pairs']:>7}{r_['studied']:>9}"
              f"{r_['auc']:>7.3f}")
    g4 = pa[pa["generation"] == 4]["auc"]
    print(f"\n  generations 1-3: AUC {pa[pa.generation < 4]['auc'].min():.2f}"
          f"-{pa[pa.generation < 4]['auc'].max():.2f}")
    print(f"  generation 4:    AUC {g4.min():.2f}-{g4.max():.2f}  (chance = 0.50)")
    print("  Fourth-generation inhibitors were designed so single mutations barely shift")
    print("  potency, leaving component values too flat to rank combinations.")


if __name__ == "__main__":
    sys.exit(main())
