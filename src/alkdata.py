"""Loading and context-aware analysis of the ALK-TKI potency dataset.

Everything else in this repository imports from here, so the rules that make the dataset
internally valid are written once.

The central rule: a potency measurement is only comparable with another measured in the same
experimental context. Context here means paper, value type, assay class, cell line, expression
construct and drug. Fold change is always computed against the wild type of a measurement's own
context, never against a pooled wild type, because the seven source laboratories differ
systematically by up to 0.34 log10. `fold_change()` enforces this; do not recompute it by hand.

    from alkdata import load, fold_change, compound_table
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "alk_tki_potency_dataset_v0.csv")

#: MD5 of the file as published at doi:10.5281/zenodo.22914209. Checked on every load so an
#: edited or truncated copy cannot silently change a published number.
PUBLISHED_MD5 = "e3d1c306ef16774548cb6b53bd065c99"

#: One experimental context. Two measurements may be compared only within the same tuple.
CONTEXT = ["paper_id", "value_type", "assay_class", "cell_line", "expression_construct",
           "drug_canonical"]

#: The harmonized subset the published concordance statistics are computed on. Both the cell
#: line AND the construct are required; filtering on cell line alone gives a wider set whose
#: counts do not match the paper.
HARMONIZED = {"value_type": "Cellular_IC50", "cell_line": "Ba/F3",
              "expression_construct": "EML4-ALK_v1"}

#: Inhibitor generation as the field uses the term. Gilteritinib is a FLT3 inhibitor included
#: as an off-target comparator, not an ALK-directed therapeutic; it carries no generation.
GENERATION = {"crizotinib": 1,
              "ceritinib": 2, "alectinib": 2, "brigatinib": 2, "ensartinib": 2,
              "entrectinib": 2,
              "lorlatinib": 3, "repotrectinib": 3,
              "zotizalkib": 4, "neladalkib": 4}

SINGLE = re.compile(r"^[A-Z]\d{3,4}[A-Z]$")


def checksum(path: str = CSV) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str = CSV, gene: str = "ALK", drop_censored: bool = True,
         verify: bool = True) -> pd.DataFrame:
    """The dataset, with the columns every analysis needs added.

    drop_censored removes right-censored records (`value_is_upper_bound`). Keep them only if
    your model handles bounds: treating a bound as a point estimate understates resistance,
    dropping it overstates sensitivity at the top of the range.
    """
    if verify:
        got = checksum(path)
        if got != PUBLISHED_MD5:
            raise ValueError(
                f"{path} does not match the published file.\n"
                f"  expected {PUBLISHED_MD5}\n  found    {got}\n"
                "Re-download from doi:10.5281/zenodo.22914209, or pass verify=False if you "
                "intend to work with a modified copy.")

    d = pd.read_csv(path)
    d["censored"] = d["value_is_upper_bound"].astype(str).str.lower().isin(
        ["true", "1", "yes"])
    if gene:
        d = d[d["gene"] == gene]
    if drop_censored:
        d = d[~d["censored"]]
    d = d[d["value_nM"].notna() & (d["value_nM"] > 0)].copy()

    for c in CONTEXT:
        d[c] = d[c].fillna("na").astype(str)
    d["variant"] = d["protein_variant"].astype(str).str.strip()
    d["is_wt"] = d["variant"].str.upper() == "WT"
    d["parts"] = d["variant"].str.split("+").apply(lambda ps: [p.strip() for p in ps])
    d["n_mut"] = d["parts"].str.len()
    d["is_single"] = d["variant"].str.match(SINGLE).fillna(False) & ~d["is_wt"]
    d["is_compound"] = d["n_mut"] > 1
    d["generation"] = d["drug_canonical"].map(GENERATION)
    return d


def harmonized(d: pd.DataFrame) -> pd.DataFrame:
    """The Ba/F3 EML4-ALK v1 cellular IC50 subset used for cross-source statistics."""
    m = np.ones(len(d), bool)
    for col, val in HARMONIZED.items():
        m &= (d[col] == val).to_numpy()
    return d[m].copy()


def fold_change(d: pd.DataFrame) -> pd.DataFrame:
    """Add `wt_nM` and `lfc` = log10(IC50 / IC50 of the wild type of the SAME context).

    Rows in a context with no wild-type measurement are dropped, as are the wild-type rows
    themselves. This is the only correct way to compute fold change from this dataset.
    """
    wt = (d[d["is_wt"]].groupby(CONTEXT)["value_nM"].median().rename("wt_nM"))
    out = d.merge(wt, on=CONTEXT, how="left")
    out = out[out["wt_nM"].notna() & (out["wt_nM"] > 0) & ~out["is_wt"]].copy()
    out["lfc"] = np.log10(out["value_nM"] / out["wt_nM"])
    return out


def compound_table(d: pd.DataFrame) -> pd.DataFrame:
    """Compound mutations whose every component single was measured in the same context.

    Returns one row per (context, compound) with the observed lfc, the additive expectation
    from its components, and the deviation epsilon = observed - expected.
    """
    lf = fold_change(d)
    level = lf.groupby(CONTEXT + ["variant"])["lfc"].median()
    known = {(k[:-1], k[-1]): v for k, v in level.items()}

    rows = []
    for _, r in lf[lf["is_compound"]].drop_duplicates(subset=CONTEXT + ["variant"]).iterrows():
        ctx = tuple(r[c] for c in CONTEXT)
        parts = r["parts"]
        vals = [known.get((ctx, p)) for p in parts]
        if any(v is None for v in vals):
            continue
        obs = known[(ctx, r["variant"])]
        exp = float(np.sum(vals))
        rows.append({"drug": r["drug_canonical"], "generation": r["generation"],
                     "paper": r["paper_id"], "assay": r["assay_class"],
                     "compound": r["variant"], "n_mut": r["n_mut"],
                     "observed": obs, "expected": exp, "epsilon": obs - exp,
                     "components": "|".join(parts),
                     "component_lfc": "|".join(f"{v:+.2f}" for v in vals)})
    return pd.DataFrame(rows)


def cross_lab_pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Every pair of laboratories that measured the same variant against the same drug."""
    lev = (d.groupby(["variant", "drug_canonical", "paper_id"])["value_nM"]
             .median().reset_index())
    lev["log10_nM"] = np.log10(lev["value_nM"])
    p = lev.merge(lev, on=["variant", "drug_canonical"], suffixes=("_i", "_j"))
    p = p[p["paper_id_i"] < p["paper_id_j"]].copy()
    p["difference"] = p["log10_nM_i"] - p["log10_nM_j"]
    return p


def lab_offsets(pairs: pd.DataFrame) -> pd.DataFrame:
    """Each laboratory's median signed offset in log10 IC50 against all the others."""
    rows = []
    for lab in sorted(set(pairs["paper_id_i"]) | set(pairs["paper_id_j"])):
        a = pairs.loc[pairs["paper_id_i"] == lab, "difference"]
        b = -pairs.loc[pairs["paper_id_j"] == lab, "difference"]
        off = pd.concat([a, b])
        if len(off):
            rows.append({"paper_id": lab, "n_pairs": len(off), "median": off.median(),
                         "q1": np.percentile(off, 25), "q3": np.percentile(off, 75)})
    return pd.DataFrame(rows).sort_values("median").reset_index(drop=True)


def bootstrap_ci(x, statistic=np.mean, n: int = 4000, seed: int = 0):
    """Percentile bootstrap confidence interval, used for every interval in the paper."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan, np.nan
    b = [statistic(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def auc(score, positive) -> float:
    """P(score higher for a positive than a negative), via the rank-sum identity.

    Used instead of a correlation wherever class prevalence differs between groups, since AUC
    is insensitive to how many positives there are.
    """
    from scipy import stats
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
