"""Technical validation: reproduces every number in the data descriptor from the CSV.

Nothing here is read from the curation notes or the manuscript. If a number in the paper is
wrong, this script disagrees with it.

    python3 -I src/validation.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works under python3 -I too

import alkdata as A

SINGLE_KEY = ["L1196M", "G1202R", "I1171N", "F1174L", "G1269A", "L1198F"]
DRUG_ORDER = ["crizotinib", "ceritinib", "alectinib", "brigatinib", "ensartinib",
              "entrectinib", "gilteritinib", "lorlatinib", "repotrectinib",
              "zotizalkib", "neladalkib"]


def head(s):
    print(f"\n{s}\n{'=' * len(s)}")


def main():
    raw = pd.read_csv(A.CSV)
    print(f"file      {A.CSV}")
    print(f"checksum  {A.checksum()}  (published: {A.PUBLISHED_MD5})")

    head("1. Composition")
    cen = raw["value_is_upper_bound"].astype(str).str.lower().isin(["true", "1", "yes"])
    alk = raw[raw["gene"] == "ALK"]
    v = alk["protein_variant"].astype(str).str.strip()
    v = sorted(set(v[v.str.upper() != "WT"]))
    comp = [x for x in v if "+" in x]
    print(f"  records                {len(raw):>6,}")
    print(f"  source publications    {raw['paper_id'].nunique():>6}")
    print(f"  compounds              {raw['drug_canonical'].nunique():>6}")
    print(f"  unique ALK variants    {len(v):>6}")
    print(f"    single missense      {sum(bool(A.SINGLE.match(x)) for x in v):>6}")
    print(f"    in-frame indels      "
          f"{sum(('del' in x or 'ins' in x) and '+' not in x for x in v):>6}")
    print(f"    double compounds     {sum(x.count('+') == 1 for x in comp):>6}")
    print(f"    triple compounds     {sum(x.count('+') == 2 for x in comp):>6}")
    print(f"  cellular records       {(raw['assay_class'] == 'cellular').sum():>6}")
    print(f"    cellular IC50        {(raw['value_type'] == 'Cellular_IC50').sum():>6}")
    print(f"    cellular Emax        "
          f"{raw['value_type'].fillna('').str.contains('max', case=False).sum():>6}")
    print(f"  biochemical IC50       "
          f"{((raw['assay_class'] == 'biochemical') & (raw['value_type'] == 'IC50')).sum():>6}")
    print(f"  off-target records     "
          f"{raw['gene'].isin(['ROS1', 'NTRK1', 'NTRK2', 'NTRK3']).sum():>6}")
    print(f"  right-censored         {cen.sum():>6}")
    for g, n in raw.loc[cen, "gene"].value_counts().items():
        print(f"    {g:<20}{n:>6}")
    print("\n  records per source")
    for p, n in raw["paper_id"].value_counts().items():
        print(f"    {p:<16}{n:>6,}")

    head("2. Internal consistency")
    bad = 0
    for unit, factor in (("nM", 1.0), ("uM", 1e3), ("µM", 1e3), ("mM", 1e6),
                         ("M", 1e9), ("pM", 1e-3)):
        s = raw[(raw["reported_unit"] == unit) & raw["reported_value"].notna()
                & raw["value_nM"].notna()]
        if not len(s):
            continue
        err = ((s["reported_value"] * factor - s["value_nM"]).abs()
               / s["value_nM"].clip(lower=1e-9))
        n = int((err > 0.01).sum())
        bad += n
        print(f"  unit {unit:<3} n = {len(s):<6} conversions off by >1%: {n}")
    print(f"  -> {'PASS' if not bad else 'FAIL'}")

    d = A.load()
    lf = A.fold_change(d)
    k = lf[lf["fold_change_vs_wt"].notna()].copy()
    k["recomputed"] = k["value_nM"] / k["wt_nM"]
    if len(k) > 2:
        r = stats.spearmanr(k["fold_change_vs_wt"], k["recomputed"])
        ratio = np.log10(k["fold_change_vs_wt"] / k["recomputed"])
        print(f"\n  stored fold_change_vs_wt vs recomputed: n = {len(k)}, "
              f"rho = {r.statistic:.4f}, within 10%: {(abs(ratio) < 0.041).mean() * 100:.1f}%")

    head("3. Replication and cross-source concordance")
    h = A.harmonized(d)
    print(f"  harmonized subset (Ba/F3, EML4-ALK v1, cellular IC50, uncensored): {len(h)}")
    combo = h.groupby(["variant", "drug_canonical"])["paper_id"].nunique()
    print(f"  variant x drug combinations profiled   {len(combo):>5}")
    for k_ in (2, 3, 4):
        print(f"    measured in >= {k_} laboratories      {(combo >= k_).sum():>5}")

    p = A.cross_lab_pairs(h)
    pr = stats.pearsonr(p["log10_nM_i"], p["log10_nM_j"])
    sp = stats.spearmanr(p["log10_nM_i"], p["log10_nM_j"])
    diff = p["difference"].to_numpy()
    print(f"\n  matched cross-laboratory pairs         {len(p):>5}")
    print(f"    Pearson r                            {pr.statistic:.3f}")
    print(f"    Spearman rho                         {sp.statistic:.3f}")
    print(f"    SD of difference                     {diff.std():.3f} log10 "
          f"({10 ** diff.std():.1f}-fold)")
    print(f"    95% of pairs within                  ±{np.percentile(abs(diff), 95):.2f} log10")
    print("\n  systematic per-laboratory offset (log10 IC50)")
    off = A.lab_offsets(p)
    print(f"    {'laboratory':<16}{'n':>6}{'median':>9}{'IQR':>18}")
    for _, r_ in off.iterrows():
        print(f"    {r_['paper_id']:<16}{r_['n_pairs']:>6}{r_['median']:>+9.3f}"
              f"{f'[{r_.q1:+.2f}, {r_.q3:+.2f}]':>18}")
    print(f"    spread between extremes: {10 ** (off['median'].max() - off['median'].min()):.1f}-fold")

    head("4. Biological validation")
    cell = lf[lf["value_type"] == "Cellular_IC50"]
    piv = (cell[cell["is_single"]]
           .pivot_table(index="variant", columns="drug_canonical", values="value_nM",
                        aggfunc="median"))
    piv = piv.reindex(columns=[c for c in DRUG_ORDER if c in piv.columns])
    print("  Absolute median IC50 (nM), censored records excluded. Gilteritinib is marked *")
    print("  as an off-target comparator and excluded from the 'best' column, which asks")
    print("  whether the most potent ALK-DIRECTED drug is lorlatinib or a fourth-generation one.")
    print(f"    {'variant':<9}"
          + "".join(f"{(c[:5] + '*' if not A.GENERATION.get(c) else c[:6]):>9}"
                   for c in piv.columns) + "   best ALK drug")
    for var in SINGLE_KEY:
        if var not in piv.index:
            continue
        row = piv.loc[var]
        cells = "".join(f"{row[c]:>9.1f}" if pd.notna(row[c]) else f"{'--':>9}"
                        for c in piv.columns)
        alk_only = row[[c for c in piv.columns if A.GENERATION.get(c)]]
        print(f"    {var:<9}{cells}   {alk_only.idxmin()}")

    def get(var, drug):
        return piv.loc[var, drug] if var in piv.index and drug in piv.columns else float("nan")

    print(f"\n  L1198F resensitisation: crizotinib {get('L1198F', 'crizotinib'):.1f} nM vs "
          f"ceritinib {get('L1198F', 'ceritinib'):.0f} nM, "
          f"alectinib {get('L1198F', 'alectinib'):.0f} nM")
    print(f"  I1171N separates 4th gen: zotizalkib {get('I1171N', 'zotizalkib'):.0f} nM vs "
          f"neladalkib {get('I1171N', 'neladalkib'):.0f} nM")

    head("5. Field completeness")
    for c in raw.columns:
        frac = raw[c].notna().mean()
        if frac < 1.0:
            print(f"  {c:<38}{frac * 100:>6.1f}%")


if __name__ == "__main__":
    sys.exit(main())
