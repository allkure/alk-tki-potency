# ALK-TKI potency dataset — analysis code

Code accompanying the data descriptor *"A harmonized dataset of ALK tyrosine kinase inhibitor
potency against kinase-domain resistance variants, with variant-level clinical linkage."*

The dataset itself is archived at Zenodo under
**[doi:10.5281/zenodo.22914209](https://doi.org/10.5281/zenodo.22914209)** (CC BY 4.0). A copy of
the potency file is included here under `data/` for convenience; Zenodo is the citable source,
and the code refuses to run against a copy whose MD5 does not match the published one.

Every number and figure in the descriptor is reproduced by the three entry points below. Nothing
reads an intermediate file.

## Quick start

```bash
git clone https://github.com/allkure/alk-tki-potency.git
cd alk-tki-potency
pip install -r requirements.txt

python3 src/validation.py     # technical validation: every reported statistic
python3 src/additivity.py     # compound-mutation additivity and prediction
jupyter lab figures.ipynb     # all four figures -> figures/
```

## What is here

| Path | Purpose |
| --- | --- |
| `src/alkdata.py` | Loading, checksum verification, context-aware fold change, compound decomposition. Everything else imports this. |
| `src/validation.py` | Reproduces the composition counts, cross-source concordance, per-laboratory offsets and biological sanity checks. |
| `src/additivity.py` | The compound-mutation model: additivity, the compression diagnostic, leave-one-drug-out calibration, and the prioritisation test. |
| `figures.ipynb` | Figures 1–4, written to `figures/` as vector PDF and 600 dpi PNG. |
| `data/` | The potency CSV and its curation protocol. |

## Two conventions the code enforces

**Fold change is computed within context.** A measurement is comparable only with another made in
the same paper, assay class, cell line, expression construct and drug. `alkdata.fold_change()`
computes every fold change against the wild type of a measurement's own context and drops rows
whose context has no wild type. Pooling wild-type values across papers discards the design
feature that makes the dataset internally valid — the seven source laboratories differ
systematically by up to 0.34 log<sub>10</sub>, a 4.6-fold spread between the extremes.

**Gilteritinib is a comparator, not a therapy.** It is a FLT3 inhibitor included in the dataset
for off-target context. It carries no inhibitor generation in `alkdata.GENERATION`, is excluded
from every generation-based analysis, and appears in the potency matrix marked with an asterisk.
Including it would move the fitted additivity slope from 0.652 to 0.647.

## Censored measurements

Fifteen ALK records (and nine off-target NTRK records, 24 in all) are right-censored: the drug did
not reach IC<sub>50</sub> at the highest concentration tested, so `value_nM` holds a bound and
`value_is_upper_bound` is set. `alkdata.load()` drops them by default. Keep them only if your
model handles bounds — treating a bound as a point estimate understates resistance, dropping it
overstates sensitivity at the top of the range.

## Reproducing the main results

```
cross-laboratory agreement   409 pairs, Pearson r = 0.80, Spearman rho = 0.81
                             SD 0.50 log10 (3.2-fold), 95% within ±1.03 log10
laboratory offsets           -0.33 (mizuta_2021) to +0.34 (lin_2024) log10
compound additivity          150 observations, 20 compounds, 9 ALK-directed drugs
                             observed = 0.290 + 0.652 x expected, r = 0.866
leave-one-drug-out error     0.50 log10 naive -> 0.37 log10 calibrated (3.2x -> 2.4x)
prioritisation AUC           0.61-0.77 for generations 1-3; at chance for generation 4
```

## Requirements

Python 3.9+, with NumPy, pandas, SciPy and Matplotlib. Jupyter only for the notebook. No
compiled dependencies, no GPU, and the whole analysis runs in under a minute on a laptop.

## Licence

Code is MIT (see `LICENSE`). The dataset is CC BY 4.0, as deposited at Zenodo — if you use it,
cite the Zenodo record and the data descriptor, not this repository alone.

## Contact

ALLKURE ETS — <info@allkure.org>
