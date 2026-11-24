# Reproduced outputs: EV charging access and rental affordability, 2017–2024

These files correspond to the revised manuscript's main and supplementary results.
The main four figures were visually compared against the 2026-11-24 clean manuscript.

**Table 2 correction:** The pooled, public 24-hour DC-fast, *proximity* coefficient
is **1.355** (scaled by 10^4), rather than 1.335 as initially typed in the
clean manuscript. The model-result CSVs were not altered.

**Table SI.5:** `supplemental_tables/si05/Table_SI_5_verified_from_models.csv`
was taken from the Gautschi validation output. All nine charging-category rows,
moderate/severe outcomes, coefficient estimates and 95% confidence intervals
were compared against saved model outputs.

**Table SI.13:** `supplemental_tables/si13/Table_SI_13_pooled_descriptive_statistics.csv`
was taken from the Gautschi validation output; it has 18 variables and the
reported 2017–2024 descriptive statistics.

Legacy output names and their final table numbers:
- `Table_SI_1_Modified_Poisson_2017_2024.csv` = **SI.3**.
- `Table_SI_2_Random_Forest_2017_2024.csv` = **SI.4**.
- `SI3_FE_only_2017_2024.csv` = **SI.6**.
- `SI4A_access_patterns_2017_2024.csv` and `SI4B_first_gain_by_year_2017_2024.csv` = **SI.7**.
- Files under `si13/` with older `Table_SI_11_dmv` or `_philadelphia` names
  provide metro-specific descriptive summaries, NOT the published pooled SI.13.

**Cautions:** Table SI.8 uses 2017–2024, 2017–2023 and 2018–2024 nonoverlapping
ACS comparisons. Some additional geographic/model-input files remain outside this
results-only package. Full end-to-end reproduction, including the cross-sectional
EPSG:3857 vs panel EPSG:5070 distance-construction distinction, should be
assessed separately before describing the entire repository as fully reproducible.

The `RESULTS_MANIFEST.csv` records the SHA-256 checksum for every file in this
results directory. The original staged numerical outputs and figures were not
edited by this packaging script; only verified SI.5/SI.13 additions and the
removal of an obsolete duplicate pooled SI.13 export are applied.
