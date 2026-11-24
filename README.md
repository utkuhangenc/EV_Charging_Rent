# EV charging access, neighborhood characteristics, and rent burden (2017–2024)

Code and selected results for a study of public electric-vehicle (EV) charging access and neighborhood housing conditions in the **Washington–Arlington–Alexandria (DMV)** and **Philadelphia–Camden–Wilmington** metropolitan areas, 2017–2024. The analyses combine harmonized American Community Survey (ACS) block-group indicators, U.S. Department of Energy Alternative Fuels Data Center (AFDC) station histories, and built-environment measures. They include annual cross-sectional models, two-way fixed-effects (TWFE) panel models, lead–lag specifications, and sensitivity analyses.

**Interpretation:** Results describe associations, not established causal effects. Changes in charging deployment and housing conditions may affect each other or reflect other neighborhood changes. Annual ACS five-year estimates overlap, limiting precise temporal interpretation.

## Data and code availability

- **Code and verified manuscript outputs:** this repository (`code/` and `results/`).
- **Analysis-ready data:** [Zenodo dataset concept DOI](https://doi.org/10.5281/zenodo.21346452). **Release status:** the 2017–2024 version is being prepared. Until that new version is published, this DOI may resolve to the earlier **2017–2023** dataset. Select and use the version labeled **2017–2024** when available.
- **Historical workflow:** `archive/2017_2023/` preserves the older R Markdown analysis and documentation. It is not the current 2017–2024 workflow.

The current data deposit contains **eight derived datasets** used across the analyses, not every raw source download or geographic input. See [data/README.md](data/README.md) for the data inventory and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for what can be run directly and what requires additional geographic data.

## Repository layout

```text
code/                  Python analysis and figure-preparation scripts (2017–2024)
data/                  Data instructions; downloaded research data are not committed here
results/
  figures/main/        Main Figures 1–4 (PDF, PNG)
  figures/supplement/  Supplementary Figures SI.1–SI.4
  figure_data/         Selected underlying figure inputs
  main_tables/         Tables 1–3 and supporting estimates
  supplemental_tables/ Tables SI.1–SI.13 and supporting exports
  model_outputs/       Selected cross-sectional, TWFE and lead–lag results
archive/2017_2023/     Prior 2017–2023 R workflow
requirements.txt       Python dependencies (not a frozen environment)
verify_release.py      Static Python syntax/path/credential screening
```

`results/README.md` explains differences between some legacy filenames and their final manuscript table numbers. `results/RESULTS_MANIFEST.csv` records SHA-256 hashes for the curated results.

## Quick start

The analysis code was prepared using Python 3.12 on a research computing system. Create a Python environment and install dependencies:

```bash
python -m pip install -r requirements.txt
python verify_release.py
```

**Download the 2017–2024 Zenodo version first.** Extract the ZIP into a **separate directory** (not directly into the repository root, because the Zenodo archive also contains its own `README.md`). Within the extracted directory, the deposited inputs are under `data/`. Point `EVCS_DATA_ROOT` at that `data/` directory, for example:

```bash
# Linux / macOS
export EVCS_DATA_ROOT="/path/to/zenodo_extracted/data"
```

```powershell
# Windows PowerShell
$env:EVCS_DATA_ROOT = "C:\path\to\zenodo_extracted\data"
```

`EVCS_DATA_ROOT` must name the **data directory**, not its parent. The scripts also accept optional `EVCS_SHAPE_ROOT`, `EVCS_BG_SHP`, and `EVCS_CBSA_SHP` environment variables for geographic files that are not included in Zenodo.

After installation, these two synthetic self-tests do not require the study data:

```bash
python code/two_period_twfe_evcs_2017_2024.py --self-test
python code/evcs_tract_sensitivity_2017_2024.py --stage self-test
```

To inspect the downloaded analysis-data availability (without rebuilding geographic exposures):

```bash
python code/evcs_revision_remaining_analyses_v2.py --stage audit
```

See the [reproducibility guide](docs/REPRODUCIBILITY.md) before running the model scripts. Depending on the analysis, additional geographic source files and processing steps are required. Executing the self-tests or passing the static scan does **not** demonstrate full end-to-end reproduction of the paper.

## Manuscript tables and figures

| Exhibit | Release location |
|---|---|
| Figures 1–4 | `results/figures/main/` |
| Supplementary Figures SI.1–SI.4 | `results/figures/supplement/` |
| Table 1: random forest variable importance | `results/main_tables/table1/` |
| Table 2: charging categories and rent-to-income ratio | `results/main_tables/table2/` |
| Table 3: robustness-grid significance summaries | `results/main_tables/table3/` |
| Tables SI.1–SI.13 | `results/supplemental_tables/si01/` through `si13/` |
| Detailed coefficients and selected diagnostics | `results/model_outputs/` |

**Table 2 correction:** The pooled, 24-hour public DC-fast **proximity** coefficient is **1.355** when scaled by 10⁴. An earlier manuscript draft displayed **1.335** by transcription error; the source model output supports 1.355. See `results/README.md`.

## Important reproducibility limitations

- The deposited files support analysis at several stages; they do **not** include all raw Census downloads, Census geometry, the complete EPA Smart Location Database geodatabase, or every historical intermediate.
- The earlier 2017–2024 cross-sectional extension retains a historical **EPSG:3857** buffer convention for comparability with previous models; the final TWFE geographic exposure construction uses **EPSG:5070**. Do not silently treat the two procedures as identical distance measures.
- The Python replication of earlier random forest models is a methodological approximation to the original R workflow; stochastic algorithms and package implementations can produce numerical differences.
- Targeted data/model checks and figure/table comparisons were completed for the 2026 release; a clean-machine reproduction of every GIS processing step and all 810 robustness models has **not** been independently verified.
- `requirements.txt` lists dependencies but does not pin an exact computational environment; record versions when rerunning analyses.

## Sources and attribution

Principal sources include U.S. Census Bureau ACS estimates and geographic relationship files; U.S. DOE Alternative Fuels Data Center (AFDC) station records; and U.S. EPA Smart Location Database. Their respective data-use and attribution conditions continue to apply. The Zenodo archive includes a data dictionary and source/reproducibility notes. This repository and its deposit do not imply endorsement by the source agencies.
