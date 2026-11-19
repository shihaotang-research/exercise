# Exercise Multiomics | Analysis code

Analysis and visualization scripts for exercise-training proteomics, metabolomics, molecular networks, and UK Biobank incident disease analyses.

## Contents

- `00_preprocessing` — protein and metabolite preprocessing
- `01_response` — mixed-effects models and response classification
- `02_enrichment` — protein GO and metabolite class enrichment
- `03_network` — resting-state and acute exercise-response networks and module–phenotype analysis
- `04_UKB` — incident disease associations and LASSO-Cox prediction
- `05_figures` — Figures 2–6 and Supplementary Figures S1–S3

## Usage

Run each script with its required input files. For scripts that support it, set `MULTIOMICS_DATA_DIR` (and `MULTIOMICS_UKB_DIR`) to the data directory. Python and R dependencies are imported in the respective scripts.

The archive contains **no supplementary-table-only scripts** and no participant-level or UK Biobank raw data. Several figure scripts read precomputed analysis results. Preserve the original cohort split and input definitions when reproducing the analyses.
