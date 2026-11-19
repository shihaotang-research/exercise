# -*- coding: utf-8 -*-

"""Assign baseline-only, dual-response, and dynamic-only response classes using LMM results."""

import os
from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
BASELINE_FILE = DATA_DIR / "baseline_lmm_results.tsv"
TREND_FILE = DATA_DIR / "interaction_lmm_results.tsv"
FC_FILE = DATA_DIR / "protein_fold_changes.tsv"

FDR_CUTOFF = 0.05
FC_UP = 2.0
FC_DOWN = 0.5

OUTPUT_FILES = {
    "protein_baseline_only": DATA_DIR / "protein_baseline_only.tsv",
    "protein_both": DATA_DIR / "protein_dual_response.tsv",
    "protein_trend_only": DATA_DIR / "protein_dynamic_only.tsv",
    "metabolite_baseline_only": DATA_DIR / "metabolite_baseline_only.tsv",
    "metabolite_both": DATA_DIR / "metabolite_dual_response.tsv",
    "metabolite_trend_only": DATA_DIR / "metabolite_dynamic_only.tsv",
}

def norm_id(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().replace("\\", "/").rsplit("/", 1)[-1]
    return re.sub(r"\.txt$", "", x, flags=re.I).strip()

def save_list(values, path, col):
    pd.DataFrame({col: sorted(values)}).to_csv(
        path, sep="\t", index=False
    )

for f in [BASELINE_FILE, TREND_FILE, FC_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"File not found: {f.resolve()}")

baseline = pd.read_csv(BASELINE_FILE, sep="\t", dtype=str)

if not {"Molecule", "FDR"}.issubset(baseline.columns):
    raise ValueError(
        f"{BASELINE_FILE} must contain Molecule and FDR columns, "
        f"Current columns: {list(baseline.columns)}"
    )

baseline["Molecule"] = baseline["Molecule"].map(norm_id)
baseline["FDR"] = pd.to_numeric(baseline["FDR"], errors="coerce")

baseline["Omics"] = baseline["Molecule"].apply(
    lambda x: "Metabolite" if x.lower().startswith("met") else "Protein"
)

baseline_met = set(
    baseline.loc[
        (baseline["Omics"] == "Metabolite") &
        (baseline["FDR"] < FDR_CUTOFF),
        "Molecule"
    ]
)

fc = pd.read_csv(FC_FILE, sep="\t", dtype=str)

if not {"protein", "FCAD"}.issubset(fc.columns):
    raise ValueError(
        f"{FC_FILE} must contain protein and FCAD columns, "
        f"Current columns: {list(fc.columns)}"
    )

fc["protein"] = fc["protein"].map(norm_id)
fc["FCAD"] = pd.to_numeric(fc["FCAD"], errors="coerce")

fc = (
    fc[["protein", "FCAD"]]
    .dropna(subset=["FCAD"])
    .drop_duplicates("protein", keep="first")
)

baseline_pro_df = baseline.loc[
    baseline["Omics"] == "Protein",
    ["Molecule", "FDR"]
].merge(
    fc,
    left_on="Molecule",
    right_on="protein",
    how="left",
    validate="one_to_one"
)

baseline_pro = set(
    baseline_pro_df.loc[
        (baseline_pro_df["FDR"] < FDR_CUTOFF) &
        (
            (baseline_pro_df["FCAD"] > FC_UP) |
            (baseline_pro_df["FCAD"] < FC_DOWN)
        ),
        "Molecule"
    ]
)

baseline_pro.discard("")
baseline_met.discard("")

trend = pd.read_csv(TREND_FILE, sep="\t", dtype=str)

if not {"File", "Omics", "FDR_all_3181"}.issubset(trend.columns):
    raise ValueError(
        f"{TREND_FILE} must contain File, Omics, FDR_all_3181 columns, "
        f"Current columns: {list(trend.columns)}"
    )

trend["Molecule"] = trend["File"].map(norm_id)
trend["Omics"] = trend["Omics"].astype(str).str.strip().str.lower()
trend["FDR_all_3181"] = pd.to_numeric(
    trend["FDR_all_3181"], errors="coerce"
)

trend_sig = trend.loc[
    trend["FDR_all_3181"] < FDR_CUTOFF
].copy()

trend_pro = set(
    trend_sig.loc[
        trend_sig["Omics"] == "protein",
        "Molecule"
    ]
)

trend_met = set(
    trend_sig.loc[
        trend_sig["Omics"] == "metabolite",
        "Molecule"
    ]
)

trend_pro.discard("")
trend_met.discard("")

protein_baseline_only = baseline_pro - trend_pro
protein_both = baseline_pro & trend_pro
protein_trend_only = trend_pro - baseline_pro

metabolite_baseline_only = baseline_met - trend_met
metabolite_both = baseline_met & trend_met
metabolite_trend_only = trend_met - baseline_met

baseline_ids = set(baseline["Molecule"]) - {""}
trend_ids = set(trend["Molecule"]) - {""}

print("=" * 68)
print("3181molecules input")
print("=" * 68)
print(f"Baseline rows : {len(baseline)}")
print(f"Trend rows    : {len(trend)}")
print(f"ID overlap    : {len(baseline_ids & trend_ids)}")

print("\n[Protein]")
print(f"Baseline FDR<0.05                    : "
      f"{sum((baseline['Omics']=='Protein') & (baseline['FDR']<FDR_CUTOFF))}")
print(f"Baseline FDR<0.05 + FC criterion    : {len(baseline_pro)}")
print(f"Trend FDR<0.05                      : {len(trend_pro)}")
print(f"Baseline-only                       : {len(protein_baseline_only)}")
print(f"Dual-response                       : {len(protein_both)}")
print(f"Trend-only                          : {len(protein_trend_only)}")

print("\n[Metabolite]")
print(f"Baseline FDR<0.05                   : {len(baseline_met)}")
print(f"Trend FDR<0.05                      : {len(trend_met)}")
print(f"Baseline-only                       : {len(metabolite_baseline_only)}")
print(f"Dual-response                       : {len(metabolite_both)}")
print(f"Trend-only                          : {len(metabolite_trend_only)}")

save_list(
    protein_baseline_only,
    OUTPUT_FILES["protein_baseline_only"],
    "protein"
)
save_list(
    protein_both,
    OUTPUT_FILES["protein_both"],
    "protein"
)
save_list(
    protein_trend_only,
    OUTPUT_FILES["protein_trend_only"],
    "protein"
)

save_list(
    metabolite_baseline_only,
    OUTPUT_FILES["metabolite_baseline_only"],
    "Metabolite"
)
save_list(
    metabolite_both,
    OUTPUT_FILES["metabolite_both"],
    "Metabolite"
)
save_list(
    metabolite_trend_only,
    OUTPUT_FILES["metabolite_trend_only"],
    "Metabolite"
)

print("\n[DONE] input file: ")
for k, v in OUTPUT_FILES.items():
    print(f"{k:<28} -> {v}")
print("=" * 68)
