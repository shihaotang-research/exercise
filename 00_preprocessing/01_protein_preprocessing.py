#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protein preprocessing and quality control for longitudinal exercise multiomics."""
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ======================== USER SETTINGS ========================
DATA_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
INPUT_FILE = DATA_DIR / "protein_raw.tsv"
REFERENCE_LIST = DATA_DIR / "protein_ids.txt"  # optional cross-check; no unannounced intersection
PHENOTYPE_FILE = DATA_DIR / "protein_matrix.tsv"  # optional
OUTPUT_DIR = DATA_DIR / "protein_preprocessed"
FEATURE_COL = "Accession"
TIMEPOINTS = ("A", "B1", "B2", "B3", "D", "F1", "F2", "F3")
MAX_MISSING = 0.50
TREAT_ZERO_AS_MISSING = False  # False matches original protein NA-based screening/imputation
WRITE_LOG2 = False  # Set True ONLY if protein_raw.tsv contains positive, non-log2 intensities
EXPECTED_PROTEINS = 2039  # audit check; pipeline does NOT force a fixed number

SAMPLE_RE = re.compile(r"^(ae\d+)(B[123]|F[123]|A|D)$", re.IGNORECASE)
TIME_ORDER = {t: i for i, t in enumerate(TIMEPOINTS)}

def sample_info(col: str):
    match = SAMPLE_RE.fullmatch(str(col))
    return (match.group(1).lower(), match.group(2).upper()) if match else None

def numeric_matrix(df: pd.DataFrame, cols):
    x = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    bad = df.loc[:, cols].notna() & x.isna()
    if bad.to_numpy().any():
        c = np.argwhere(bad.to_numpy())[0]
        raise ValueError(f"Non-numeric value in column {cols[c[1]]}, row {c[0] + 2}")
    return x.astype(float)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
    if FEATURE_COL not in raw:
        raise ValueError(f"Expected '{FEATURE_COL}' column; found: {raw.columns[:8].tolist()}")
    if raw[FEATURE_COL].isna().any() or raw[FEATURE_COL].duplicated().any():
        raise ValueError("Accession must be nonmissing and unique per protein")

    sample_cols = [c for c in raw.columns if sample_info(c) is not None]
    groups = {t: [c for c in sample_cols if sample_info(c)[1] == t] for t in TIMEPOINTS}
    if any(not cols for cols in groups.values()):
        raise ValueError("Missing sample columns for: " + str([t for t, v in groups.items() if not v]))
    print("[Raw protein rows]", len(raw), "[eight-timepoint sample columns]", len(sample_cols))
    values = numeric_matrix(raw, sample_cols)
    if (values < 0).to_numpy().any():
        raise ValueError("Negative intensities found. Could be log-transformed values; "
                         "do not apply the 'global min / 2' rule to logged data.")

    # Calculate missingness over the input sample matrix as supplied.
    screen = values.isna() | ((values == 0) if TREAT_ZERO_AS_MISSING else False)
    missing_rates = {t: screen[cols].mean(axis=1) for t, cols in groups.items()}
    rates = pd.DataFrame(missing_rates)
    keep_all = rates.le(MAX_MISSING).all(axis=1)
    keep_one = rates.le(MAX_MISSING).any(axis=1)

    audit = pd.DataFrame({FEATURE_COL: raw[FEATURE_COL]})
    for t in TIMEPOINTS:
        audit[f"missing_rate_{t}"] = rates[t]
    audit["at_least_one_timepoint_pass"] = keep_one
    audit["pass_all_eight"] = keep_all
    audit.to_csv(OUTPUT_DIR / "protein_filter_audit.tsv", sep="\t", index=False)
    selected = raw.loc[keep_all, FEATURE_COL].astype(str).tolist()
    (OUTPUT_DIR / "protein_selected_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")

    # Compare with original list only; prevent silent selection changes.
    list_check = "not provided"
    if REFERENCE_LIST.exists():
        reference = {s.strip() for s in REFERENCE_LIST.read_text(encoding="utf-8").splitlines() if s.strip()}
        calculated = set(selected)
        list_check = f"matched={reference == calculated}; only_in_list={len(reference-calculated)}; only_calculated={len(calculated-reference)}"
        if reference != calculated:
            print("[WARNING] protein_ids.txt differs from current 8-timepoint screening.")

    selected_values = values.loc[keep_all].copy()
    selected_values.index = selected
    selected_sample_cols = list(sample_cols)
    if not selected_sample_cols:
        raise ValueError("No biological sample columns available")
    selected_values = selected_values[selected_sample_cols]
    selected_values = selected_values.T
    selected_values.index.name = "SampleID"
    selected_values.index = selected_values.index.astype(str)
    selected_values.insert(0, "id", [sample_info(c)[0] for c in selected_values.index])
    selected_values.insert(1, "time", [sample_info(c)[1] for c in selected_values.index])
    selected_values = selected_values.sort_values(
        ["id", "time"], key=lambda s: s.map(TIME_ORDER) if s.name == "time" else s
    ).reset_index(drop=True)

    feats = list(selected)
    if TREAT_ZERO_AS_MISSING:
        selected_values[feats] = selected_values[feats].replace(0, np.nan)
    vals = selected_values[feats].to_numpy(dtype=float)
    valid = vals[np.isfinite(vals) & (vals != 0)]
    if valid.size == 0 or valid.min() <= 0:
        raise ValueError("Cannot compute strictly positive global nonzero minimum")
    fill_value = float(valid.min() / 2)
    n_missing = int(np.isnan(vals).sum())
    n_zeros = int((vals == 0).sum())
    selected_values[feats] = selected_values[feats].fillna(fill_value)

    # Phenotype file may have several timepoints/rows per participant.
    # To preserve the user's historical workflow, keep the first row per id;
    # warn when alleged 'static' columns actually vary within participants.
    phenotype_added = []
    if PHENOTYPE_FILE.exists():
        info = pd.read_csv(PHENOTYPE_FILE, sep="\t", low_memory=False)
        wanted = [c for c in ("fat", "sex", "age", "bmi") if c in info.columns]
        if "id" in info and wanted:
            info["id"] = info["id"].astype(str).str.lower().str.strip()
            changing = [c for c in wanted if info.groupby("id")[c].nunique(dropna=True).gt(1).any()]
            if changing:
                print("[WARNING] These phenotype fields vary within subjects; using first row:", changing)
            static = info[["id", *wanted]].drop_duplicates("id")
            selected_values = selected_values.merge(static, on="id", how="left", validate="many_to_one")
            phenotype_added = wanted
        else:
            print("[WARNING] Phenotype file lacks 'id' or metadata columns; skipped merge")

    raw_output = OUTPUT_DIR / "protein_imputed_matrix.tsv"
    selected_values.to_csv(raw_output, sep="\t", index=False)

    if WRITE_LOG2:
        if (selected_values[feats].to_numpy(dtype=float) <= 0).any():
            raise ValueError("Cannot apply log2 to zero-valued data; set TREAT_ZERO_AS_MISSING=True only if biologically appropriate")
        logged = selected_values.copy()
        logged[feats] = np.log2(logged[feats].astype(float))
        logged.to_csv(OUTPUT_DIR / "protein_log2_candidate.tsv", sep="\t", index=False)

    note = (
        f"original_protein_count={len(raw)}\n"
        f"at_least_one_timepoint_pass={int(keep_one.sum())}\n"
        f"all_eight_pass={len(feats)}\n"
        f"expected_2039_match={len(feats) == EXPECTED_PROTEINS}\n"
        f"reference_list_check={list_check}\n"
        f"n_samples={len(selected_values)}\n"
        f"n_participants={selected_values['id'].nunique()}\n"
        f"missing_imputed={n_missing}\n"
        f"zeros_remaining={n_zeros if not TREAT_ZERO_AS_MISSING else 0}\n"
        f"imputation_half_min={fill_value:.10g}\n"
        f"phenotype_columns={','.join(phenotype_added)}\n"
        f"log2_written={WRITE_LOG2}\n"
        "protein_normalization=NOT PERFORMED IN THIS SCRIPT (verify upstream)\n"
        f"input={INPUT_FILE}\noutput={raw_output}\n"
    )
    (OUTPUT_DIR / "protein_processing_summary.tsv").write_text(note, encoding="utf-8")
    print(note)
    if len(feats) != EXPECTED_PROTEINS:
        print("[WARNING] Selected protein count differs from 2039. Review protein_filter_audit.tsv.")

if __name__ == "__main__":
    main()
