#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metabolite preprocessing, quality control, and filtering."""
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ========================= USER SETTINGS =========================
DATA_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
INPUT_FILE = DATA_DIR / "metabolite_raw.tsv"
OUTPUT_DIR = DATA_DIR / "metabolite_preprocessed"
IDENTIFICATION_RULE = "historical_score"  # OR "level_specific" (latest L1/L2 logic)
QC_CV_MAX = 30.0
MS2_SCORE_MIN = 2.4
MAX_MISSING = 0.50
# 'zero_only' matches your original R missingness screen;
# 'zero_or_na' is safer if actual NA values also appear.
MISSING_RATE_MODE = "zero_only"
SCREEN_TIMEPOINTS = ("A", "B1", "B2", "B3", "B4", "D", "F1", "F2", "F3", "F4")
FINAL_TIMEPOINTS = ("A", "B1", "B2", "B3", "D", "F1", "F2", "F3")
WRITE_LOG2 = False

FEATURE = "MS2_name"
META = ("MS2_name", "MS2_score", "level", "mzmed", "rtmed", "type")
IS_NAMES = {f"IS{i}" for i in range(1, 7)}
SAMPLE_RE = re.compile(r"^(ae\d+)(B[1-4]|F[1-4]|A|D|C)$", re.IGNORECASE)
ORDER = {t: i for i, t in enumerate(FINAL_TIMEPOINTS)}

def sample_info(s: str):
    match = SAMPLE_RE.fullmatch(str(s))
    return (match.group(1).lower(), match.group(2).upper()) if match else None

def read_and_check():
    df = pd.read_csv(INPUT_FILE, sep="\t", low_memory=False)
    missing_meta = sorted(set(META) - set(df.columns))
    if missing_meta:
        raise ValueError(f"Missing metadata columns: {missing_meta}")
    if df[FEATURE].isna().any() or df[FEATURE].duplicated().any():
        raise ValueError("MS2_name must have one unique, nonmissing row per feature")
    sample_cols = list(df.columns[6:])
    if set(sample_cols) & set(META):
        raise ValueError("Expected six metadata columns followed by sample columns")
    numbers = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    invalid = df[sample_cols].notna() & numbers.isna()
    if invalid.to_numpy().any():
        r, c = np.argwhere(invalid.to_numpy())[0]
        raise ValueError(f"Invalid number: row={r+2}, column={sample_cols[c]}")
    df[sample_cols] = numbers.astype(float)
    if (numbers < 0).to_numpy().any():
        raise ValueError("Negative data found; source might be log-scale. Check before half-min imputation")
    return df, sample_cols

def calculate_missing_rate(df: pd.DataFrame, cols):
    x = df[cols]
    if MISSING_RATE_MODE == "zero_only":
        return x.eq(0).sum(axis=1) / len(cols)
    if MISSING_RATE_MODE == "zero_or_na":
        return (x.eq(0) | x.isna()).sum(axis=1) / len(cols)
    raise ValueError("MISSING_RATE_MODE must be 'zero_only' or 'zero_or_na'")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw, sample_cols = read_and_check()
    all_groups = {t: [c for c in sample_cols if sample_info(c) and sample_info(c)[1] == t]
                  for t in SCREEN_TIMEPOINTS}
    missing_groups = [t for t, cols in all_groups.items() if not cols]
    if missing_groups:
        raise ValueError(f"Missing sample columns for timepoints: {missing_groups}")
    qc_cols = [c for c in sample_cols if str(c).upper().startswith("QC")]
    if len(qc_cols) < 2:
        raise ValueError("At least two QC columns needed for SD/mean CV calculation")
    if IDENTIFICATION_RULE not in {"historical_score", "level_specific"}:
        raise ValueError("IDENTIFICATION_RULE must be historical_score or level_specific")

    # QC CV on RAW intensities, matching original R code; SD uses sample ddof=1.
    qc = raw[qc_cols]
    mean = qc.mean(axis=1)
    sd = qc.std(axis=1, ddof=1)
    cv = sd / mean * 100.0
    cv = cv.where(mean.gt(0) & np.isfinite(cv), np.inf)
    level = raw["level"].astype(str).str.lower().str.strip()
    score = pd.to_numeric(raw["MS2_score"], errors="coerce")
    is_internal_standard = raw[FEATURE].astype(str).isin(IS_NAMES)
    cv_pass = cv.lt(QC_CV_MAX) & ~is_internal_standard
    score_pass = score.ge(MS2_SCORE_MIN)

    # Both selection logics are recorded for a transparent comparison.
    historical_pre = cv_pass & score_pass
    level_pre = cv_pass & (level.eq("level1") | (level.eq("level2") & score_pass))

    # Missingness screen covers the ten designated collection groups.
    missing_rates = {t: calculate_missing_rate(raw, cols) for t, cols in all_groups.items()}
    rate_df = pd.DataFrame(missing_rates)
    missing_pass = rate_df.le(MAX_MISSING).all(axis=1)
    historical_final = historical_pre & missing_pass
    level_final = level_pre & missing_pass
    final_mask = historical_final if IDENTIFICATION_RULE == "historical_score" else level_final

    audit = pd.DataFrame({
        FEATURE: raw[FEATURE], "level": raw["level"], "MS2_score": score,
        "QC_CV_percent": cv, "passes_CV": cv_pass,
        "historical_score_pass_before_missing": historical_pre,
        "level_specific_pass_before_missing": level_pre,
    })
    for t in SCREEN_TIMEPOINTS:
        audit[f"missing_rate_{t}"] = missing_rates[t]
    audit["passes_all_10_missing_rates"] = missing_pass
    audit["historical_selected"] = historical_final
    audit["level_specific_selected"] = level_final
    audit["final_selected"] = final_mask
    audit.to_csv(OUTPUT_DIR / "metabolite_filter_audit.tsv", sep="\t", index=False)
    chosen_features = raw.loc[final_mask, FEATURE].astype(str).tolist()
    if not chosen_features:
        raise ValueError("No metabolites passed filters; examine filter audit")
    (OUTPUT_DIR / "metabolite_selected_ids.txt").write_text("\n".join(chosen_features) + "\n", encoding="utf-8")

    # Normalize ALL intensities first (including QC and internal standards), as
    # in the R workflow. Then use the global minimum of this full normalized table.
    def standard_row(name):
        matches = raw.loc[raw[FEATURE].astype(str).eq(name)]
        if len(matches) != 1:
            raise ValueError(f"Internal standard {name} missing or duplicated")
        result = matches.iloc[0][sample_cols].astype(float)
        if result.isna().any() or (~np.isfinite(result)).any() or (result <= 0).any():
            raise ValueError(f"Internal standard {name} has missing, zero or invalid intensity")
        return result

    is1 = standard_row("IS1")
    is4 = standard_row("IS4")
    types = raw["type"].astype(str).str.upper().str.strip()
    if (~types.isin(["NEG", "POS"]) & ~is_internal_standard).any():
        bad = raw.loc[~types.isin(["NEG", "POS"]) & ~is_internal_standard, FEATURE].head().tolist()
        raise ValueError(f"Some metabolite ionization types are not NEG/POS: {bad}")

    norm = raw.copy()
    neg = types.eq("NEG")
    pos = types.eq("POS")
    norm.loc[neg, sample_cols] = norm.loc[neg, sample_cols].div(is1, axis="columns")
    norm.loc[pos, sample_cols] = norm.loc[pos, sample_cols].div(is4, axis="columns")
    arr = norm[sample_cols].to_numpy(dtype=float)
    if (arr < 0).any():
        raise ValueError("Normalized values contain negatives")
    good = arr[np.isfinite(arr) & (arr != 0)]
    if len(good) == 0 or good.min() <= 0:
        raise ValueError("No positive nonzero normalized value to compute imputation")
    fill_value = float(good.min() / 2)
    n_zero = int(np.count_nonzero(arr == 0))
    n_na = int(np.isnan(arr).sum())
    norm[sample_cols] = norm[sample_cols].replace(0, np.nan).fillna(fill_value)
    norm.to_csv(OUTPUT_DIR / "metabolite_normalized_matrix.tsv", sep="\t", index=False)

    final_cols = [c for c in sample_cols if sample_info(c)
                  and sample_info(c)[1] in FINAL_TIMEPOINTS
                  ]
    if not final_cols:
        raise ValueError("No samples remain after sample selection")
    final_wide = norm.loc[final_mask, [FEATURE, *final_cols]].copy()
    final_wide.to_csv(OUTPUT_DIR / "metabolite_normalized_screened.tsv", sep="\t", index=False)

    # Analysis matrix: rows = id x time, columns = metabolites.
    sample_matrix = final_wide.set_index(FEATURE).T
    sample_matrix.insert(0, "id", [sample_info(c)[0] for c in sample_matrix.index])
    sample_matrix.insert(1, "time", [sample_info(c)[1] for c in sample_matrix.index])
    sample_matrix = sample_matrix.sort_values(
        ["id", "time"], key=lambda s: s.map(ORDER) if s.name == "time" else s
    ).reset_index(drop=True)
    sample_matrix.to_csv(OUTPUT_DIR / "metabolite_imputed_matrix.tsv", sep="\t", index=False)
    if WRITE_LOG2:
        feat = list(chosen_features)
        if (sample_matrix[feat].to_numpy(dtype=float) <= 0).any():
            raise ValueError("Cannot log2 nonpositive values")
        log2_matrix = sample_matrix.copy()
        log2_matrix[feat] = np.log2(log2_matrix[feat].astype(float))
        log2_matrix.to_csv(OUTPUT_DIR / "metabolite_log2_candidate.tsv", sep="\t", index=False)

    note = (
        f"raw_features_including_IS={len(raw)}\n"
        f"historical_CV_and_MS2_pass={int(historical_pre.sum())}\n"
        f"historical_CV_MS2_and_10time_missing_pass={int(historical_final.sum())}\n"
        f"level_specific_CV_and_ID_pass={int(level_pre.sum())}\n"
        f"level_specific_with_10time_missing_pass={int(level_final.sum())}\n"
        f"applied_identification_rule={IDENTIFICATION_RULE}\n"
        f"final_selected_features={len(chosen_features)}\n"
        f"screen_missing_mode={MISSING_RATE_MODE}\n"
        f"normalized_full_matrix_zero_count={n_zero}\n"
        f"normalized_full_matrix_na_count={n_na}\n"
        f"imputation_global_min_half={fill_value:.10g}\n"
        f"n_final_samples={len(sample_matrix)}\n"
        f"n_final_participants={sample_matrix['id'].nunique()}\n"
        "QC_CV_calculated_on=raw_unnormalized_QC_values\n"
        "normalization=NEG/IS1;POS/IS4\n"
        "imputation_minimum_population=all_rows_and_samples_in_normalized_full_matrix\n"
        f"log2_written={WRITE_LOG2}\n"
    )
    (OUTPUT_DIR / "metabolite_processing_summary.tsv").write_text(note, encoding="utf-8")
    print(note)
    if int(historical_pre.sum()) != 1453 or int(historical_final.sum()) != 1445:
        print("[NOTE] Historical counts differ from 1453 -> 1445; inspect input, NA vs zero, or QC columns")
    if IDENTIFICATION_RULE == "level_specific":
        print("[NOTE] Final count need not be 1445: level1/level2 rule differs from the original screen")

if __name__ == "__main__":
    main()
