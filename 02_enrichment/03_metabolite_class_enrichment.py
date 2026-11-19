# -*- coding: utf-8 -*-
"""Metabolite class enrichment with the 1,142-metabolite universe.
Standard fold enrichment = (a/n)/(K/N); Fisher exact test and BH-FDR."""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

DATA_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
UNIVERSE_FILE = DATA_DIR / "metabolite_ids.txt"
TARGET_FILE = DATA_DIR / "metabolite_response_classes.tsv"
ANNOTATION_FILE = DATA_DIR / "metabolite_annotations.tsv"
OUTPUT_FILE = DATA_DIR / "metabolite_class_enrichment.tsv"

EXPECTED_BACKGROUND_N = 1142
GROUP_ORDER = ["Baseline-only", "Dual-response", "Trend-only"]

def clean_text(value):
    if pd.isna(value):
        return ""
    s = " ".join(str(value).replace("\r", " ").replace("\n", " ").split())
    return "" if s.lower() in {"", "nan", "none", "na", "n/a"} else s

def normalize_group(value):
    s = clean_text(value)
    key = " ".join(s.lower().replace("_", " ").replace("-", " ").replace("+", " ").split())
    aliases = {
        "baseline only": "Baseline-only",
        "trend only": "Trend-only",
        "dynamic only": "Trend-only",
        "both": "Dual-response",
        "baseline trend": "Dual-response",
        "baseline and trend": "Dual-response",
        "baseline and dynamic": "Dual-response",
        "dual response": "Dual-response",
    }
    return aliases.get(key, s)

def bh_fdr(p_values):
    """Load and validate the input data using the original definitions."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("BH adjustment requires a one-dimensional array of finite P values in [0, 1].")
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out

def join_unique(values):
    seen, output = set(), []
    for value in values:
        s = clean_text(value)
        if s and s not in seen:
            seen.add(s)
            output.append(s)
    return " | ".join(output)

def resolve_file(name):
    p = Path(name).expanduser()
    candidates = [p]
    if not p.is_absolute() and "__file__" in globals():
        candidates.append(Path(__file__).resolve().parent / p)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"File not found: {name}; working directory: {Path.cwd()}")

def load_universe(filename):
    """Load and validate the input data using the original definitions."""
    path = resolve_file(filename)
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            mid = clean_text(line.split()[0].strip('"\''))
            if mid.lower() in {"metabolite", "metabolite_id", "id"}:
                continue
            if mid:
                ids.append(mid)
    if len(ids) != len(set(ids)):
        duplicate_ids = pd.Series(ids).loc[lambda s: s.duplicated()].unique().tolist()
        raise ValueError(f"Duplicate IDs in the 1,142-metabolite candidate list, examples: {duplicate_ids[:8]}")
    if len(ids) != EXPECTED_BACKGROUND_N:
        raise ValueError(
            f"Candidate list expected {EXPECTED_BACKGROUND_N} unique metabolites, found {len(ids)}. "
            "Verify that UNIVERSE_FILE is the analysis input list. "
        )
    return ids

def read_table(filename, required):
    df = pd.read_csv(resolve_file(filename), sep="\t", encoding="utf-8-sig",
                     dtype=str, low_memory=False)
    df.columns = [str(col).strip() for col in df.columns]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{filename} Missing columns: {sorted(missing)}")
    return df

def prepare_data():
    universe_ids = load_universe(UNIVERSE_FILE)
    universe_set = set(universe_ids)

    annotation = read_table(ANNOTATION_FILE, ["Metabolite", "Name", "main_class"])
    annotation = annotation[["Metabolite", "Name", "main_class"]].copy()
    for col in annotation.columns:
        annotation[col] = annotation[col].map(clean_text)

    annotation = annotation.loc[annotation["Metabolite"].isin(universe_set)].copy()

    valid_class = annotation["main_class"].replace("", "Unclassified")
    conflicts = (annotation.assign(_class=valid_class)
                 .groupby("Metabolite")["_class"].nunique())
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise ValueError(f"Conflicting main_class annotations for the same metabolite: {conflicts.index.tolist()[:8]}")

    annotation = annotation.drop_duplicates("Metabolite", keep="first")
    missing_ids = sorted(universe_set - set(annotation["Metabolite"]))
    if missing_ids:
        raise ValueError(
            f"Among 1,142 candidate metabolites, {len(missing_ids)} lack annotations; "
            f"examples: {missing_ids[:8]}. check ANNOTATION_FILE. "
        )

    background = pd.DataFrame({"Metabolite": universe_ids}).merge(
        annotation, on="Metabolite", how="left", validate="one_to_one"
    )
    background["main_class"] = background["main_class"].replace("", "Unclassified")
    assert len(background) == EXPECTED_BACKGROUND_N

    target = read_table(TARGET_FILE, ["Metabolite", "Group"])
    target = target[["Metabolite", "Group"]].copy()
    target["Metabolite"] = target["Metabolite"].map(clean_text)
    target["Group"] = target["Group"].map(normalize_group)
    target = target.loc[target["Metabolite"] != ""].copy()

    unknown = sorted(set(target["Group"]) - set(GROUP_ORDER))
    if unknown:
        raise ValueError(f"Unrecognized group names: {unknown}; update normalize_group aliases. ")

    overlaps = target.groupby("Metabolite")["Group"].nunique()
    overlaps = overlaps[overlaps > 1]
    if not overlaps.empty:
        raise ValueError(f"Metabolites assigned to multiple response classes: {overlaps.index.tolist()[:8]}")
    target = target.drop_duplicates("Metabolite", keep="first")

    missing_target = sorted(set(target["Metabolite"]) - universe_set)
    if missing_target:
        raise ValueError(
            f"There are {len(missing_target)} group metabolites absent from the 1,142 candidates; "
            f"examples: {missing_target[:8]}. Verify dataset consistency. "
        )

    background = background.merge(target, on="Metabolite", how="left", validate="one_to_one")
    return background

def calculate_enrichment(background):
    n_background = len(background)
    assert n_background == EXPECTED_BACKGROUND_N

    terms = sorted(background.loc[
        background["main_class"] != "Unclassified", "main_class"
    ].unique())
    if not terms:
        raise ValueError("No usable main_class annotations in the 1,142-metabolite background. ")

    records = []
    for group in GROUP_ORDER:
        group_mask = background["Group"].eq(group)
        n_group = int(group_mask.sum())
        if n_group == 0:
            raise ValueError(f"{group} has no metabolites in the 1,142-metabolite background. ")
        n_rest = n_background - n_group

        for term in terms:
            term_mask = background["main_class"].eq(term)
            a = int((group_mask & term_mask).sum())
            b = n_group - a
            c = int((~group_mask & term_mask).sum())
            d = n_rest - c
            if min(a, b, c, d) < 0 or n_rest <= 0:
                raise ValueError(f"Invalid Fisher contingency table: {group}, {term}")

            odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
            fraction_group = a / n_group
            fraction_rest = c / n_rest
            # Standard over-representation fold enrichment compares the class
            # fraction in the target response group with the full 1,142 background.
            # Do not divide by the complement ('rest') fraction here.
            k_background = int(term_mask.sum())
            background_fraction = k_background / n_background
            fold_enrichment = fraction_group / background_fraction
            expected_count = n_group * background_fraction
            members = background.loc[group_mask & term_mask]
            display_names = members["Name"].where(members["Name"] != "", members["Metabolite"])

            records.append({
                "Group": group,
                "main_class": term,
                "count_in_group": a,
                "n_group": n_group,
                "proportion_in_group": fraction_group,
                "count_in_background": k_background,
                "n_background": n_background,
                "count_in_background_rest": c,
                "n_background_rest": n_rest,
                "proportion_in_background_rest": fraction_rest,
                "proportion_in_background": background_fraction,
                "expected_count": expected_count,
                "fold_enrichment": fold_enrichment,
                "odds_ratio": odds_ratio,
                "pvalue": float(pvalue),
                "metabolite_ids": join_unique(members["Metabolite"]),
                "metabolite_names": join_unique(display_names),
            })

    result = pd.DataFrame(records)
    if not np.allclose(
        result["fold_enrichment"].to_numpy(float),
        (result["count_in_group"] / result["n_group"] /
         (result["count_in_background"] / result["n_background"])).to_numpy(float),
        rtol=1e-12, atol=1e-12,
    ):
        raise AssertionError("Standard fold enrichment verification failed")

    result["fdr_bh"] = result.groupby("Group", sort=False)["pvalue"].transform(bh_fdr)
    tiny = np.finfo(float).tiny
    result["neg_log10_p"] = -np.log10(np.maximum(result["pvalue"].to_numpy(float), tiny))
    result["neg_log10_fdr"] = -np.log10(np.maximum(result["fdr_bh"].to_numpy(float), tiny))
    result["significant_fdr_0.05"] = result["fdr_bh"] < 0.05

    result = result.loc[result["count_in_group"] > 0].copy()
    result["Group"] = pd.Categorical(result["Group"], categories=GROUP_ORDER, ordered=True)
    result = result.sort_values(
        ["Group", "fdr_bh", "pvalue", "main_class"], kind="mergesort"
    ).reset_index(drop=True)
    result["Group"] = result["Group"].astype(str)
    return result

def main():
    background = prepare_data()
    result = calculate_enrichment(background)
    output_path = Path(OUTPUT_FILE).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False, na_rep="NA", float_format="%.8g")

    print(f"Background: {len(background)} (original candidate metabolites)")
    for group in GROUP_ORDER:
        count = int(background["Group"].eq(group).sum())
        significant = int((result["Group"].eq(group) & result["significant_fdr_0.05"]).sum())
        print(f"{group}: n={count}, FDR<0.05 main_class={significant}")
    print(f"Saved: {output_path}")
    return result

if __name__ == "__main__":
    main()
