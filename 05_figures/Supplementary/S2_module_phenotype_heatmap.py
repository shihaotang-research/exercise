#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Figure S2: module–phenotype association heatmap."""

from pathlib import Path
import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# 1. PATHS AND GLOBAL SETTINGS

ROOT = Path(
    os.environ.get(
        "MULTIOMICS_DATA_DIR",
        str(Path(__file__).resolve().parents[2] / "data")
    )
)

STATIC_DIR = ROOT / "resting_module_phenotype"
DYNAMIC_DIR = ROOT / "acute_module_phenotype"

OUTPUT_PDF = ROOT / "Supplementary_Figure_S2_Module_Phenotype_Heatmaps.pdf"

SUPP_FIG = "S2"  # Uploaded filename S2; supplied PDF caption still says S1. Confirm final numbering.

# Keep the original significance criterion
SIG_COL = "empirical_p"
SIG_CUTOFF = 0.05

# If True, annotate all cells; otherwise only significant cells
ANNOTATE_ALL = False

RHO_LIMIT = 0.8

PHENOS = [
    "BMI",
    "WC",
    "FatMass",
    "BFP",
    "VO2",
    "O2pluse",
    "RHR",
    "VCmax",
]

PHENO_LABELS = [
    "BMI",
    "WC",
    "Fat mass",
    "BFP",
    r"VO$_2$",
    r"O$_2$ pulse",
    "RHR",
    r"VC$_{max}$",
]

# 2. MODULE COLORS

STATIC_BG = {
    "M1": "#F4C7C3",
    "M2": "#BFD7F2",
    "M3": "#BFE3C0",
    "M4": "#F3E2A7",
    "M5": "#D9C8F0",
    "M6": "#BFE3EA",
    "M7": "#F7D6BF",
    "M8": "#C9D0D8",
    "N1": "#F6D1E6",
    "N2": "#D8D1C5",
    "N3": "#E6CFC4",
}

STATIC_EDGE = {
    "M1": "#E78B84",
    "M2": "#7FB0E3",
    "M3": "#7FC68A",
    "M4": "#D8B94F",
    "M5": "#B497DE",
    "M6": "#7DC8D6",
    "M7": "#E5AE7B",
    "M8": "#93A0AD",
    "N1": "#E49BC4",
    "N2": "#B7AA93",
    "N3": "#D0A792",
}

DYNAMIC_BG = {
    **STATIC_BG,
    "M9": "#D6E8C5",
}

DYNAMIC_EDGE = {
    **STATIC_EDGE,
    "M9": "#9FC37A",
}

# 3. MATPLOTLIB SETTINGS

plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

CMAP = LinearSegmentedColormap.from_list(
    "diverging_rho",
    ["#2F6DAE", "#FAFAFA", "#D75A4A"]
)

NORM = TwoSlopeNorm(
    vmin=-RHO_LIMIT,
    vcenter=0,
    vmax=RHO_LIMIT
)

# 4. FIND INPUT DIRECTORY

def input_dir(preferred, name):
    backup = (
        ROOT /
        ("resting_module_resolution_check" if name == "Static" else "acute_module_resolution_check") /
        "by_parameter" /
        "gamma_1.20_min_10_weighted"
    )

    files = [
        "module_scoring_groups.tsv",
        "module_phenotype_all_tests.tsv",
    ]

    for directory in [preferred, backup]:
        if all((directory / f).is_file() for f in files):
            return directory

    raise FileNotFoundError(
        f"{name}: cannot find results.\n"
        f"Checked:\n{preferred}\n{backup}"
    )

# 5. LOAD RESULTS

def load_results(preferred, name, palette):
    directory = input_dir(preferred, name)

    mapping = pd.read_csv(
        directory / "module_scoring_groups.tsv",
        sep="\t"
    )
    corr = pd.read_csv(
        directory / "module_phenotype_all_tests.tsv",
        sep="\t"
    )

    for col in ["module", "kind"]:
        if col not in mapping.columns:
            raise ValueError(f"{name}: mapping missing column {col}")

    for col in ["module", "phenotype", "rho", SIG_COL]:
        if col not in corr.columns:
            raise ValueError(f"{name}: correlation table missing column {col}")

    if "resolution" in corr.columns:
        if not np.allclose(pd.to_numeric(corr["resolution"]), 1.2):
            raise ValueError(f"{name}: resolution is not 1.2")

    if "min_size" in corr.columns:
        if not pd.to_numeric(corr["min_size"]).eq(10).all():
            raise ValueError(f"{name}: min_size is not 10")

    if "weight_mode" in corr.columns:
        if not corr["weight_mode"].astype(str).eq("weighted").all():
            raise ValueError(f"{name}: weight_mode is not weighted")

    mapping["module"] = mapping["module"].astype(str)
    corr["module"] = corr["module"].astype(str)

    if mapping["module"].duplicated().any():
        raise ValueError(f"{name}: duplicated module IDs")

    if corr.duplicated(["module", "phenotype"]).any():
        raise ValueError(f"{name}: duplicated module-phenotype rows")

    modules = mapping["module"].tolist()

    if set(modules) != set(corr["module"].tolist()):
        raise ValueError(f"{name}: module sets differ between mapping and correlation tables")

    if set(corr["phenotype"].tolist()) != set(PHENOS):
        raise ValueError(f"{name}: phenotype list is incomplete or inconsistent")

    if len(corr) != len(modules) * len(PHENOS):
        raise ValueError(f"{name}: incomplete module-phenotype table")

    missing_colors = set(modules) - set(palette.keys())
    if missing_colors:
        raise ValueError(f"{name}: missing module colors for {sorted(missing_colors)}")

    valid_kinds = {"Matched", "W0_only", "W8_new"}
    if not mapping["kind"].isin(valid_kinds).all():
        raise ValueError(f"{name}: invalid module kinds detected")

    corr["rho"] = pd.to_numeric(corr["rho"], errors="coerce")
    corr[SIG_COL] = pd.to_numeric(corr[SIG_COL], errors="coerce")

    if corr["rho"].abs().gt(1 + 1e-8).any():
        raise ValueError(f"{name}: invalid rho values")

    valid_p = corr[SIG_COL].dropna()
    if ((valid_p < 0) | (valid_p > 1)).any():
        raise ValueError(f"{name}: invalid significance values")

    print(f"[{name}] {len(modules)} modules; {len(corr)} tests")
    print("Input:", directory)

    return mapping, corr

# 6. DRAW ONE HEATMAP PANEL

def draw_heatmap(fig, pos, results, bg, edge):
    mapping, corr = results

    mids = mapping["module"].tolist()
    kinds = mapping["kind"].tolist()

    rho = (
        corr.pivot(index="module", columns="phenotype", values="rho")
        .reindex(index=mids, columns=PHENOS)
        .to_numpy(float)
    )

    pval = (
        corr.pivot(index="module", columns="phenotype", values=SIG_COL)
        .reindex(index=mids, columns=PHENOS)
        .to_numpy(float)
    )

    y = np.arange(len(mids), dtype=float)

    # Add small gaps between module-status groups
    for i in range(1, len(y)):
        if kinds[i] != kinds[i - 1]:
            y[i:] += 0.24

    ax = fig.add_axes(pos)

    for i, m in enumerate(mids):
        # Module label background
        ax.add_patch(
            Rectangle(
                (-1.34, y[i] - 0.43),
                0.70,
                0.86,
                facecolor=bg[m],
                edgecolor="none",
                clip_on=False,
            )
        )

        # Dark left strip
        ax.add_patch(
            Rectangle(
                (-1.34, y[i] - 0.43),
                0.075,
                0.86,
                facecolor=edge[m],
                edgecolor="none",
                clip_on=False,
            )
        )

        ax.text(
            -0.99,
            y[i],
            m,
            ha="center",
            va="center",
            fontsize=9.0,
            color="black",
            clip_on=False,
        )

        for j in range(8):
            v = rho[i, j]

            if np.isfinite(v):
                face = CMAP(NORM(np.clip(v, -RHO_LIMIT, RHO_LIMIT)))
            else:
                face = "#E7E8EA"

            ax.add_patch(
                Rectangle(
                    (j - 0.47, y[i] - 0.45),
                    0.94,
                    0.90,
                    facecolor=face,
                    edgecolor="white",
                    linewidth=0.85,
                )
            )

            significant = np.isfinite(pval[i, j]) and (pval[i, j] < SIG_CUTOFF)

            if np.isfinite(v) and (ANNOTATE_ALL or significant):
                label = f"{v:+.2f}"
                if significant:
                    label += "*"

                txt_color = "white" if abs(v) > 0.58 else "black"

                ax.text(
                    j,
                    y[i],
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color=txt_color,
                )

        # Group separator
        if i > 0 and kinds[i] != kinds[i - 1]:
            ax.axhline(
                (y[i - 1] + y[i]) / 2,
                color="#B7BFC7",
                linewidth=0.8,
            )

    # Vertical split between phenotype categories
    ax.axvline(
        3.5,
        color="#98A4AE",
        linewidth=0.9,
    )

    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.55, y[-1] + 0.55)
    ax.invert_yaxis()

    ax.set_yticks([])

    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(PHENO_LABELS, fontsize=8.6)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Category labels
    ax.text(
        1.5,
        1.16,
        "Body composition",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#5E6973",
        clip_on=False,
    )

    ax.text(
        5.5,
        1.16,
        "Cardiorespiratory",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9.0,
        color="#5E6973",
        clip_on=False,
    )

# 7. MAKE FULL FIGURE

def make_figure(static, dynamic):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")

    # A. Resting-state network

    fig.text(
        0.075, 0.967,
        "A",
        fontsize=11,
        fontweight="bold",
        ha="left", va="center"
    )

    fig.text(
        0.108, 0.967,
        "Resting-state network",
        fontsize=10.8,
        fontweight="bold",
        ha="left", va="center"
    )

    fig.text(
        0.354, 0.967,
        r"(module $\Delta$Baseline vs. phenotype change)",
        fontsize=8.9,
        ha="left", va="center"
    )

    draw_heatmap(
        fig,
        [0.175, 0.626, 0.690, 0.255],
        static,
        STATIC_BG,
        STATIC_EDGE,
    )

    # B. Acute exercise-response network

    fig.text(
        0.075, 0.590,
        "B",
        fontsize=11,
        fontweight="bold",
        ha="left", va="center"
    )

    fig.text(
        0.108, 0.590,
        "Acute exercise-response network",
        fontsize=10.8,
        fontweight="bold",
        ha="left", va="center"
    )

    fig.text(
        0.465, 0.590,
        r"(module $\Delta$AbsAUC vs. phenotype change)",
        fontsize=8.9,
        ha="left", va="center"
    )

    draw_heatmap(
        fig,
        [0.175, 0.274, 0.690, 0.248],
        dynamic,
        DYNAMIC_BG,
        DYNAMIC_EDGE,
    )

    # Shared colorbar

    cax = fig.add_axes([0.900, 0.435, 0.018, 0.220])

    sm = plt.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, cax=cax)

    cb.set_ticks([-RHO_LIMIT, 0, RHO_LIMIT])
    cb.set_ticklabels([f"{-RHO_LIMIT:.1f}", "0.0", f"{RHO_LIMIT:.1f}"])
    cb.ax.tick_params(labelsize=8.0, length=0, pad=5)
    cb.outline.set_linewidth(0.45)
    cb.set_label(r"Spearman $\rho$", fontsize=9.0, labelpad=6)

    # Caption

    if SIG_COL == "empirical_p":
        sig_text = "empirical permutation P"
    elif SIG_COL == "empirical_fdr_global":
        sig_text = "global BH-FDR of permutation P"
    else:
        sig_text = SIG_COL

    if ANNOTATE_ALL:
        label_text = "All correlations are labeled"
    else:
        label_text = "Only significant correlations are labeled"

    caption = (
        f"Supplementary Figure {SUPP_FIG}. Associations between network-module changes and "
        "changes in eight phenotypes. "
        "(A) Resting-state network: module scores summarize mean changes in baseline molecular "
        "abundance (W8 - W0). "
        "(B) Acute exercise-response network: module scores summarize mean changes in acute-response "
        "absolute AUC (W8 - W0). "
        "Cell color indicates Spearman correlation (rho). "
        f"{label_text}; * denotes {sig_text} < {SIG_CUTOFF:g}. "
        "Original analyses used 10,000 label permutations. "
        "Matched modules use the shared molecules between W0 and W8, whereas unmatched modules "
        "use all molecules in the W0-only or W8-new module. "
        "Weighted Louvain clustering used gamma = 1.2 and minimum module size = 10. "
        "Pastel module labels follow the main-figure palette, and horizontal separators indicate "
        "module matching groups."
    )

    fig.text(
        0.080,
        0.215,
        textwrap.fill(caption, width=108),
        fontsize=7.8,
        ha="left",
        va="top",
        color="black",
        linespacing=1.28,
    )

    # Save PDF only

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_PDF,
        format="pdf",
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved PDF:", OUTPUT_PDF.resolve())

# 8. MAIN

def main():
    static = load_results(
        STATIC_DIR,
        "Static",
        STATIC_BG,
    )

    dynamic = load_results(
        DYNAMIC_DIR,
        "Dynamic",
        DYNAMIC_BG,
    )

    make_figure(static, dynamic)

if __name__ == "__main__":
    main()
