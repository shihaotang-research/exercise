"""Figure 2B: detected molecule counts and summaries."""
# Compact QC plots:
# Identified proteins / metabolites across W0 and W8
#
# X-axis:
#   Baseline / 2 min / 30 min / 60 min
#
# W0 and W8 are dodged within each time point
#
# Colors:
#   W0 = #2E8AC4
#   W8 = #E64B87
#
# Y-axis starts from 0

import warnings
warnings.filterwarnings("ignore")

import re
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

# 1. Input files

DATA_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
PROTEIN_FILE = DATA_DIR / "protein_raw.tsv"
PROTEIN_LIST = DATA_DIR / "protein_ids.txt"

METABOLITE_FILE = DATA_DIR / "metabolite_raw.tsv"
METABOLITE_LIST = DATA_DIR / "metabolite_ids.txt"

# 2. Output files

PROTEIN_PDF = "Protein_identified_compact_from0.pdf"

METABOLITE_PDF = "Metabolite_identified_compact_from0.pdf"

COMBINED_PDF = "Protein_Metabolite_identified_compact_from0.pdf"
COMBINED_PNG = "Protein_Metabolite_identified_compact_from0.png"

# 3. Colors

COLORS = {
    "W0": "#2E8AC4",
    "W8": "#E64B87"
}

# 4. Final panel size

PANEL_W = 2.05  # slightly narrower (was 2.20)
PANEL_H = 1.90  # slightly taller (was 1.75)

# 5. Global plotting style

plt.rcParams.update({
    "font.family": "Arial",

    "font.size": 7,

    "text.color": "#000000",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",

    "axes.linewidth": 0.8,

    # vector-editable PDF text
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

sns.set_style("ticks")

# 6. Time-point definitions

TIME_MAP = {

    # W0
    "A":  ("W0", "Baseline"),
    "B1": ("W0", "2 min"),
    "B2": ("W0", "30 min"),
    "B3": ("W0", "60 min"),

    # W8
    "D":  ("W8", "Baseline"),
    "F1": ("W8", "2 min"),
    "F2": ("W8", "30 min"),
    "F3": ("W8", "60 min")
}

TIME_ORDER = [
    "Baseline",
    "2 min",
    "30 min",
    "60 min"
]

WEEK_ORDER = [
    "W0",
    "W8"
]

# 7. Read feature list

def read_list(path):

    with open(path, "r") as f:

        return [
            line.strip()
            for line in f
            if line.strip()
        ]

# 8. Parse sample name

def parse_sample(sample_name):

    s = str(sample_name).strip()

    # QC
    if "QC" in s.upper():
        return None, None

        # match time point at end
    match = re.search(
        r"(A|B1|B2|B3|D|F1|F2|F3)$",
        s,
        flags=re.IGNORECASE
    )

    if match is None:
        return None, None

    tp = match.group(1).upper()

    return TIME_MAP.get(
        tp,
        (None, None)
    )

# 9. Load data and count identified molecules

def load_and_count_data(
    file_path,
    list_path,
    data_type="Protein"
):

    print("\n" + "=" * 70)
    print(data_type.upper())
    print("=" * 70)

    print("Input data  :", file_path)
    print("Feature list:", list_path)

    # Feature list

    keep_ids = set(
        read_list(list_path)
    )

    print(
        "IDs in list:",
        len(keep_ids)
    )

    # Read matrix

    df = pd.read_csv(
        file_path,
        sep=None,
        engine="python"
    )

    id_col = df.columns[0]

    # Keep requested proteins / metabolites

    df = df[
        df[id_col]
        .astype(str)
        .isin(keep_ids)
    ].copy()

    print(
        "Matched features:",
        len(df)
    )

    # Biological sample columns

    sample_info = []

    for col in df.columns:

        if col == id_col:
            continue

        week, time = parse_sample(col)

        if week is None:
            continue

        sample_info.append({
            "Sample": col,
            "Week": week,
            "Time": time
        })

    sample_info = pd.DataFrame(
        sample_info
    )

    if sample_info.empty:

        raise ValueError(
            f"{data_type}: no valid biological sample columns found."
        )

    print(
        "Biological samples:",
        len(sample_info)
    )

    # Count values > 0

    results = []

    for _, row in sample_info.iterrows():

        sample = row["Sample"]

        values = pd.to_numeric(
            df[sample],
            errors="coerce"
        )

        count = int(
            (values > 0).sum()
        )

        results.append({
            "Sample": sample,
            "Week": row["Week"],
            "Time": row["Time"],
            "Count": count
        })

    counts = pd.DataFrame(
        results
    )

    # Plotting order

    counts["Time"] = pd.Categorical(
        counts["Time"],
        categories=TIME_ORDER,
        ordered=True
    )

    counts["Week"] = pd.Categorical(
        counts["Week"],
        categories=WEEK_ORDER,
        ordered=True
    )

    counts = (
        counts
        .sort_values(
            [
                "Time",
                "Week",
                "Sample"
            ]
        )
        .reset_index(drop=True)
    )

    # Summary

    summary = (
        counts
        .groupby(
            ["Week", "Time"],
            observed=True
        )["Count"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max"
            ]
        )
    )

    print("\nSummary:")
    print(summary)

    return counts

# 10. Y-axis starts from 0

def set_zero_ylim(
    ax,
    values
):

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna()

    if len(values) == 0:
        return

    ymax = values.max()

    # leave ~5% empty space above highest point
    upper = ymax * 1.05

    ax.set_ylim(
        0,
        upper
    )

    # around 4-5 major ticks
    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=5,
            integer=True
        )
    )

# 11. Draw one compact panel

def draw_compact_panel(
    ax,
    data,
    data_type="Protein",
    show_legend=True
):

    # Individual samples

    sns.stripplot(
        data=data,

        x="Time",
        y="Count",

        hue="Week",

        order=TIME_ORDER,
        hue_order=WEEK_ORDER,

        palette=COLORS,

        # W0 and W8 side by side
        dodge=True,

        # slight jitter
        jitter=0.16,

        size=3.6,

        alpha=0.88,

        edgecolor="none",
        linewidth=0,

        ax=ax
    )

    # Y axis starts from zero

    set_zero_ylim(
        ax,
        data["Count"]
    )

    # X-axis

    ax.set_xlabel("")

    ax.set_xticks(
        range(
            len(TIME_ORDER)
        )
    )

    ax.set_xticklabels(
        [
            "Baseline",
            "2 min",
            "30 min",
            "60 min"
        ],
        rotation=0,
        ha="center",
        fontsize=6.6,
        color="#000000"
    )

    # Y-axis label

    if data_type == "Protein":

        ylabel = "Identified proteins"

    else:

        ylabel = "Identified metabolites"

    ax.set_ylabel(
        ylabel,
        fontsize=7.2,
        labelpad=3,
        color="#000000"
    )

    # Tick style

    ax.tick_params(
        axis="x",
        labelsize=6.6,
        width=0.8,
        length=2.5,
        pad=2,
        colors="#000000"
    )

    ax.tick_params(
        axis="y",
        labelsize=6.5,
        width=0.8,
        length=2.5,
        pad=2,
        colors="#000000"
    )

    # Frame

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    ax.grid(False)

    # Remove seaborn legend

    if ax.legend_ is not None:
        ax.legend_.remove()

    # Custom legend

    if show_legend:

        legend_handles = [

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLORS["W0"],
                markeredgecolor="none",
                markersize=4.3,
                label="W0"
            ),

            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=COLORS["W8"],
                markeredgecolor="none",
                markersize=4.3,
                label="W8"
            )
        ]

        ax.legend(
            handles=legend_handles,

            loc="lower center",

            bbox_to_anchor=(
                0.5,
                1.01
            ),

            ncol=2,

            frameon=False,

            fontsize=6.5,

            handletextpad=0.3,
            columnspacing=0.9,

            borderaxespad=0
        )

# 12. Load data

protein_counts = load_and_count_data(
    PROTEIN_FILE,
    PROTEIN_LIST,
    data_type="Protein"
)

metabolite_counts = load_and_count_data(
    METABOLITE_FILE,
    METABOLITE_LIST,
    data_type="Metabolite"
)

# 13. Protein individual figure

fig, ax = plt.subplots(
    figsize=(
        PANEL_W,
        PANEL_H
    ),
    facecolor="white"
)

draw_compact_panel(
    ax,
    protein_counts,
    data_type="Protein",
    show_legend=True
)

plt.subplots_adjust(
    left=0.26,
    right=0.98,
    bottom=0.19,
    top=0.85
)

fig.savefig(
    PROTEIN_PDF,
    format="pdf",
    facecolor="white"
)

plt.close(fig)

print(
    "\nSaved:",
    PROTEIN_PDF
)

# 14. Metabolite individual figure

fig, ax = plt.subplots(
    figsize=(
        PANEL_W,
        PANEL_H
    ),
    facecolor="white"
)

draw_compact_panel(
    ax,
    metabolite_counts,
    data_type="Metabolite",
    show_legend=True
)

plt.subplots_adjust(
    left=0.26,
    right=0.98,
    bottom=0.19,
    top=0.85
)

fig.savefig(
    METABOLITE_PDF,
    format="pdf",
    facecolor="white"
)

plt.close(fig)

print(
    "Saved:",
    METABOLITE_PDF
)

# 15. Combined Protein + Metabolite figure

fig, axes = plt.subplots(
    1,
    2,

    figsize=(
        4.30,  # was 4.55
        1.98   # was 1.80
    ),

    facecolor="white"
)

# Protein

draw_compact_panel(
    axes[0],
    protein_counts,
    data_type="Protein",
    show_legend=False
)

# Metabolite

draw_compact_panel(
    axes[1],
    metabolite_counts,
    data_type="Metabolite",
    show_legend=False
)

# 16. Shared legend

legend_handles = [

    Line2D(
        [0],
        [0],

        marker="o",
        linestyle="",

        markerfacecolor=COLORS["W0"],
        markeredgecolor="none",

        markersize=4.5,

        label="W0"
    ),

    Line2D(
        [0],
        [0],

        marker="o",
        linestyle="",

        markerfacecolor=COLORS["W8"],
        markeredgecolor="none",

        markersize=4.5,

        label="W8"
    )
]

fig.legend(
    handles=legend_handles,

    loc="upper center",

    bbox_to_anchor=(
        0.50,
        0.995
    ),

    ncol=2,

    frameon=False,

    fontsize=6.8,

    handletextpad=0.35,
    columnspacing=1.0
)

# 17. Layout

plt.subplots_adjust(
    left=0.14,
    right=0.985,

    bottom=0.19,
    top=0.85,

    wspace=0.42
)

# 18. Save combined

fig.savefig(
    COMBINED_PDF,
    format="pdf",
    facecolor="white"
)

fig.savefig(
    COMBINED_PNG,
    format="png",
    dpi=600,
    facecolor="white"
)

plt.show()

# 19. Final summary

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(PROTEIN_PDF)
print(METABOLITE_PDF)
print(COMBINED_PDF)
print(COMBINED_PNG)
