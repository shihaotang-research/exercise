# -*- coding: utf-8 -*-
"""Figure 5: acute exercise-response network topology summary."""
from __future__ import annotations

import warnings
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 0. Paths and display parameters
ROOT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
DATA_DIR = ROOT / "acute_response_network"

EDGE_FILES = {
    "W0": DATA_DIR / "02_W0_significant_edges_all_signs.txt",
    "W8": DATA_DIR / "02_W8_significant_edges_all_signs.txt",
}
MODULE_FILES = {
    "W0": DATA_DIR / "03_W0_module_assignment.txt",
    "W8": DATA_DIR / "03_W8_module_assignment.txt",
}

OUT_TABLE = DATA_DIR / "07_W0_W8_Delta3TP_basic_network_properties_POLISHED_FINAL.tsv"
OUT_PDF = DATA_DIR / "07_W0_W8_Delta3TP_basic_network_properties_POLISHED_FINAL.pdf"
OUT_PNG = DATA_DIR / "07_W0_W8_Delta3TP_basic_network_properties_POLISHED_FINAL.png"

FIG_W_MM = 48.5
FIG_H_MM = 79.1
DPI = 600
W0_COLOR = "#2E8AC4"
W8_COLOR = "#E64B87"
LINE_COLOR = "#A7AFB6"
GRID_COLOR = "#E8ECF0"
TEXT_COLOR = "#20252A"

# The six original metrics; shortened labels only for the narrow panel.
METRICS = [
    ("active_nodes", "Active nodes"),
    ("edges", "Edges"),
    ("density", "Density"),
    ("mean_degree", "Mean degree"),
    ("avg_clustering", "Avg clustering"),
    ("modularity", "Modularity"),
]

def load_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df.columns = [str(x).strip() for x in df.columns]
    return df

def load_inputs(tp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = load_tsv(EDGE_FILES[tp])
    modules = load_tsv(MODULE_FILES[tp])
    if not {"source", "target", "rho"}.issubset(edges.columns):
        raise ValueError(f"Missing source/target/rho columns in {EDGE_FILES[tp]}")
    if not {"node", "module"}.issubset(modules.columns):
        raise ValueError(f"Missing node/module columns in {MODULE_FILES[tp]}")

    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    edges["rho"] = pd.to_numeric(edges["rho"], errors="coerce")
    edges = edges.dropna(subset=["rho"]).copy()
    edges["abs_rho"] = edges["rho"].abs()
    edges["edge_key"] = [
        (a, b) if a <= b else (b, a)
        for a, b in zip(edges["source"], edges["target"])
    ]
    edges = edges.drop_duplicates("edge_key", keep="first").reset_index(drop=True)
    modules["node"] = modules["node"].astype(str).str.strip()
    modules["module"] = modules["module"].astype(str).str.strip()
    return edges, modules

def make_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for row in edges.itertuples(index=False):
        G.add_edge(str(row.source), str(row.target),
                   rho=float(row.rho), abs_rho=float(row.abs_rho))
    return G

def calculate_existing_modularity(edges: pd.DataFrame,
                                  modules: pd.DataFrame) -> float:
    # Same rule as the original script: positive edges among assigned nodes;
    # use existing communities, weighted by |rho|; no community redetection.
    assigned = modules.loc[
        ~modules["module"].str.lower().isin(
            {"", "nan", "na", "none", "unassigned"}),
        ["node", "module"],
    ].copy()
    nodes = set(assigned["node"])
    positive = edges.loc[
        edges["rho"].gt(0)
        & edges["source"].isin(nodes)
        & edges["target"].isin(nodes)
    ].copy()
    G = make_graph(positive)
    if G.number_of_edges() == 0:
        return np.nan
    graph_nodes = set(G.nodes())
    communities = []
    for _, block in assigned.groupby("module", sort=False):
        group_nodes = set(block["node"]) & graph_nodes
        if group_nodes:
            communities.append(group_nodes)
    covered = set().union(*communities) if communities else set()
    if covered != graph_nodes:
        communities.extend([{n} for n in sorted(graph_nodes - covered)])
    return float(nx.community.modularity(G, communities, weight="abs_rho"))

def calculate_metrics(tp: str, edges: pd.DataFrame,
                      modules: pd.DataFrame) -> dict:
    G = make_graph(edges)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    return {
        "network": tp,
        "active_nodes": n,
        "edges": e,
        "density": nx.density(G) if n > 1 else np.nan,
        "mean_degree": 2.0 * e / n if n else np.nan,
        "avg_clustering": (
            float(nx.average_clustering(G, weight="abs_rho"))
            if n > 1 else np.nan
        ),
        "modularity": calculate_existing_modularity(edges, modules),
    }

def percentage_change(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or abs(a) < 1e-15:
        return np.nan
    return 100.0 * (b / a - 1.0)

def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.2,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

def draw_dumbbell(metrics: pd.DataFrame) -> None:
    configure_style()
    indexed = metrics.set_index("network")
    w0, w8 = indexed.loc["W0"], indexed.loc["W8"]
    changes = np.array([
        percentage_change(float(w0[key]), float(w8[key]))
        for key, _ in METRICS
    ])
    w8_index = 100 + changes  # W0=100 separately for each network metric.
    finite = w8_index[np.isfinite(w8_index)]
    lo = min(90., np.floor((float(finite.min()) - 4.) / 5.) * 5.) if len(finite) else 90.
    hi = max(155., np.ceil((float(finite.max()) + 5.) / 5.) * 5.) if len(finite) else 155.

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    # Reserve top for legend and bottom for the axis title.
    ax = fig.add_axes([0.08, 0.11, 0.87, 0.76])
    ys = np.arange(len(METRICS))[::-1] * 2.05

    ax.set_xlim(lo, hi)
    ax.set_ylim(-1.0, ys[0] + 1.20)
    ax.set_yticks([])

    # x is normalized, y represents metric. Top row: metric / change annotation;
    # below: the dumbbell endpoints. No raw numbers are displayed.
    ytrans = ax.get_yaxis_transform()
    for i, ((key, label), y) in enumerate(zip(METRICS, ys)):
        change = changes[i]
        if i > 0:
            ax.axhline(y + 1.05, color=GRID_COLOR, lw=0.65, zorder=0)

        ax.text(0.01, y + 0.49, label,
                transform=ytrans, ha="left", va="center", fontsize=7.5)
        if np.isfinite(change):
            ax.text(0.99, y + 0.49, f"{change:+.1f}%",
                    transform=ytrans, ha="right", va="center",
                    fontsize=7.5, fontweight="bold")
            end = w8_index[i]
            ax.plot([100, end], [y, y], color=LINE_COLOR,
                    lw=1.6, solid_capstyle="round", zorder=2)
            ax.scatter([100], [y], s=28, facecolors=W0_COLOR,
                       edgecolors="white", linewidths=0.5, zorder=3)
            ax.scatter([end], [y], s=28, facecolors=W8_COLOR,
                       edgecolors="white", linewidths=0.5, zorder=4)
        else:
            ax.text(0.99, y + 0.49, "NA", transform=ytrans,
                    ha="right", va="center", fontsize=7.5)

    xticks = [tick for tick in (90, 100, 120, 140, 160)
              if lo <= tick <= hi]
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=7.1, direction="out",
                   width=0.75, length=2.8, pad=2.2)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#69737A")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_xlabel("W0-normalized index (W0 = 100)", fontsize=7.0, labelpad=4)

    handles = [
        Line2D([0], [0], marker="o", ls="None", markersize=5.4,
               markerfacecolor=W0_COLOR, markeredgecolor="white",
               markeredgewidth=0.5, label="W0"),
        Line2D([0], [0], marker="o", ls="None", markersize=5.4,
               markerfacecolor=W8_COLOR, markeredgecolor="white",
               markeredgewidth=0.5, label="W8"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.50, 0.985), frameon=False, fontsize=7.5,
               columnspacing=1.5, handletextpad=0.3, borderaxespad=0.0)

    # Exact artboard dimensions; do not use bbox_inches='tight'.
    fig.savefig(OUT_PDF, dpi=DPI, facecolor="white")
    fig.savefig(OUT_PNG, dpi=DPI, facecolor="white")
    plt.close(fig)

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tp in ("W0", "W8"):
        edges, modules = load_inputs(tp)
        rows.append(calculate_metrics(tp, edges, modules))
    metrics = pd.DataFrame(rows)

    change_row = {"network": "W8_vs_W0_change_pct"}
    for key, _ in METRICS:
        change_row[key] = percentage_change(
            float(metrics.loc[metrics.network == "W0", key].iloc[0]),
            float(metrics.loc[metrics.network == "W8", key].iloc[0]),
        )
    metrics_out = pd.concat([metrics, pd.DataFrame([change_row])], ignore_index=True)
    metrics_out.to_csv(OUT_TABLE, sep="\t", index=False, na_rep="NA")
    draw_dumbbell(metrics)
    print(OUT_PDF.resolve())

if __name__ == "__main__":
    main()
