# -*- coding: utf-8 -*-

"""Figure 5: acute exercise-response molecular network remodeling."""

from __future__ import annotations

import math

import re

import warnings

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter, MaxNLocator

import networkx as nx

import numpy as np

import pandas as pd

from scipy.sparse.linalg import eigsh

from scipy.stats import kruskal, mannwhitneyu

from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# 0. Parameters

SCRIPT_VERSION = "DELTA3TP_SELECTED5_STATIC_V19_MATCHED_SIZE"

# Run the script from the directory that contains the network-results folder.

BASE_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
NETWORK_DIR = BASE_DIR / "acute_response_network"

OUT_DIR = NETWORK_DIR

ANN_FILE = NETWORK_DIR / "00_selected_molecules_final_same_set.txt"

NETWORK_INPUT_FILES = {

    "W0": NETWORK_DIR / "01_W0_Delta2_Delta30_Delta60_network_input.txt",

    "W8": NETWORK_DIR / "01_W8_Delta2_Delta30_Delta60_network_input.txt",

}

RHO_FILES = {

    "W0": NETWORK_DIR / "02_W0_Delta3TP_Spearman_rho_matrix.txt",

    "W8": NETWORK_DIR / "02_W8_Delta3TP_Spearman_rho_matrix.txt",

}

EDGE_FILES = {

    "W0": NETWORK_DIR / "02_W0_significant_edges_all_signs.txt",

    "W8": NETWORK_DIR / "02_W8_significant_edges_all_signs.txt",

}

MODULE_FILES = {

    "W0": NETWORK_DIR / "03_W0_module_assignment.txt",

    "W8": NETWORK_DIR / "03_W8_module_assignment.txt",

}

OUT_RESPONSE = OUT_DIR / "06_W0_W8_Delta3TP_mean_acute_response_metrics.tsv"

OUT_METRICS = OUT_DIR / "06_W0_W8_Delta3TP_node_remodeling_metrics.tsv"

OUT_NODES = OUT_DIR / "06_W0_W8_Delta3TP_nodes_for_class_comparison.tsv"

OUT_PAIRWISE = OUT_DIR / "06_W0_W8_Delta3TP_pairwise_stats_selected6.tsv"
OUT_SUMMARY = OUT_DIR / "06_W0_W8_Delta3TP_group_summary_selected6.tsv"
OUT_PDF = OUT_DIR / "06_W0_W8_Delta3TP_selected6_metrics_by_class_STATIC_STYLE.pdf"
OUT_PNG = OUT_DIR / "06_W0_W8_Delta3TP_selected6_metrics_by_class_STATIC_STYLE.png"

# This global three-class comparison intentionally uses EVERY selected

# molecule, including isolates.  All available measures are first calculated on

# the complete fixed W0/W8 networks; this choice controls only the rows used

# in the three-class plot/statistics.

#

# Other scopes below are retained only for separately labelled supplementary

# module analyses.  Do not use them for this global all-molecule figure.

#     "module_intersection" retained module at both W0 and W8

#     "module_union"        retained module at W0 or W8

#     "active_in_either"    positive significant edge at W0 or W8

ANALYSIS_NODE_SCOPE = "all_selected"

# Fixed class names, order and colours used across the project.

GROUP_ORDER = ["Baseline-only", "Dual-response", "Trend-only"]

GROUP_COLORS = {

    "Baseline-only": "#4C95BC",

    "Dual-response": "#55A96E",

    "Trend-only": "#E57C5E",

}

GROUP_EDGE_COLORS = {

    "Baseline-only": "#2F6F8F",

    "Dual-response": "#347A49",

    "Trend-only": "#A94F37",

}

# Pairwise figure-wide BH-FDR: all 5 panels x 3 class-pairs = 15 comparisons.
# Report within-metric q values as well, but use p_bh_figure for annotations.
SIGNIF_THRESHOLD = 0.05
GROUP_DISPLAY_NAMES = {
    "Baseline-only": "Baseline-only",
    "Trend-only": "Dynamic-only",
    "Dual-response": "Dual-response",
}
GROUP_TICK_NAMES = {
    "Baseline-only": "Baseline",
    "Trend-only": "Dynamic",
    "Dual-response": "Dual",
}

# Centrality convention for the positive significant-edge graph.

CLOSENESS_USE_WEIGHTED_DISTANCE = True

EIGENVECTOR_USE_EDGE_STRENGTH = True

TURNOVER_EMPTY_AS_NAN = True

RESPONSE_ORDER = ["Delta2", "Delta30", "Delta60"]

ACUTE_TIME_MIN = np.array([0.0, 2.0, 30.0, 60.0], dtype=float)

FONT_FAMILY = "Arial"

# Five panels, with the same per-panel width as the static five-panel figure.
# Compact vertical plotting: avoid the older 0.76 artificial height reduction.
# Data, KDE bandwidth, individual points and statistical tests remain unchanged.
FIG_W_MM = 154.5
FIG_H_MM = 74.0
FIG_W = FIG_W_MM / 25.4
FIG_H = FIG_H_MM / 25.4
FONT_SIZE = 7.0
GROUP_SHORT = {"Baseline-only": "Base", "Dual-response": "Dual", "Trend-only": "Dyn"}
GROUP_DISPLAY = GROUP_DISPLAY_NAMES

DPI = 600

SHOW_GROUP_MEAN_ON_XTICK = True

SHOW_GLOBAL_KW_P = False

# 1. General helpers

def require_file(path: Path) -> None:

    if not path.is_file():

        raise FileNotFoundError(f"Required file not found: {path}")

def read_tsv(path: Path, index_col: int | None = None) -> pd.DataFrame:

    require_file(path)

    table = pd.read_csv(path, sep="\t", index_col=index_col, low_memory=False)

    table.columns = [str(column).strip() for column in table.columns]

    if index_col is not None:

        table.index = table.index.astype(str).str.strip()

    return table

def resolve_column(

    table: pd.DataFrame,

    candidates: list[str],

    required: bool = True,

) -> str | None:

    lookup = {str(column).strip().lower(): str(column) for column in table.columns}

    for candidate in candidates:

        found = lookup.get(str(candidate).strip().lower())

        if found is not None:

            return found

    if required:

        raise ValueError(

            "Could not find any of the required columns: "

            + ", ".join(map(str, candidates))

        )

    return None

def p_text(p: float) -> str:

    if not np.isfinite(p):

        return "P=NA"

    if p < 1e-4:

        return f"P={p:.1e}"

    if p < 0.001:

        return f"P={p:.2e}"

    return f"P={p:.3f}"

def fdr_text(p: float) -> str:
    if not np.isfinite(p):
        return "FDR=NA"
    if p < 1e-4:
        return f"FDR={p:.1e}"
    if p < 0.001:
        return f"FDR={p:.2e}"
    return f"FDR={p:.3f}"

def plain_tick(value: float, _position: int) -> str:

    if not np.isfinite(value):

        return ""

    if abs(value) < 1e-14:

        return "0"

    magnitude = abs(value)

    if magnitude >= 1000 or magnitude < 0.001:

        return f"{value:.1e}"

    if magnitude >= 100:

        return f"{value:.0f}"

    if magnitude >= 10:

        return f"{value:.1f}"

    if magnitude >= 1:

        return f"{value:.2f}"

    return f"{value:.2f}"

def canonical_edge(source: str, target: str) -> tuple[str, str]:

    source = str(source)

    target = str(target)

    return (source, target) if source <= target else (target, source)

def normalise_response_label(value: object) -> str | None:

    """Map saved network-input response labels to the fixed Delta2/30/60 order."""

    key = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    lookup = {

        "delta2": "Delta2",

        "2": "Delta2",

        "delta30": "Delta30",

        "30": "Delta30",

        "delta60": "Delta60",

        "60": "Delta60",

    }

    return lookup.get(key)

def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#70777E",
        "axes.labelcolor": "#20252A",
        "xtick.color": "#20252A",
        "ytick.color": "#20252A",
        "text.color": "#20252A",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def prepare_annotation() -> pd.DataFrame:

    ann = read_tsv(ANN_FILE).copy()

    required = {"node", "label", "moltype", "group"}

    missing = required - set(ann.columns)

    if missing:

        raise ValueError(f"Annotation file is missing columns: {sorted(missing)}")

    for column in ["node", "label", "moltype", "group"]:

        ann[column] = ann[column].astype(str).str.strip()

    if ann["node"].duplicated().any():

        duplicated = ann.loc[ann["node"].duplicated(keep=False), "node"].unique()

        raise ValueError(

            "Annotation contains duplicate node IDs; first examples: "

            f"{duplicated[:10].tolist()}"

        )

    unknown_groups = sorted(set(ann["group"]) - set(GROUP_ORDER))

    if unknown_groups:

        raise ValueError(

            "Annotation has unexpected response-class labels: "

            f"{unknown_groups}. Expected: {GROUP_ORDER}"

        )

    return ann

def prepare_all_sign_edges(path: Path, known_nodes: set[str]) -> pd.DataFrame:
    """Read all existing significant edges, retaining positive and negative edges."""
    edges = read_tsv(path).copy()
    required = {"source", "target", "rho"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    edges["source"] = edges["source"].astype(str).str.strip()
    edges["target"] = edges["target"].astype(str).str.strip()
    edges["rho"] = pd.to_numeric(edges["rho"], errors="coerce")
    edges = edges.loc[np.isfinite(edges["rho"]) & edges["source"].ne(edges["target"])].copy()
    edges["abs_rho"] = edges["rho"].abs()
    edges["edge_key"] = [canonical_edge(a, b) for a, b in zip(edges["source"], edges["target"])]
    edges = edges.sort_values("abs_rho", ascending=False).drop_duplicates("edge_key",
        keep="first").reset_index(drop=True)
    edge_nodes = set(edges["source"]) | set(edges["target"])
    unknown = sorted(edge_nodes - known_nodes)
    if unknown:
        raise RuntimeError(f"{path} contains {len(unknown)} nodes absent from annotation; first: {unknown[:10]}")
    return edges

def prepare_positive_edges(path: Path, known_nodes: set[str]) -> pd.DataFrame:

    """Read existing significant edges and retain positive edges only."""

    edges = read_tsv(path).copy()

    required = {"source", "target", "rho"}

    missing = required - set(edges.columns)

    if missing:

        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    edges["source"] = edges["source"].astype(str).str.strip()

    edges["target"] = edges["target"].astype(str).str.strip()

    edges["rho"] = pd.to_numeric(edges["rho"], errors="coerce")

    edges = edges.loc[

        np.isfinite(edges["rho"])

        & edges["rho"].gt(0)

        & edges["source"].ne(edges["target"])

    ].copy()

    edges["strength"] = edges["rho"].astype(float)

    edges["edge_key"] = [

        canonical_edge(source, target)

        for source, target in zip(edges["source"], edges["target"])

    ]

    edges = (

        edges.sort_values("rho", ascending=False)

        .drop_duplicates("edge_key", keep="first")

        .reset_index(drop=True)

    )

    edge_nodes = set(edges["source"]) | set(edges["target"])

    unknown = sorted(edge_nodes - known_nodes)

    if unknown:

        raise RuntimeError(

            f"{path} contains {len(unknown)} nodes absent from the annotation; "

            f"first examples: {unknown[:10]}"

        )

    return edges

def load_module_nodes(path: Path, expected_network: str) -> set[str]:

    modules = read_tsv(path).copy()

    required = {"node", "module"}

    missing = required - set(modules.columns)

    if missing:

        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    modules["node"] = modules["node"].astype(str).str.strip()

    modules["module"] = modules["module"].astype(str).str.strip()

    valid = ~modules["module"].str.lower().isin(

        {"", "nan", "na", "none", "unassigned"}

    )

    nodes = set(modules.loc[valid, "node"])

    print(f"[{expected_network}] retained-module nodes read: {len(nodes):,}")

    return nodes

def read_rho_matrix(path: Path) -> pd.DataFrame:

    rho = read_tsv(path, index_col=0).copy()

    rho.columns = rho.columns.astype(str).str.strip()

    if rho.index.has_duplicates or rho.columns.has_duplicates:

        raise ValueError(f"Rho matrix has duplicate node IDs: {path}")

    return rho

# 2. Acute-response magnitude metrics

def read_mean_delta_response(path: Path, nodes: list[str], network: str) -> np.ndarray:

    """Return the mean Delta2/Delta30/Delta60 trajectory for every node.

    The saved network input has one row per subject x response time.  Averaging

    across subjects first gives the same mean baseline-centred trajectory used

    for the molecular-response comparison, while preserving the exact subject

    set used to construct the W0/W8 network.

    """

    table = read_tsv(path).copy()

    time_column = resolve_column(

        table,

        ["response_time", "response time", "time", "Time"],

    )

    missing_nodes = [node for node in nodes if node not in table.columns]

    if missing_nodes:

        raise RuntimeError(

            f"{network} network-input table is missing {len(missing_nodes)} nodes; "

            f"first examples: {missing_nodes[:10]}"

        )

    values = table[nodes].apply(pd.to_numeric, errors="coerce")

    if values.isna().to_numpy().any():

        raise RuntimeError(

            f"{network} network-input table contains missing/non-numeric molecular values."

        )

    labels = table[time_column].map(normalise_response_label)

    unknown = sorted(set(table.loc[labels.isna(), time_column].astype(str)))

    if unknown:

        raise ValueError(

            f"{network} network-input table has unrecognised response-time labels: "

            f"{unknown[:10]}"

        )

    mean_response = values.groupby(labels, sort=False).mean().reindex(RESPONSE_ORDER)

    if mean_response.isna().to_numpy().any():

        missing_labels = mean_response.index[mean_response.isna().all(axis=1)].tolist()

        raise RuntimeError(

            f"{network} has no valid rows for required response times: {missing_labels}"

        )

    return mean_response.to_numpy(dtype=float)

def calculate_response_metrics(nodes: list[str]) -> pd.DataFrame:

    means_w0 = read_mean_delta_response(NETWORK_INPUT_FILES["W0"], nodes, "W0")

    means_w8 = read_mean_delta_response(NETWORK_INPUT_FILES["W8"], nodes, "W8")

    # Add the baseline-centred 0-min point before taking the 0/2/30/60-min

    # absolute AUC or the 2/30/60-min absolute peak.

    trajectory_w0 = np.vstack([np.zeros((1, len(nodes))), means_w0])

    trajectory_w8 = np.vstack([np.zeros((1, len(nodes))), means_w8])

    abs_w0 = np.abs(trajectory_w0)

    abs_w8 = np.abs(trajectory_w8)

    intervals = np.diff(ACUTE_TIME_MIN)[:, None]

    auc_w0 = np.sum(0.5 * (abs_w0[:-1] + abs_w0[1:]) * intervals, axis=0)

    auc_w8 = np.sum(0.5 * (abs_w8[:-1] + abs_w8[1:]) * intervals, axis=0)

    peak_w0 = np.max(abs_w0[1:], axis=0)

    peak_w8 = np.max(abs_w8[1:], axis=0)

    return pd.DataFrame({

        "node": nodes,

        "abs_auc_W0": auc_w0,

        "abs_auc_W8": auc_w8,

        "delta_abs_auc": auc_w8 - auc_w0,
        "abs_delta_abs_auc": np.abs(auc_w8 - auc_w0),

        "abs_peak_W0": peak_w0,

        "abs_peak_W8": peak_w8,

        "delta_abs_peak": peak_w8 - peak_w0,
        "abs_delta_abs_peak": np.abs(peak_w8 - peak_w0),

    })

# 3. Fixed-network topology and rewiring metrics

def build_graph(nodes: list[str], positive_edges: pd.DataFrame) -> nx.Graph:

    graph = nx.Graph()

    graph.add_nodes_from(nodes)

    for row in positive_edges.itertuples(index=False):

        strength = float(row.strength)

        graph.add_edge(

            str(row.source),

            str(row.target),

            rho=float(row.rho),

            strength=strength,

            distance=1.0 / max(strength, 1e-12),

        )

    return graph

def degree_map(graph: nx.Graph) -> dict[str, float]:

    return {node: float(graph.degree(node)) for node in graph.nodes()}

def strength_map(graph: nx.Graph) -> dict[str, float]:

    return {node: float(value) for node, value in graph.degree(weight="strength")}

def closeness_map(graph: nx.Graph) -> dict[str, float]:

    if CLOSENESS_USE_WEIGHTED_DISTANCE:

        return nx.closeness_centrality(graph, distance="distance", wf_improved=True)

    return nx.closeness_centrality(graph, wf_improved=True)

def global_principal_eigenvector(

    graph: nx.Graph,

    weight: str | None,

) -> dict[str, float]:

    """Return one globally normalized principal eigenvector.

    This is used only if NetworkX power iteration fails.  Individual connected

    components must not be normalized separately: that would make a small

    disconnected component artificially comparable with the main network.

    The global spectral solution keeps nodes outside the dominant spectral

    component at zero, which is the coherent whole-network definition.

    """

    nodelist = list(graph.nodes())

    if not nodelist:

        return {}

    if len(nodelist) == 1 or graph.number_of_edges() == 0:

        return {node: 0.0 for node in nodelist}

    try:

        adjacency = nx.to_scipy_sparse_array(

            graph,

            nodelist=nodelist,

            weight=weight,

            dtype=float,

            format="csr",

        )

    except AttributeError:

        # Compatibility with NetworkX versions that predate sparse arrays.

        adjacency = nx.to_scipy_sparse_matrix(

            graph,

            nodelist=nodelist,

            weight=weight,

            dtype=float,

            format="csr",

        )

    try:

        _, vectors = eigsh(

            adjacency,

            k=1,

            which="LA",

            tol=1e-10,

            maxiter=max(10000, 50 * len(nodelist)),

        )

    except Exception as exc:

        raise RuntimeError(

            "Global eigenvector centrality did not converge; component-wise "

            "normalization was not applied."

        ) from exc

    values = np.abs(np.asarray(vectors[:, 0], dtype=float))

    norm = float(np.linalg.norm(values))

    if not np.isfinite(norm) or norm <= 0:

        raise RuntimeError("Global eigenvector centrality has an invalid norm.")

    values /= norm

    return {node: float(value) for node, value in zip(nodelist, values)}

def eigenvector_map(graph: nx.Graph) -> dict[str, float]:

    """Whole-network weighted eigenvector centrality with zero isolates."""

    weight = "strength" if EIGENVECTOR_USE_EDGE_STRENGTH else None

    result = {node: 0.0 for node in graph.nodes()}

    active_nodes = [node for node, degree in graph.degree() if degree > 0]

    if not active_nodes:

        return result

    # Removing isolates guarantees their intended value of zero without

    # changing eigenvector values for the active positive-edge network.

    active_graph = graph.subgraph(active_nodes).copy()

    try:

        values = nx.eigenvector_centrality(

            active_graph,

            max_iter=10000,

            tol=1e-10,

            weight=weight,

        )

    except Exception:

        values = global_principal_eigenvector(active_graph, weight=weight)

    result.update({node: float(values.get(node, 0.0)) for node in active_nodes})

    return result

def neighbor_sets(graph: nx.Graph) -> dict[str, set[str]]:

    return {node: set(graph.neighbors(node)) for node in graph.nodes()}

def neighbor_sets_from_edges(edges: pd.DataFrame, nodes: list[str]) -> dict[str, set[str]]:
    out = {node: set() for node in nodes}
    for row in edges.itertuples(index=False):
        a, b = str(row.source), str(row.target)
        out[a].add(b); out[b].add(a)
    return out

def signed_neighbor_states_from_edges(edges: pd.DataFrame, nodes: list[str]) -> dict[str, set[tuple[str, str]]]:
    out = {node: set() for node in nodes}
    for row in edges.itertuples(index=False):
        a, b, rho = str(row.source), str(row.target), float(row.rho)
        sign = "+" if rho > 0 else "-"
        out[a].add((b, sign)); out[b].add((a, sign))
    return out

def align_rho_matrix(rho: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:

    missing = [node for node in nodes if node not in rho.index or node not in rho.columns]

    if missing:

        raise RuntimeError(

            f"Rho matrix is missing {len(missing)} selected nodes; "

            f"first examples: {missing[:10]}"

        )

    aligned = rho.loc[nodes, nodes].apply(pd.to_numeric, errors="coerce")

    if aligned.isna().to_numpy().any():

        raise RuntimeError("Rho matrix contains missing/non-numeric values after alignment.")

    return aligned.astype(float)

def calculate_network_metrics(

    ann: pd.DataFrame,

    rho_w0: pd.DataFrame,

    rho_w8: pd.DataFrame,

    positive_edges_w0: pd.DataFrame,

    positive_edges_w8: pd.DataFrame,
    all_sign_edges_w0: pd.DataFrame,
    all_sign_edges_w8: pd.DataFrame,

) -> pd.DataFrame:

    nodes = ann["node"].astype(str).tolist()

    graph_w0 = build_graph(nodes, positive_edges_w0)

    graph_w8 = build_graph(nodes, positive_edges_w8)

    degree_w0 = degree_map(graph_w0)

    degree_w8 = degree_map(graph_w8)

    strength_w0 = strength_map(graph_w0)

    strength_w8 = strength_map(graph_w8)

    closeness_w0 = closeness_map(graph_w0)

    closeness_w8 = closeness_map(graph_w8)

    eigenvector_w0 = eigenvector_map(graph_w0)

    eigenvector_w8 = eigenvector_map(graph_w8)

    neighbors_w0 = neighbor_sets(graph_w0)

    neighbors_w8 = neighbor_sets(graph_w8)

    all_neighbors_w0 = neighbor_sets_from_edges(all_sign_edges_w0, nodes)
    all_neighbors_w8 = neighbor_sets_from_edges(all_sign_edges_w8, nodes)
    signed_states_w0 = signed_neighbor_states_from_edges(all_sign_edges_w0, nodes)
    signed_states_w8 = signed_neighbor_states_from_edges(all_sign_edges_w8, nodes)

    aligned_w0 = align_rho_matrix(rho_w0, nodes)

    aligned_w8 = align_rho_matrix(rho_w8, nodes)

    delta_rho = aligned_w8 - aligned_w0
    abs_delta_rho = delta_rho.abs()

    np.fill_diagonal(delta_rho.values, np.nan)
    np.fill_diagonal(abs_delta_rho.values, np.nan)

    rows = []

    for node in nodes:

        n_w0 = neighbors_w0[node]

        n_w8 = neighbors_w8[node]

        union = n_w0 | n_w8
        shared = n_w0 & n_w8
        turnover_positive = (
            np.nan if TURNOVER_EMPTY_AS_NAN else 0.0
        ) if len(union) == 0 else 1.0 - len(shared) / len(union)

        all_n0 = all_neighbors_w0[node]
        all_n8 = all_neighbors_w8[node]
        all_union = all_n0 | all_n8
        all_shared = all_n0 & all_n8
        turnover_all_sign = (
            np.nan if TURNOVER_EMPTY_AS_NAN else 0.0
        ) if len(all_union) == 0 else 1.0 - len(all_shared) / len(all_union)

        state0 = signed_states_w0[node]
        state8 = signed_states_w8[node]
        state_union = state0 | state8
        state_shared = state0 & state8
        turnover_sign_aware = (
            np.nan if TURNOVER_EMPTY_AS_NAN else 0.0
        ) if len(state_union) == 0 else 1.0 - len(state_shared) / len(state_union)

        rows.append({

            "node": node,

            "degree_W0": degree_w0[node],

            "degree_W8": degree_w8[node],

            "delta_degree": degree_w8[node] - degree_w0[node],
            "abs_delta_degree": abs(degree_w8[node] - degree_w0[node]),

            "strength_W0": strength_w0[node],

            "strength_W8": strength_w8[node],

            "delta_strength": strength_w8[node] - strength_w0[node],
            "abs_delta_strength": abs(strength_w8[node] - strength_w0[node]),

            "closeness_W0": closeness_w0[node],

            "closeness_W8": closeness_w8[node],

            "delta_closeness": closeness_w8[node] - closeness_w0[node],
            "abs_delta_closeness": abs(closeness_w8[node] - closeness_w0[node]),

            "eigenvector_W0": eigenvector_w0[node],

            "eigenvector_W8": eigenvector_w8[node],

            "delta_eigenvector": eigenvector_w8[node] - eigenvector_w0[node],
            "abs_delta_eigenvector": abs(eigenvector_w8[node] - eigenvector_w0[node]),

            "mean_delta_rho": float(delta_rho.loc[node].mean(skipna=True)),
            "mean_abs_delta_rho": float(abs_delta_rho.loc[node].mean(skipna=True)),

            "positive_neighbors_W0": len(n_w0),

            "positive_neighbors_W8": len(n_w8),

            "shared_positive_neighbors": len(shared),

            "union_positive_neighbors": len(union),

            "local_edge_turnover_positive": turnover_positive,
            "all_sign_neighbors_W0": len(all_n0),
            "all_sign_neighbors_W8": len(all_n8),
            "shared_all_sign_neighbors": len(all_shared),
            "union_all_sign_neighbors": len(all_union),
            "local_edge_turnover_all_sign": turnover_all_sign,
            "local_edge_turnover_sign_aware": turnover_sign_aware,

        })

    metrics = pd.DataFrame(rows).merge(

        ann[["node", "label", "moltype", "group"]],

        on="node",

        how="left",

        validate="one_to_one",

    )

    return metrics

def choose_analysis_nodes(

    ann: pd.DataFrame,

    module_nodes_w0: set[str],

    module_nodes_w8: set[str],

    positive_edges_w0: pd.DataFrame,

    positive_edges_w8: pd.DataFrame,

) -> tuple[set[str], str]:

    all_selected = set(ann["node"])

    active_w0 = set(positive_edges_w0["source"]) | set(positive_edges_w0["target"])

    active_w8 = set(positive_edges_w8["source"]) | set(positive_edges_w8["target"])

    if ANALYSIS_NODE_SCOPE == "all_selected":

        nodes = all_selected

        description = "all selected molecules"

    elif ANALYSIS_NODE_SCOPE == "module_intersection":

        nodes = module_nodes_w0 & module_nodes_w8

        description = "retained modules in BOTH W0 and W8"

    elif ANALYSIS_NODE_SCOPE == "module_union":

        nodes = module_nodes_w0 | module_nodes_w8

        description = "a retained module in EITHER W0 or W8"

    elif ANALYSIS_NODE_SCOPE == "active_in_either":

        nodes = active_w0 | active_w8

        description = "at least one positive significant edge in EITHER W0 or W8"

    else:

        raise ValueError(

            "ANALYSIS_NODE_SCOPE must be one of: all_selected, "

            "module_intersection, module_union, active_in_either"

        )

    return nodes & all_selected, description

# 4. Statistics

def metric_stats(metrics: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    valid = metrics[["group", metric]].dropna().copy()

    valid = valid.loc[valid["group"].isin(GROUP_ORDER)].copy()

    arrays = [

        valid.loc[valid["group"] == group, metric].to_numpy(float)

        for group in GROUP_ORDER

    ]

    if all(len(values) > 0 for values in arrays):

        try:

            kw_stat, kw_p = kruskal(*arrays)

        except ValueError:

            kw_stat, kw_p = 0.0, 1.0

    else:

        kw_stat, kw_p = np.nan, np.nan

    pairs = []

    for i, group1 in enumerate(GROUP_ORDER):

        for group2 in GROUP_ORDER[i + 1:]:

            x1 = valid.loc[valid["group"] == group1, metric].to_numpy(float)

            x2 = valid.loc[valid["group"] == group2, metric].to_numpy(float)

            if len(x1) and len(x2):

                statistic, p_raw = mannwhitneyu(x1, x2, alternative="two-sided")

            else:

                statistic, p_raw = np.nan, np.nan

            pairs.append({

                "metric": metric,

                "group1": group1,

                "group2": group2,

                "n1": len(x1),

                "n2": len(x2),

                "U": statistic,

                "p_raw": p_raw,

                "kw_stat": kw_stat,

                "kw_p": kw_p,

            })

    pair_df = pd.DataFrame(pairs)

    pair_df["p_bh_within_metric"] = np.nan

    tested = pair_df["p_raw"].notna()

    if tested.any():

        pair_df.loc[tested, "p_bh_within_metric"] = multipletests(

            pair_df.loc[tested, "p_raw"], method="fdr_bh"

        )[1]

    summary_rows = []

    for group in GROUP_ORDER:

        values = valid.loc[valid["group"] == group, metric].to_numpy(float)

        if len(values):

            summary_rows.append({

                "metric": metric,

                "group": group,

                "n": len(values),

                "mean": float(np.mean(values)),

                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,

                "median": float(np.median(values)),

                "q1": float(np.quantile(values, 0.25)),

                "q3": float(np.quantile(values, 0.75)),

                "kw_stat": kw_stat,

                "kw_p": kw_p,

            })

    return pair_df, pd.DataFrame(summary_rows)

def prepare_figure_stats(
    metrics: pd.DataFrame,
    panels,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BH-FDR over all displayed panels and all three between-class contrasts."""
    pairwise_tables = []
    group_tables = []
    for metric, _label, _include_zero in panels:
        pair_df, group_df = metric_stats(metrics, metric)
        pairwise_tables.append(pair_df)
        group_tables.append(group_df)

    pairwise = pd.concat(pairwise_tables, ignore_index=True)
    group_summary = pd.concat(group_tables, ignore_index=True)
    pairwise["p_bh_figure"] = np.nan
    valid = pairwise["p_raw"].notna()
    if valid.any():
        pairwise.loc[valid, "p_bh_figure"] = multipletests(
            pairwise.loc[valid, "p_raw"].to_numpy(dtype=float), method="fdr_bh"
        )[1]
    pairwise["fdr_family"] = "five_plotted_metrics_x_three_pairs"
    pairwise["significant_fdr05"] = pairwise["p_bh_figure"].lt(SIGNIF_THRESHOLD)
    return pairwise, group_summary

# 5. Plot helpers

def mean_ci95(values: np.ndarray) -> tuple[float, float, float]:

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:

        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))

    if len(values) == 1:

        return mean, mean, mean

    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))

    return mean, mean - 1.96 * se, mean + 1.96 * se

def padded_ylim(values: np.ndarray, include_zero: bool = False) -> tuple[float, float]:

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:

        return -1.0, 1.0

    lower, upper = float(np.min(values)), float(np.max(values))

    if include_zero:

        lower = min(lower, 0.0)

        upper = max(upper, 0.0)

    span = upper - lower

    if span == 0:

        span = max(abs(upper), 1.0)

    # Balanced vertical spacing: provide a small margin below the violins,
    # including the two non-negative panels, without the excessive whitespace
    # of the original 0.76 setting. These are AXIS margins only, not data edits.
    if not include_zero and lower >= 0.0:
        y_bottom = -0.045 * span
    else:
        y_bottom = lower - 0.12 * span
    y_top = upper + 0.30 * span   # leave room for significance brackets
    if not 0.0 < VIOLIN_VERTICAL_FILL <= 1.0:
        raise ValueError("VIOLIN_VERTICAL_FILL must be within (0, 1].")
    center = (y_bottom + y_top) / 2.0
    half_range = (y_top - y_bottom) / (2.0 * VIOLIN_VERTICAL_FILL)
    return center - half_range, center + half_range

def p_text(p):
    return fdr_text(p)

def compute_panel_ylim(values, pair_df, include_zero=False):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (-1, 1)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    span = hi - lo
    if span == 0:
        span = max(abs(hi), 1.0)
    sig_n = int(pair_df[pair_df["p_bh_figure"].notna() & (pair_df["p_bh_figure"] < SIGNIF_THRESHOLD)].shape[0])
    # More top whitespace than before so 3 brackets breathe.
    bottom_pad = 0.08 * span
    top_pad = 0.23 * span + max(sig_n - 1, 0) * 0.10 * span
    return lo - bottom_pad, hi + top_pad

def add_pairwise_brackets(ax, pair_df, base_y, step_y, p_col):
    sig = pair_df.loc[
        pair_df[p_col].notna() & (pair_df[p_col] < SIGNIF_THRESHOLD)
    ].copy()
    if sig.empty:
        return base_y
    pos = {g: i + 1 for i, g in enumerate(GROUP_ORDER)}
    sig["span"] = sig.apply(lambda r: pos[r.group2] - pos[r.group1], axis=1)
    sig = sig.sort_values(["span", p_col], ascending=[True, True])
    y = base_y
    for row in sig.itertuples(index=False):
        x1 = pos[row.group1]
        x2 = pos[row.group2]
        p = getattr(row, p_col)
        h = step_y * 0.17
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + h, y + h, y],
            color="#4B5157",
            lw=0.78,
            clip_on=False,
        )
        # In five narrow panels, nudge adjacent-pair labels inward to keep
        # long scientific-notation FDR strings clear of neighboring y-axes.
        label_x = (x1 + x2) / 2
        if (x1, x2) == (1, 2):
            label_x += 0.13
        elif (x1, x2) == (2, 3):
            label_x -= 0.13
        ax.text(
            label_x,
            y + h + step_y * 0.05,
            p_text(p),
            ha="center",
            va="bottom",
            fontsize=6.6,
        )
        y += step_y
    return y

def violin_panel(
    ax, metrics, metric, title, include_zero=False, ylim=None, *, pair_df, summary_df
):
    data = []
    for group in GROUP_ORDER:
        vals = metrics.loc[
            metrics["group"] == group, metric
        ].dropna().to_numpy(float)
        data.append(vals)

    parts = ax.violinplot(
        data,
        positions=[1, 2, 3],
        widths=0.79,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.35,
    )
    for body, group in zip(parts["bodies"], GROUP_ORDER):
        body.set_facecolor(GROUP_COLORS[group])
        body.set_edgecolor(GROUP_EDGE_COLORS[group])
        body.set_alpha(0.68)
        body.set_linewidth(0.85)
        body.set_zorder(2)

    # ONLY mean (white diamond) + its 95% CI. No median black line.
    for x, vals in zip([1, 2, 3], data):
        mean, lo, hi = mean_ci95(vals)
        if np.isfinite(mean):
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="D",
                markersize=3.2,
                markerfacecolor="white",
                markeredgecolor="#25292E",
                markeredgewidth=0.75,
                ecolor="#25292E",
                elinewidth=0.9,
                capsize=2.4,
                capthick=0.8,
                zorder=7,
            )

    vals_all = (
        np.concatenate([x for x in data if len(x)])
        if any(len(x) for x in data)
        else np.array([0.0])
    )
    if ylim is None:
        ylo, yhi = compute_panel_ylim(vals_all, pair_df, include_zero=include_zero)
    else:
        ylo, yhi = ylim
    ax.set_ylim(ylo, yhi)
    if include_zero and ylo < 0 < yhi:
        ax.axhline(0, color="#AEB5BB", lw=0.7, ls=(0, (3, 2)), zorder=0)
    finite_vals = vals_all[np.isfinite(vals_all)]
    data_hi = float(np.max(finite_vals)) if len(finite_vals) else 0.0
    if include_zero:
        data_hi = max(data_hi, 0.0)
    span_data = float(np.max(finite_vals) - np.min(finite_vals)) if len(finite_vals) else 1.0
    if span_data == 0:
        span_data = max(abs(data_hi), 1.0)
    bracket_base = data_hi + 0.10 * span_data
    bracket_step = 0.10 * span_data
    add_pairwise_brackets(ax, pair_df, bracket_base, bracket_step, "p_bh_figure")

    # ONLY class labels. No second-line median numbers below each violin.
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        [GROUP_SHORT[group] for group in GROUP_ORDER],
        fontsize=FONT_SIZE,
    )
    ax.set_title(title, fontsize=7.1, pad=7, fontweight="medium")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#70777E")
    ax.spines["bottom"].set_color("#70777E")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", width=0.7, length=2.4, pad=2, direction="out")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_tick))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.grid(False)
    return pair_df, summary_df

METRIC_PANELS = [
    ("abs_delta_abs_auc", "|ΔAbsAUC|", False),
    ("delta_degree", "ΔDegree", True),
    ("delta_closeness", "ΔCloseness", True),
    ("delta_eigenvector", "ΔEigenvector", True),
    ("local_edge_turnover_all_sign", "Edge turnover", False),
]

def draw_figure(metrics: pd.DataFrame, panels, out_pdf: Path, out_png: Path):
    """Match static five-panel layout: fixed PDF 154.5 x 74.0 mm."""
    pair_stats, group_summary = prepare_figure_stats(metrics, panels)
    set_plot_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    grid = fig.add_gridspec(
        1, len(panels),
        width_ratios=[1.0] * len(panels),
        left=0.062, right=0.986, bottom=0.205, top=0.865,
        wspace=0.44,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(panels))]
    for ax, (metric, title, include_zero) in zip(axes, panels):
        p = pair_stats.loc[pair_stats["metric"] == metric].copy()
        summary = group_summary.loc[group_summary["metric"] == metric].copy()
        violin_panel(
            ax, metrics, metric, title, include_zero=include_zero,
            pair_df=p, summary_df=summary,
        )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", markersize=5, linestyle="None",
               markerfacecolor=GROUP_COLORS[group],
               markeredgecolor=GROUP_EDGE_COLORS[group],
               markeredgewidth=0.6, label=GROUP_DISPLAY[group])
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.083),
        ncol=3, frameon=False, fontsize=FONT_SIZE,
        handletextpad=0.35, columnspacing=1.3,
    )
    fig.text(
        0.5, 0.038,
        "White diamond: mean ± 95% CI; brackets: BH-FDR (15 tests)",
        ha="center", va="center", fontsize=FONT_SIZE, color="#5D646B",
    )
    # Do not use bbox_inches='tight'; preserve EXACT page dimensions.
    fig.savefig(out_pdf, dpi=DPI, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    return pair_stats, group_summary

# 7. Run

def main() -> None:

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running script version: {SCRIPT_VERSION}")

    print("Dynamic input: W0=[B1-A, B2-A, B3-A]; W8=[F1-D, F2-D, F3-D]")

    ann = prepare_annotation()

    nodes = ann["node"].astype(str).tolist()

    known_nodes = set(nodes)
    all_sign_edges_w0 = prepare_all_sign_edges(EDGE_FILES["W0"], known_nodes)
    all_sign_edges_w8 = prepare_all_sign_edges(EDGE_FILES["W8"], known_nodes)
    positive_edges_w0 = prepare_positive_edges(EDGE_FILES["W0"], known_nodes)
    positive_edges_w8 = prepare_positive_edges(EDGE_FILES["W8"], known_nodes)

    rho_w0 = read_rho_matrix(RHO_FILES["W0"])

    rho_w8 = read_rho_matrix(RHO_FILES["W8"])

    response_metrics = calculate_response_metrics(nodes)

    response_metrics.to_csv(OUT_RESPONSE, sep="\t", index=False)

    # All graph metrics are calculated on the fixed complete W0/W8 networks;

    # the module/node filter is intentionally applied only after this step.

    metrics = calculate_network_metrics(

        ann, rho_w0, rho_w8, positive_edges_w0, positive_edges_w8,
        all_sign_edges_w0, all_sign_edges_w8,

    )

    metrics = metrics.merge(response_metrics, on="node", how="left", validate="one_to_one")

    module_nodes_w0 = load_module_nodes(MODULE_FILES["W0"], "W0")

    module_nodes_w8 = load_module_nodes(MODULE_FILES["W8"], "W8")

    analysis_nodes, scope_description = choose_analysis_nodes(

        ann, module_nodes_w0, module_nodes_w8, positive_edges_w0, positive_edges_w8,

    )

    analysis_ann = ann.loc[ann["node"].isin(analysis_nodes)].copy()

    if analysis_ann.empty:

        raise RuntimeError(

            "No nodes remain after ANALYSIS_NODE_SCOPE filtering; check the "

            "module assignments or choose a less restrictive scope."

        )

    analysis_ann["analysis_scope"] = ANALYSIS_NODE_SCOPE

    analysis_ann["analysis_scope_description"] = scope_description

    analysis_ann.to_csv(OUT_NODES, sep="\t", index=False)

    print(

        f"[Node filter] {scope_description}: retained {len(analysis_ann):,}/"

        f"{len(ann):,} selected molecules."

    )

    metrics = metrics.loc[metrics["node"].isin(analysis_nodes)].copy()

    metrics["analysis_scope"] = ANALYSIS_NODE_SCOPE

    metrics["analysis_scope_description"] = scope_description

    metrics.to_csv(OUT_METRICS, sep="\t", index=False)

    print(
        f"[Metric rows] retained {len(metrics):,} nodes for the "
        "three-class selected-five metric analysis."
    )

    pair_stats, group_summary = draw_figure(
        metrics, METRIC_PANELS, OUT_PDF, OUT_PNG,
    )
    pair_stats["analysis_scope"] = ANALYSIS_NODE_SCOPE
    pair_stats["analysis_scope_description"] = scope_description
    group_summary["analysis_scope"] = ANALYSIS_NODE_SCOPE
    group_summary["analysis_scope_description"] = scope_description
    pair_stats.to_csv(OUT_PAIRWISE, sep="\t", index=False)
    group_summary.to_csv(OUT_SUMMARY, sep="\t", index=False)

    # Only report the final PDF location on completion.
    print(str(OUT_PDF.resolve()))

if __name__ == "__main__":

    main()
