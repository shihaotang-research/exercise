# -*- coding: utf-8 -*-
"""Figure 4: resting-state molecular network remodeling."""

import os
import warnings
import sys

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.sparse.linalg import eigsh
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
SCRIPT_VERSION = "FINAL_v19_BRACKETS_SPACIOUS_TOP"

# 0. Parameters
DATA_DIR = os.environ.get("MULTIOMICS_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
NETWORK_DIR = os.path.join(DATA_DIR, "resting_network")
OUT_DIR = NETWORK_DIR
ANN_FILE = os.path.join(NETWORK_DIR, "00_selected_molecules_final_same_set.txt")
MEAN_FILE = os.path.join(DATA_DIR, "response_mean_profiles.tsv")
RHO_W0_FILE = os.path.join(NETWORK_DIR, "02_W0_Spearman_rho_matrix.txt")
RHO_W8_FILE = os.path.join(NETWORK_DIR, "02_W8_Spearman_rho_matrix.txt")
EDGE_W0_FILE = os.path.join(NETWORK_DIR, "02_W0_significant_edges_all_signs.txt")
EDGE_W8_FILE = os.path.join(NETWORK_DIR, "02_W8_significant_edges_all_signs.txt")
MODULE_W0_FILE = os.path.join(NETWORK_DIR, "03_W0_module_assignment.txt")
MODULE_W8_FILE = os.path.join(NETWORK_DIR, "03_W8_module_assignment.txt")

# Primary node population for the three-class comparison.
# all_common_network_nodes (DEFAULT): every common node in ANN_FILE.
# This is the appropriate primary scope for degree/closeness/eigenvector,
# mean |Delta rho|, and local edge turnover. Nodes without significant edges
# have zero centrality; turnover remains NaN when both neighbor sets are empty.
# retained_module_intersection / retained_module_union: sensitivity analyses.
ANALYSIS_NODE_SCOPE = "all_common_network_nodes"
VALID_ANALYSIS_NODE_SCOPES = {
    "all_common_network_nodes",
    "retained_module_intersection",
    "retained_module_union",
}

# CHANGED: plotting, legend, and pairwise comparison order.
GROUP_ORDER = ["Baseline-only", "Dual-response", "Trend-only"]
GROUP_DISPLAY = {
    "Baseline-only": "Baseline-only",
    "Trend-only": "Dynamic-only",
    "Dual-response": "Dual-response",
}
GROUP_COLORS = {
    "Baseline-only": "#4C95BC",
    "Trend-only": "#E57C5E",
    "Dual-response": "#55A96E",
}
GROUP_EDGE_COLORS = {
    "Baseline-only": "#2F6F8F",
    "Trend-only": "#A94F37",
    "Dual-response": "#347A49",
}

# Significant brackets: BH-FDR across all 15 pairwise comparisons.
SIGNIF_THRESHOLD = 0.05
CLOSENESS_USE_WEIGHTED_DISTANCE = True
EIGENVECTOR_USE_ABS_RHO = True
TURNOVER_EMPTY_AS_NAN = True

FONT_FAMILY = "Arial"
# Final exported PDF size: exactly 154.5 x 76 mm (within 155 mm limit).
# Do not use bbox_inches="tight": that can enlarge the exported page.
FIG_W_MM = 154.5
FIG_H_MM = 74.0
FIG_W = FIG_W_MM / 25.4
FIG_H = FIG_H_MM / 25.4
DPI = 600
FONT_SIZE = 7.0

# At this width the full three class names cannot fit under each violin.
# Use compact x-tick names, with full names in the bottom legend.
GROUP_SHORT = {
    "Baseline-only": "Base",
    "Dual-response": "Dual",
    "Trend-only": "Dyn",
}

# Default stdout output: final absolute PDF path only.
VERBOSE = False

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, file=sys.stderr, **kwargs)

# 1. Utilities
def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

def read_tsv(path, index_col=None):
    require_file(path)
    return pd.read_csv(path, sep="\t", index_col=index_col)

def p_text(p):
    """BH-FDR text (same statistic as the original code)."""
    if not np.isfinite(p):
        return "FDR=NA"
    if p < 1e-4:
        return f"FDR={p:.1e}"
    if p < 0.001:
        return f"FDR={p:.2e}"
    return f"FDR={p:.3f}"

def plain_tick(value, position):
    """Compact tick labels without Matplotlib offset text."""
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

# 2. Mean trajectory / ΔExp helpers
def normalize_colname(x):
    return "".join(ch.lower() for ch in str(x) if ch.isalnum())

def detect_mean_id_column(df):
    preferred = [
        "node", "molecule", "moleculeid", "protein", "proteinid",
        "metabolite", "metaboliteid", "feature", "featureid", "id",
    ]
    norm_map = {normalize_colname(c): c for c in df.columns}
    for key in preferred:
        if key in norm_map:
            return norm_map[key]
    return df.columns[0]

def find_mean_time_column(df, tp):
    target = normalize_colname(tp)
    candidates = {
        target,
        normalize_colname(f"mean_{tp}"),
        normalize_colname(f"mean{tp}"),
    }
    for c in df.columns:
        if normalize_colname(c) in candidates:
            return c
    for c in df.columns:
        n = normalize_colname(c)
        if n.endswith(target) and ("mean" in n or n == target):
            return c
    raise KeyError(
        f"Could not find mean trajectory column for {tp!r} in {MEAN_FILE}. "
        f"Available columns: {list(df.columns)}"
    )

def load_delta_exp_table():
    require_file(MEAN_FILE)
    mean_df = pd.read_csv(MEAN_FILE, sep="\t")
    id_col = detect_mean_id_column(mean_df)
    a_col = find_mean_time_column(mean_df, "A")
    d_col = find_mean_time_column(mean_df, "D")
    out = mean_df[[id_col, a_col, d_col]].copy()
    out.columns = ["node", "mean_A", "mean_D"]
    out["node"] = out["node"].astype(str).str.strip()
    out["mean_A"] = pd.to_numeric(out["mean_A"], errors="coerce")
    out["mean_D"] = pd.to_numeric(out["mean_D"], errors="coerce")
    dup = int(out["node"].duplicated().sum())
    if dup:
        log(f"[WARNING] {dup} duplicated molecule IDs in {MEAN_FILE}; keeping first.")
        out = out.drop_duplicates("node", keep="first")
    out["delta_exp"] = out["mean_D"] - out["mean_A"]
    out["abs_delta_exp"] = out["delta_exp"].abs()
    return out

# 3. Build graphs from significant edges
def build_graph(nodes, edges):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for row in edges.itertuples(index=False):
        rho = float(row.rho)
        g.add_edge(
            str(row.source),
            str(row.target),
            rho=rho,
            abs_rho=abs(rho),
            distance=1.0 / max(abs(rho), 1e-12),
        )
    return g

def degree_map(g):
    return {n: float(g.degree(n)) for n in g.nodes()}

def closeness_map(g):
    if CLOSENESS_USE_WEIGHTED_DISTANCE:
        return nx.closeness_centrality(g, distance="distance", wf_improved=True)
    return nx.closeness_centrality(g, wf_improved=True)

def global_principal_eigenvector(g, weight):
    """Globally normalized principal eigenvector, never per-component scaling."""
    nodelist = list(g.nodes())
    if len(nodelist) == 0:
        return {}
    if len(nodelist) == 1 or g.number_of_edges() == 0:
        return {node: 0.0 for node in nodelist}
    try:
        adjacency = nx.to_scipy_sparse_array(
            g,
            nodelist=nodelist,
            weight=weight,
            dtype=float,
            format="csr",
        )
    except AttributeError:
        adjacency = nx.to_scipy_sparse_matrix(
            g,
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
            "Global eigenvector centrality did not converge; no component-wise "
            "normalization was applied."
        ) from exc
    values = np.abs(np.asarray(vectors[:, 0], dtype=float))
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("Global eigenvector centrality has an invalid norm.")
    values = values / norm
    return {node: float(value) for node, value in zip(nodelist, values)}

def eigenvector_map(g):
    """Whole-network weighted eigenvector centrality with exact zero isolates."""
    weight = "abs_rho" if EIGENVECTOR_USE_ABS_RHO else None
    result = {node: 0.0 for node in g.nodes()}
    active_nodes = [node for node, degree in g.degree() if degree > 0]
    if not active_nodes:
        return result
    active_g = g.subgraph(active_nodes).copy()
    try:
        values = nx.eigenvector_centrality(
            active_g,
            max_iter=10000,
            tol=1e-10,
            weight=weight,
        )
    except Exception:
        values = global_principal_eigenvector(active_g, weight=weight)
    result.update({node: float(values.get(node, 0.0)) for node in active_nodes})
    return result

def neighbor_sets(g):
    return {n: set(g.neighbors(n)) for n in g.nodes()}

def signed_neighbor_states(edges, nodes):
    """Per-node (neighbor, sign) states from all significant edges."""
    out = {n: set() for n in nodes}
    for row in edges.itertuples(index=False):
        a, b, rho = str(row.source), str(row.target), float(row.rho)
        sign = "+" if rho > 0 else "-"
        out[a].add((b, sign))
        out[b].add((a, sign))
    return out

def retained_module_nodes(path, expected_network):
    tab = read_tsv(path)
    required = {"node", "module"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    tab["node"] = tab["node"].astype(str)
    module = tab["module"].astype(str).str.strip()
    valid = module.notna() & ~module.str.lower().isin(
        {"", "nan", "na", "none", "unassigned"}
    )
    nodes = set(tab.loc[valid, "node"])
    log(f"[{expected_network}] retained-module nodes read: {len(nodes):,}")
    return nodes

def select_analysis_nodes(ann):
    """Choose primary or optional sensitivity node scope without rebuilding graphs."""
    if ANALYSIS_NODE_SCOPE not in VALID_ANALYSIS_NODE_SCOPES:
        raise ValueError(
            "ANALYSIS_NODE_SCOPE must be one of "
            f"{sorted(VALID_ANALYSIS_NODE_SCOPES)}; got {ANALYSIS_NODE_SCOPE!r}."
        )
    all_nodes = set(ann["node"].astype(str))
    if ANALYSIS_NODE_SCOPE == "all_common_network_nodes":
        return all_nodes, "all common static-network nodes"
    w0_module_nodes = retained_module_nodes(MODULE_W0_FILE, "W0")
    w8_module_nodes = retained_module_nodes(MODULE_W8_FILE, "W8")
    if ANALYSIS_NODE_SCOPE == "retained_module_intersection":
        return (
            w0_module_nodes & w8_module_nodes,
            "retained-module intersection (W0 AND W8)",
        )
    return (
        w0_module_nodes | w8_module_nodes,
        "retained-module union (W0 OR W8)",
    )

# 4. Node-level remodeling metrics
def align_rho_matrix(rho, nodes):
    rho.index = rho.index.astype(str)
    rho.columns = rho.columns.astype(str)
    missing = [n for n in nodes if n not in rho.index or n not in rho.columns]
    if missing:
        raise RuntimeError(
            f"Rho matrix is missing {len(missing)} nodes; first few: {missing[:10]}"
        )
    return rho.loc[nodes, nodes].astype(float)

def calculate_metrics(ann, rho_w0, rho_w8, edge_w0, edge_w8, delta_exp_df):
    nodes = ann["node"].astype(str).tolist()
    g0 = build_graph(nodes, edge_w0)
    g8 = build_graph(nodes, edge_w8)
    deg0 = degree_map(g0)
    deg8 = degree_map(g8)
    clo0 = closeness_map(g0)
    clo8 = closeness_map(g8)
    eig0 = eigenvector_map(g0)
    eig8 = eigenvector_map(g8)
    nbr0 = neighbor_sets(g0)
    nbr8 = neighbor_sets(g8)
    signed_nbr0 = signed_neighbor_states(edge_w0, nodes)
    signed_nbr8 = signed_neighbor_states(edge_w8, nodes)
    rho_w0 = align_rho_matrix(rho_w0, nodes)
    rho_w8 = align_rho_matrix(rho_w8, nodes)
    signed_delta_rho = rho_w8 - rho_w0
    abs_delta_rho = signed_delta_rho.abs()
    np.fill_diagonal(signed_delta_rho.values, np.nan)
    np.fill_diagonal(abs_delta_rho.values, np.nan)

    rows = []
    for node in nodes:
        n0 = nbr0[node]
        n8 = nbr8[node]
        union = n0 | n8
        inter = n0 & n8
        active_w0 = len(n0) > 0
        active_w8 = len(n8) > 0
        if len(union) == 0:
            turnover = np.nan if TURNOVER_EMPTY_AS_NAN else 0.0
        else:
            turnover = 1.0 - (len(inter) / len(union))
        state0 = signed_nbr0[node]
        state8 = signed_nbr8[node]
        state_union = state0 | state8
        state_inter = state0 & state8
        if len(state_union) == 0:
            turnover_sign_aware = np.nan if TURNOVER_EMPTY_AS_NAN else 0.0
        else:
            turnover_sign_aware = 1.0 - (len(state_inter) / len(state_union))
        if active_w0 and active_w8:
            activity_state = "active_both"
        elif active_w0:
            activity_state = "active_W0_only"
        elif active_w8:
            activity_state = "active_W8_only"
        else:
            activity_state = "inactive_both"
        rows.append({
            "node": node,
            "degree_W0": deg0[node],
            "degree_W8": deg8[node],
            "delta_degree": deg8[node] - deg0[node],
            "abs_delta_degree": abs(deg8[node] - deg0[node]),
            "closeness_W0": clo0[node],
            "closeness_W8": clo8[node],
            "delta_closeness": clo8[node] - clo0[node],
            "abs_delta_closeness": abs(clo8[node] - clo0[node]),
            "eigenvector_W0": eig0[node],
            "eigenvector_W8": eig8[node],
            "delta_eigenvector": eig8[node] - eig0[node],
            "abs_delta_eigenvector": abs(eig8[node] - eig0[node]),
            "mean_delta_rho": float(signed_delta_rho.loc[node].mean(skipna=True)),
            "mean_abs_delta_rho": float(abs_delta_rho.loc[node].mean(skipna=True)),
            "neighbors_W0": len(n0),
            "neighbors_W8": len(n8),
            "active_W0": active_w0,
            "active_W8": active_w8,
            "network_activity_state": activity_state,
            "shared_neighbors": len(inter),
            "union_neighbors": len(union),
            "local_edge_turnover_all_sign": turnover,
            "local_edge_turnover_sign_aware": turnover_sign_aware,
        })
    metrics = pd.DataFrame(rows)
    metrics = metrics.merge(
        ann[["node", "label", "moltype", "group"]],
        on="node",
        how="left",
        validate="one_to_one",
    )
    metrics = metrics.merge(
        delta_exp_df[["node", "mean_A", "mean_D", "delta_exp", "abs_delta_exp"]],
        on="node",
        how="left",
        validate="one_to_one",
    )
    n_missing_exp = int(metrics["delta_exp"].isna().sum())
    if n_missing_exp:
        log(
            f"[WARNING] ΔExp unavailable for {n_missing_exp:,}/{len(metrics):,} "
            "network nodes; these nodes will be omitted only from the ΔExp "
            "panel/statistics."
        )
    return metrics, g0, g8

# 5. Statistics
def metric_stats(metrics, metric):
    """Pairwise Mann–Whitney U only; no omnibus test."""
    valid = metrics[["group", metric]].dropna().copy()
    valid = valid.loc[valid["group"].isin(GROUP_ORDER)]
    pairs, summaries = [], []
    for group in GROUP_ORDER:
        values = valid.loc[valid["group"] == group, metric].to_numpy(float)
        if len(values):
            summaries.append({
                "metric": metric,
                "group": group,
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "median": float(np.median(values)),  # TSV only; not plotted.
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
            })
    for i, group1 in enumerate(GROUP_ORDER):
        for group2 in GROUP_ORDER[i + 1:]:
            x = valid.loc[valid["group"] == group1, metric].to_numpy(float)
            y = valid.loc[valid["group"] == group2, metric].to_numpy(float)
            if len(x) and len(y):
                result = mannwhitneyu(x, y, alternative="two-sided")
                statistic, p = float(result.statistic), float(result.pvalue)
            else:
                statistic, p = np.nan, np.nan
            pairs.append({
                "metric": metric,
                "group1": group1,
                "group2": group2,
                "n1": len(x),
                "n2": len(y),
                "U": statistic,
                "p_raw": p,
            })
    pair_df = pd.DataFrame(pairs)
    pair_df["p_bh_within_metric"] = np.nan
    valid_p = pair_df["p_raw"].notna()
    if valid_p.any():
        pair_df.loc[valid_p, "p_bh_within_metric"] = multipletests(
            pair_df.loc[valid_p, "p_raw"].to_numpy(float), method="fdr_bh"
        )[1]
    return pair_df, pd.DataFrame(summaries)

def prepare_figure_stats(metrics, metric_names):
    """One BH-FDR family across all plotted metrics and three class comparisons."""
    pair_parts, summary_parts = [], []
    for name in metric_names:
        pair_df, summary_df = metric_stats(metrics, name)
        pair_parts.append(pair_df)
        summary_parts.append(summary_df)
    pairs = pd.concat(pair_parts, ignore_index=True)
    summaries = pd.concat(summary_parts, ignore_index=True)
    pairs["p_bh_figure"] = np.nan
    ok = pairs["p_raw"].notna()
    if ok.any():
        pairs.loc[ok, "p_bh_figure"] = multipletests(
            pairs.loc[ok, "p_raw"].to_numpy(float), method="fdr_bh"
        )[1]
    pairs["fdr_family"] = "all_plotted_metrics_x_three_pairs"
    pairs["significant_fdr05"] = pairs["p_bh_figure"].lt(SIGNIF_THRESHOLD)
    return pairs, summaries

# 6. Plot helpers — mean and 95% CI, NO median graphics
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

def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return mean, mean - 1.96 * se, mean + 1.96 * se

def padded_ylim(values, include_zero=False):
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
    return lo - 0.08 * span, hi + 0.14 * span

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

# 7. Global network metrics
def global_efficiency_unweighted(g):
    # NetworkX global_efficiency ignores weights; keep definition explicit.
    return nx.global_efficiency(g)

def global_summary(g0, g8, edge_w0, edge_w8):
    rows = []
    for name, g, e in [("W0", g0, edge_w0), ("W8", g8, edge_w8)]:
        active = [n for n, d in g.degree() if d > 0]
        active_g = g.subgraph(active).copy()
        rows.append({
            "network": name,
            "nodes_all": g.number_of_nodes(),
            "active_nodes": len(active),
            "edges": g.number_of_edges(),
            "density_all_nodes": nx.density(g),
            "mean_abs_rho": float(e["abs_rho"].mean()) if len(e) else np.nan,
            "global_efficiency_all_nodes": global_efficiency_unweighted(g),
            "global_efficiency_active_nodes": (
                global_efficiency_unweighted(active_g)
                if active_g.number_of_nodes() > 1
                else np.nan
            ),
        })
    return pd.DataFrame(rows)

def node_activity_summary(metrics):
    """Report per-class network participation for the all-node primary scope."""
    valid = metrics.loc[metrics["group"].isin(GROUP_ORDER)].copy()
    order = {
        "active_both": 0,
        "active_W0_only": 1,
        "active_W8_only": 2,
        "inactive_both": 3,
    }
    out = (
        valid.groupby(["group", "network_activity_state"], observed=True)
        .size()
        .reset_index(name="n")
    )
    out["state_order"] = out["network_activity_state"].map(order)
    out["group_order"] = out["group"].map(
        {g: i for i, g in enumerate(GROUP_ORDER)}
    )
    return out.sort_values(["group_order", "state_order"]).drop(
        columns=["group_order", "state_order"]
    )

# 8. Main figure
STATIC_PANELS = [
    ("abs_delta_exp", "|ΔExp|", False),
    ("delta_degree", "ΔDegree", True),
    ("delta_closeness", "ΔCloseness", True),
    ("delta_eigenvector", "ΔEigenvector", True),
    ("local_edge_turnover_all_sign", "Edge turnover", False),
]

def draw_figure(metrics):
    """Five single-row panels, 154.5 mm total width, fixed-page PDF."""
    pair_stats, group_summary = prepare_figure_stats(
        metrics, [item[0] for item in STATIC_PANELS]
    )
    set_plot_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    grid = fig.add_gridspec(
        1,
        len(STATIC_PANELS),
        width_ratios=[1.0] * len(STATIC_PANELS),
        left=0.062,
        right=0.986,
        bottom=0.205,
        top=0.865,
        wspace=0.44,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(STATIC_PANELS))]
    for ax, (metric, title, include_zero) in zip(axes, STATIC_PANELS):
        p = pair_stats.loc[pair_stats["metric"] == metric].copy()
        s = group_summary.loc[group_summary["metric"] == metric].copy()
        violin_panel(
            ax,
            metrics,
            metric,
            title,
            include_zero=include_zero,
            pair_df=p,
            summary_df=s,
        )

    # All legend text is >= 7 pt. Short tick labels correspond to this legend.
    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0], [0], marker="o", markersize=5, linestyle="None",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor=GROUP_EDGE_COLORS[group],
            markeredgewidth=0.6,
            label=GROUP_DISPLAY[group],
        )
        for group in GROUP_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.083),
        ncol=3,
        frameon=False,
        fontsize=FONT_SIZE,
        handletextpad=0.35,
        columnspacing=1.3,
    )
    fig.text(
        0.5,
        0.038,
        "White diamond: mean ± 95% CI; brackets: BH-FDR (15 tests)",
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        color="#5D646B",
    )
    out_pdf = os.path.join(
        OUT_DIR,
        f"06_W0_W8_network_remodeling_SIGNED_{ANALYSIS_NODE_SCOPE.upper()}.pdf",
    )
    out_png = os.path.join(
        OUT_DIR,
        f"06_W0_W8_network_remodeling_SIGNED_{ANALYSIS_NODE_SCOPE.upper()}.png",
    )
    # Crucial: no bbox_inches='tight'. PDF stays exactly 154.5 x 76 mm.
    fig.savefig(out_pdf, dpi=DPI, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    return pair_stats, group_summary, out_pdf, out_png

# 9. Run
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log(f"Running script version: {SCRIPT_VERSION}")
    ann = read_tsv(ANN_FILE)
    ann["node"] = ann["node"].astype(str)
    required_ann = {"node", "label", "moltype", "group"}
    missing = required_ann - set(ann.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {sorted(missing)}")

    n_before = len(ann)
    analysis_nodes, analysis_scope_text = select_analysis_nodes(ann)
    analysis_ann = ann.loc[ann["node"].isin(analysis_nodes)].copy()
    n_after = len(analysis_ann)
    if n_after == 0:
        raise RuntimeError(
            "No nodes remain after scope selection; check ANN_FILE and, for "
            "optional module scopes, the module-assignment files."
        )
    log(
        f"[Analysis scope] {analysis_scope_text}: retained {n_after:,}/{n_before:,} "
        f"nodes; removed {n_before - n_after:,}."
    )
    analysis_nodes_file = os.path.join(
        OUT_DIR,
        f"06_W0_W8_analysis_nodes_{ANALYSIS_NODE_SCOPE}.tsv",
    )
    analysis_ann.assign(
        analysis_node_scope=ANALYSIS_NODE_SCOPE,
        analysis_scope_description=analysis_scope_text,
    ).to_csv(analysis_nodes_file, sep="\t", index=False)

    edge_w0 = read_tsv(EDGE_W0_FILE)
    edge_w8 = read_tsv(EDGE_W8_FILE)
    for e in [edge_w0, edge_w8]:
        e["source"] = e["source"].astype(str)
        e["target"] = e["target"].astype(str)
        e["rho"] = pd.to_numeric(e["rho"], errors="coerce")
        if "abs_rho" not in e.columns:
            e["abs_rho"] = e["rho"].abs()

    rho_w0 = read_tsv(RHO_W0_FILE, index_col=0)
    rho_w8 = read_tsv(RHO_W8_FILE, index_col=0)
    log(f"Loading ΔExp from mean trajectory file: {MEAN_FILE}")
    delta_exp_df = load_delta_exp_table()
    metrics, g0, g8 = calculate_metrics(
        ann,
        rho_w0,
        rho_w8,
        edge_w0,
        edge_w8,
        delta_exp_df,
    )
    # Filter after calculating node metrics on unchanged original networks.
    metrics = metrics.loc[
        metrics["node"].isin(set(analysis_ann["node"]))
    ].copy()
    metrics.insert(0, "analysis_node_scope", ANALYSIS_NODE_SCOPE)
    metrics.insert(1, "analysis_scope_description", analysis_scope_text)
    log(
        f"[Metric rows] retained {len(metrics):,} nodes for statistics/plotting; "
        "network metrics themselves were calculated before scope selection."
    )
    metric_file = os.path.join(
        OUT_DIR,
        f"06_W0_W8_node_remodeling_metrics_{ANALYSIS_NODE_SCOPE}.tsv",
    )
    metrics.to_csv(metric_file, sep="\t", index=False)
    pair_stats, group_summary, signed_pdf, signed_png = draw_figure(metrics)
    for tab in [pair_stats, group_summary]:
        tab.insert(0, "analysis_node_scope", ANALYSIS_NODE_SCOPE)
        tab.insert(1, "analysis_scope_description", analysis_scope_text)

    pair_file = os.path.join(
        OUT_DIR,
        f"06_W0_W8_pairwise_stats_{ANALYSIS_NODE_SCOPE}.tsv",
    )
    summary_file = os.path.join(
        OUT_DIR,
        f"06_W0_W8_group_summary_{ANALYSIS_NODE_SCOPE}.tsv",
    )
    pair_stats.to_csv(pair_file, sep="\t", index=False)
    group_summary.to_csv(summary_file, sep="\t", index=False)
    global_df = global_summary(g0, g8, edge_w0, edge_w8)
    global_df.insert(0, "analysis_node_scope", ANALYSIS_NODE_SCOPE)
    global_df.insert(1, "analysis_scope_description", analysis_scope_text)
    global_file = os.path.join(
        OUT_DIR,
        f"06_W0_W8_global_network_metrics_{ANALYSIS_NODE_SCOPE}.tsv",
    )
    global_df.to_csv(global_file, sep="\t", index=False)
    activity_df = node_activity_summary(metrics)
    activity_df.insert(0, "analysis_node_scope", ANALYSIS_NODE_SCOPE)
    activity_df.insert(1, "analysis_scope_description", analysis_scope_text)
    activity_file = os.path.join(
        OUT_DIR,
        f"06_W0_W8_node_activity_summary_{ANALYSIS_NODE_SCOPE}.tsv",
    )
    activity_df.to_csv(activity_file, sep="\t", index=False)

    log("\nDone.")
    log(f"  Metrics : {metric_file}")
    log(f"  Nodes   : {analysis_nodes_file}")
    log(f"  Stats   : {pair_file}")
    log(f"  Summary : {summary_file}")
    log(f"  Global  : {global_file}")
    log(f"  Activity: {activity_file}")
    log(f"  Signed PDF : {signed_pdf}")
    log(f"  Signed PNG : {signed_png}")
    log("\nGlobal network metrics:")
    log(global_df.to_string(index=False))
    log("\nGroup summary (mean/SD; full summaries including median in TSV):")
    log(group_summary[["metric", "group", "n", "mean", "sd"]].to_string(index=False))

    print(os.path.abspath(signed_pdf))

if __name__ == "__main__":
    main()
