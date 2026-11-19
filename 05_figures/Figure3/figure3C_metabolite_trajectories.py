#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3C: metabolite class enrichment and trajectories."""
import os
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
# 1. FILE PATHS & STYLE  (change PROJECT_DIR when necessary)
PROJECT_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
ENRICH_FILE = PROJECT_DIR / 'metabolite_class_enrichment.tsv'
MET_FILE = PROJECT_DIR / 'metabolite_matrix.tsv'
OUTPUT_DIR = PROJECT_DIR / 'Figure3_metabolite_205x52'
PDF_FILENAME = 'Figure3_Metabolite_enrichment_PS980_205x52_FDR005_VISIBLE.pdf'
FIG_WIDTH_MM = 205.0
FIG_HEIGHT_MM = 52.0
SAVE_PNG = False
BACKGROUND_N = 1142
TOP_N = 3
MIN_COUNT = 3
MIN_ENRICHMENT = 1.0
FDR_CUTOFF = 0.05
SHOW_FDR_REFERENCE = True  # Draw FDR=0.05 line on top of bar fills
GROUP_ORDER = ['Baseline-only', 'Dual-response', 'Dynamic-only']
GROUP_COLORS = {
    'Baseline-only': '#4C95BC',
    'Dual-response': '#55A96E',
    'Dynamic-only': '#E57C5E',
}
W0_COLOR = '#2E8AC4'
W8_COLOR = '#E64B87'
NONSIG_EDGE = '#B5BBC0'
NONSIG_FILL = '#F0F1F2'
NONSIG_TEXT = '#888888'
SHOW_LEGEND = False  # Legend is already shown in the protein panel above
REPRESENTATIVES = {
    'Baseline-only': ('Decatrienoylcarnitine', 'met315'),
    'Dual-response': ('PC(37:3)', 'met3'),
    'Dynamic-only': ('PS(18:0/20:4)', 'met980'),
}
ALL_TP = ['A', 'B1', 'B2', 'B3', 'D', 'F1', 'F2', 'F3']
W0_TP, W8_TP = ALL_TP[:4], ALL_TP[4:]
TIME_LABELS = ['Base', '2', '30', '60']
# Identical geometry to the protein Figure 3C script.
GROUP_STARTS = [0.006, 0.338, 0.670]
BAR_OFFSET, BAR_W = 0.008, 0.170
TIME_OFFSET, TIME_W = 0.218, 0.106
PLOT_BOTTOM, PLOT_HEIGHT = 0.245, 0.550
HEADER_Y, HEADER_H = 0.890, 0.090
HEADER_FONT = 9.0
TERM_FONT = 7.25
TICK_FONT = 7.5
MET_FONT = 8.65
LABEL_FONT = 7.6
TERM_ALIASES = {
    'Phosphosphingolipids': 'Phosphosphingolipids',
    'Glycerophosphocholines': 'Glycerophosphocholines',
    'Glycerophosphoserines': 'Glycerophosphoserines',
    'Aromatic polyketides': 'Aromatic polyketides',
}
# 2. ENRICHMENT: USE EXACTLY THE ORIGINAL FDR; NO RECALCULATION
def group_label(value):
    key = re.sub(r'[\s_+\-]+', '', str(value).strip().lower())
    return {
        'baselineonly': 'Baseline-only',
        'dualresponse': 'Dual-response',
        'both': 'Dual-response',
        'trendonly': 'Dynamic-only',
        'dynamiconly': 'Dynamic-only',
    }.get(key, None)
def read_table(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f'Input file not found: {path}')
    df = pd.read_csv(path, sep='\t', low_memory=False, encoding='utf-8-sig')
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    return df
def load_top3():
    df = read_table(ENRICH_FILE)
    required = {'Group', 'main_class', 'n_background', 'count_in_group',
                'fold_enrichment', 'n_group', 'count_in_background', 'pvalue', 'fdr_bh'}
    if missing := required - set(df.columns):
        raise ValueError(f'Enrichment table missing columns: {sorted(missing)}')
    df['Group'] = df['Group'].map(group_label)
    if df['Group'].isna().any():
        raise ValueError('Unknown response group in enrichment table.')
    df['main_class'] = df['main_class'].astype(str).str.strip()
    for c in ('n_background', 'n_group', 'count_in_group', 'count_in_background', 'fold_enrichment', 'pvalue',
        'fdr_bh'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[['n_background', 'count_in_group', 'pvalue', 'fdr_bh']].isna().any().any():
        raise ValueError('Missing numeric enrichment information.')
    expected_fold = (df['count_in_group'] / df['n_group']) / (df['count_in_background'] / df['n_background'])
    if not np.all(np.isfinite(expected_fold)) or not np.allclose(
        df['fold_enrichment'].to_numpy(float), expected_fold.to_numpy(float), rtol=1e-6, atol=1e-8
    ):
        raise ValueError('Input fold_enrichment does not match standard (a/n)/(K/N); rerun class enrichment.')
    if not (df['n_background'] == BACKGROUND_N).all():
        raise ValueError('Wrong enrichment universe: expected all n_background=1142.')
    if not (df['fdr_bh'].between(0, 1) & df['pvalue'].between(0, 1)).all():
        raise ValueError('Invalid enrichment P/FDR (must be 0..1).')
    if df.duplicated(['Group', 'main_class']).any():
        raise ValueError('Duplicate Group/main_class enrichment rows.')
    selected = []
    for group in GROUP_ORDER:
        sub = df.loc[(df.Group == group) & (df.count_in_group >= MIN_COUNT)
                     & (df.fold_enrichment > MIN_ENRICHMENT)].copy()
        if sub.empty:
            raise ValueError(f'{group} has no eligible terms.')
        sub = sub.sort_values(['fdr_bh', 'pvalue', 'main_class'], kind='mergesort').head(TOP_N)
        sub['plot_order'] = np.arange(len(sub))
        sub['FDR_significant'] = sub['fdr_bh'] < FDR_CUTOFF
        sub['neglog10FDR'] = -np.log10(sub['fdr_bh'].clip(lower=np.finfo(float).tiny))
        sub['Display_name'] = sub['main_class'].map(lambda x: TERM_ALIASES.get(x, x))
        selected.append(sub)
    return pd.concat(selected, ignore_index=True)
# 3. METABOLITE GROUP MEANS
def normalized(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())
def find_col(df, candidates):
    colmap = {normalized(c): c for c in df.columns}
    return next((colmap[normalized(c)] for c in candidates if normalized(c) in colmap), None)
def infer_time(s):
    text = str(s)
    for tp in ('B1', 'B2', 'B3', 'F1', 'F2', 'F3', 'A', 'D'):
        if (re.search(rf'(^|[^A-Za-z0-9]){tp}($|[^A-Za-z0-9])', text, re.I)
                or re.search(rf'ae\d+[_\-.]?{tp}($|[^A-Za-z0-9])', text, re.I)):
            return tp
    return ''

def load_metabolite_means():
    df = read_table(MET_FILE)
    # Feature x group-mean matrix: A..F3 or mean_A..mean_F3.
    colmap = {tp: find_col(df, [tp, f'mean_{tp}']) for tp in ALL_TP}
    if all(v is not None for v in colmap.values()):
        id_col = find_col(df, ['Metabolite', 'MetaboliteID', 'Feature', 'MoleculeID', 'ID'])
        if id_col is None:
            id_col = df.columns[0]
        out = df[[id_col] + [colmap[x] for x in ALL_TP]].copy()
        out.columns = ['met_id'] + ALL_TP
        out['met_id'] = out['met_id'].astype(str).str.strip().str.lower()
        if out['met_id'].duplicated().any():
            raise ValueError('Duplicate metabolite IDs in mean matrix.')
        for tp in ALL_TP:
            out[tp] = pd.to_numeric(out[tp], errors='coerce')
        return out.set_index('met_id')[ALL_TP]
    # Sample x metabolite matrix.
    sample_col = find_col(df, ['Sample', 'SampleID', 'File', 'ID', 'Subject', 'subject'])
    time_col = find_col(df, ['Time', 'Timepoint', 'Visit'])
    if sample_col is None and time_col is None:
        raise ValueError('No sample/time column or eight group means found in metabolite matrix.')
    work = df.copy()
    if time_col is not None:
        work['_time'] = work[time_col].astype(str).str.strip().str.upper()
    else:
        work['_time'] = work[sample_col].apply(infer_time)
    work = work.loc[work['_time'].isin(ALL_TP)].copy()
    if missing_tp := set(ALL_TP) - set(work['_time']):
        raise ValueError(f'Expression matrix missing time points: {sorted(missing_tp)}')
    feature_cols = {rec[1]: find_col(work, [rec[1]]) for rec in REPRESENTATIVES.values()}
    if missing := [metid for metid, col in feature_cols.items() if col is None]:
        raise ValueError(f'Metabolites not found as sample-matrix columns: {missing}')
    for col in feature_cols.values():
        work[col] = pd.to_numeric(work[col], errors='coerce')
    mat = work.groupby('_time')[[feature_cols[m] for m in feature_cols]].mean().T
    mat.index = list(feature_cols.keys())
    return mat[ALL_TP]
def load_trajectories():
    means = load_metabolite_means()
    output = []
    for group in GROUP_ORDER:
        label, metid = REPRESENTATIVES[group]
        # Prefer exact IDs to avoid a similarly named feature being selected.
        metid_normalized = metid.lower()
        if metid_normalized not in means.index:
            raise ValueError(f'{label} ({metid}) missing in expression means.')
        vals = means.loc[metid_normalized, ALL_TP].to_numpy(float)
        if not np.isfinite(vals).all():
            raise ValueError(f'{label}: missing/nonfinite group mean.')
        sd = float(np.std(vals, ddof=0))
        if sd <= 0:
            raise ValueError(f'{label}: zero variance across eight time points.')
        zscore = (vals - float(np.mean(vals))) / sd
        output.extend({
            'Group': group, 'Metabolite': label, 'MetaboliteID': metid,
            'Timepoint': tp, 'Mean_log2': y, 'Zscore_8means': z,
        } for tp, y, z in zip(ALL_TP, vals, zscore))
    return pd.DataFrame(output)
# 4. GRAPHICS MATCHING PROTEIN Figure 3C
def set_style():
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans')
                 if f in installed), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': font, 'font.size': 7.5, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .65, 'axes.edgecolor': '#333333',
        'axes.labelcolor': '#161616', 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'text.color': '#161616', 'xtick.color': '#3B3B3B',
        'ytick.color': '#3B3B3B', 'axes.unicode_minus': False,
    })
def setup_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(.65)
    ax.tick_params(axis='both', labelsize=TICK_FONT, width=.65, length=2.3, pad=2)
    ax.grid(False)
def nice_limit(value):
    # Avoid an unnecessarily long x-axis (e.g. 10 instead of 6).
    value = max(float(value), .3)
    magnitude = 10 ** math.floor(math.log10(value))
    for candidate in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        upper = candidate * magnitude
        if upper >= value - 1e-12:
            return float(upper)
    return float(10 * magnitude)
def enrichment_plot(ax, top, group):
    sub = top.loc[top.Group == group].sort_values('plot_order')
    ypos = np.arange(len(sub))[::-1]
    vals = sub['neglog10FDR'].to_numpy(float)
    local_upper = nice_limit(max(vals) * 1.07)
    color = GROUP_COLORS[group]
    setup_axes(ax)
    for y, row in zip(ypos, sub.itertuples(index=False)):
        length = float(row.neglog10FDR)
        significant = bool(row.FDR_significant)
        if length > 0:
            ax.barh(y, length, height=.54,
                    facecolor=to_rgba(color, .18) if significant else NONSIG_FILL,
                    edgecolor=color if significant else NONSIG_EDGE,
                    linewidth=.78 if significant else .65, zorder=2)
        # FDR=1 => bar length is mathematically zero; keep only the term label.
        else:
            pass
        ax.text(.014, y, row.Display_name, fontsize=TERM_FONT,
                ha='left', va='center', color='#161616' if significant else NONSIG_TEXT,
                transform=ax.get_yaxis_transform(), zorder=5, clip_on=False)
    if SHOW_FDR_REFERENCE:
        ax.axvline(-math.log10(FDR_CUTOFF), color='#737B83',
                   linewidth=.95, linestyle=(0, (3, 2.5)), zorder=4)
    ax.set_xlim(0, local_upper)
    ax.set_ylim(-.55, TOP_N - .45)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.set_xlabel(r'$-\log_{10}(\mathrm{FDR})$', fontsize=LABEL_FONT, labelpad=1.3)
def trajectory_plot(ax, trajectories, group):
    sub = trajectories.loc[trajectories.Group == group].set_index('Timepoint')
    label, _ = REPRESENTATIVES[group]
    w0 = sub.loc[W0_TP, 'Zscore_8means'].to_numpy(float)
    w8 = sub.loc[W8_TP, 'Zscore_8means'].to_numpy(float)
    setup_axes(ax)
    x = np.arange(4)
    ax.axhline(0, color='#B9B9B9', linewidth=.60,
               linestyle=(0, (4, 3)), zorder=0)
    for values, color in ((w0, W0_COLOR), (w8, W8_COLOR)):
        ax.plot(x, values, '-', marker='o', markersize=3.05,
                linewidth=1.40, color=color, markeredgecolor='white',
                markeredgewidth=.40, zorder=4)
    ax.set_xlim(-.14, 3.14)
    ax.set_ylim(-2.55, 2.55)
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS, fontsize=TICK_FONT)
    ax.set_yticks([-2, 0, 2])
    ax.set_ylabel('Z-score', fontsize=7.2, labelpad=.6)
    ax.tick_params(axis='y', labelsize=TICK_FONT, pad=1)
    ax.set_xlabel('Time (min)', fontsize=LABEL_FONT, labelpad=1.3)
    ax.set_title(label, fontsize=MET_FONT, fontweight='medium', pad=4.5)
def draw(top, trajectories):
    set_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4),
                     facecolor='white')
    for i, group in enumerate(GROUP_ORDER):
        start = GROUP_STARTS[i]
        color = GROUP_COLORS[group]
        strip = FancyBboxPatch(
            (start + .002, HEADER_Y), .320, HEADER_H,
            transform=fig.transFigure, zorder=1,
            boxstyle='round,pad=0,rounding_size=0.012',
            facecolor=to_rgba(color, .13), edgecolor='none')
        fig.add_artist(strip)
        fig.text(start + .162, HEADER_Y + HEADER_H * .50, group,
                 fontsize=HEADER_FONT, ha='center', va='center',
                 color=color, fontweight='bold')
        a = fig.add_axes((start + BAR_OFFSET, PLOT_BOTTOM, BAR_W, PLOT_HEIGHT))
        b = fig.add_axes((start + TIME_OFFSET, PLOT_BOTTOM, TIME_W, PLOT_HEIGHT))
        enrichment_plot(a, top, group)
        trajectory_plot(b, trajectories, group)
        if i != 2:
            fig.add_artist(Line2D([start + .328, start + .328],
                                  [.125, .815], transform=fig.transFigure,
                                  color='#E2E2E2', linewidth=.52, zorder=0))
    # Same W0 / W8 legend placement as the protein C panel.
    legends = [Line2D([0], [0], color=c, marker='o', linewidth=1.35,
                      markersize=3.3, markerfacecolor=c, markeredgecolor='white',
                      markeredgewidth=.45, label=lab)
               for lab, c in (('W0', W0_COLOR), ('W8', W8_COLOR))]
    if SHOW_LEGEND:
        fig.legend(handles=legends, loc='lower center', bbox_to_anchor=(.5, .02),
                   ncol=2, frameon=False, fontsize=7.8, columnspacing=1.7,
                   handlelength=1.8, handletextpad=.4)
    return fig
# 5. SAVE
def main():
    top = load_top3()
    trajectories = load_trajectories()
    outdir = OUTPUT_DIR.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    top.to_csv(outdir / 'Figure3C_Top3_metabolite_classes_FDR.tsv',
               sep='\t', index=False)
    trajectories.to_csv(outdir / 'Figure3C_representative_metabolite_trajectories.tsv',
                        sep='\t', index=False)
    fig = draw(top, trajectories)
    pdf = outdir / PDF_FILENAME
    fig.savefig(pdf, facecolor='white')  # exact 205-mm width; no bbox_inches='tight'
    if SAVE_PNG:
        fig.savefig(pdf.with_suffix('.png'), facecolor='white', dpi=400)
    plt.close(fig)
    print(str(pdf))                 # print only the final PDF absolute path
    return pdf
if __name__ == '__main__':
    main()
