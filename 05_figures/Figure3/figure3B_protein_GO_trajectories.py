#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 3B: enriched protein GO terms and trajectories."""
import os
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
# 1. FILE PATHS AND USER OPTIONS
PROJECT_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
GO_FILE = PROJECT_DIR / 'protein_group_go_enrichment.tsv'
MEAN_FILE = PROJECT_DIR / 'response_mean_profiles.tsv'
OUTDIR = PROJECT_DIR / 'Figure3_protein_205x61'
OUT_PDF = 'Figure3_Protein_enrichment_ETFA_205x61_FDR005_VISIBLE.pdf'
FIG_WIDTH_MM = 205.0
FIG_HEIGHT_MM = 61.0           # Use 64-68 to gain more space in the composite
FDR_THRESHOLD = 0.05
SHOW_FDR_REFERENCE = True      # Force the FDR=0.05 line in all enrichment axes
SAVE_PNG = False
W0_COLOR = '#2E8AC4'
W8_COLOR = '#E64B87'
GROUP_COLORS = {
    'Baseline-only': '#4C95BC',
    'Dual-response': '#55A96E',
    'Dynamic-only': '#E57C5E',
}
GROUP_ORDER = ['Baseline-only', 'Dual-response', 'Dynamic-only']
NONSIG_FILL = '#F0F1F2'
NONSIG_EDGE = '#B5BBC0'
NONSIG_TEXT = '#777B80'

# One legend is sufficient after compositing protein and metabolite panels.
SHOW_LEGEND = True

# Full terms are used for matching source enrichment output.
# Display labels are shortened for the compressed A4-width composite figure.
TERMS = {
    'Baseline-only': [
        ('cellular respiration', 'Cellular respiration'),
        ('oxidative phosphorylation', 'Oxidative phosphorylation'),
        ('generation of precursor metabolites and energy', 'Energy generation'),
        ('mitochondrion organization', 'Mitochondrion organization'),
        ('regulation of membrane permeability', 'Membrane permeability'),
    ],
    'Dual-response': [
        ('wound healing', 'Wound healing'),
        ('blood coagulation', 'Blood coagulation'),
        ('protein maturation', 'Protein maturation'),
        ('leukocyte migration', 'Leukocyte migration'),
        ('platelet activation', 'Platelet activation'),
    ],
    'Dynamic-only': [
        ('complement activation', 'Complement activation'),
        ('complement activation, alternative pathway', 'Alternative complement'),
        ('humoral immune response', 'Humoral immune response'),
        ('acute-phase response', 'Acute-phase response'),
        ('acute inflammatory response', 'Acute inflammatory response'),
    ],
}
REPRESENTATIVES = {
    'Baseline-only': ('ETFA', 'P13804'),
    'Dual-response': ('PF4', 'P02776'),
    'Dynamic-only': ('C3', 'P01024'),
}
W0_COLS = ['A', 'B1', 'B2', 'B3']
W8_COLS = ['D', 'F1', 'F2', 'F3']
ALL_TIME_COLS = W0_COLS + W8_COLS
TIME_LABELS = ['Base', '2', '30', '60']
# The total figure has 3 groups at equal physical width.
# In each group: left enrichment chart, right time-course chart.
GROUP_STARTS = [0.006, 0.338, 0.670]
BAR_OFFSET, BAR_W = 0.008, 0.170
TIME_OFFSET, TIME_W = 0.218, 0.106
PLOT_BOTTOM, PLOT_HEIGHT = 0.245, 0.555
HEADER_Y, HEADER_H = 0.895, 0.085
# 7+ pt text at full 210-mm published width
HEADER_FONT = 9.0
GO_FONT = 7.25
TICK_FONT = 7.5
GENE_FONT = 8.65
LABEL_FONT = 7.6
# 2. ROBUST INPUTS
def read_table(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}')
    out = pd.read_csv(path, sep='\t', low_memory=False, encoding='utf-8-sig')
    if out.shape[1] < 2:
        out = pd.read_csv(path, sep=None, engine='python', low_memory=False)
    out.columns = [str(c).strip() for c in out.columns]
    return out
def norm_term(s):
    return ' '.join(str(s).strip().casefold().split())
def normalize_cluster(s):
    v = re.sub(r'[\s_-]+', '', str(s).casefold())
    mapping = {
        'baselineonly': 'Baseline-only',
        'dualresponse': 'Dual-response',
        'both': 'Dual-response',
        'trendonly': 'Dynamic-only',
        'dynamiconly': 'Dynamic-only',
    }
    return mapping.get(v, '')
def load_selected_pathways():
    df = read_table(GO_FILE)
    required = {'Cluster', 'Description', 'FDR'}
    if missing := required - set(df.columns):
        raise ValueError(f'GO file missing columns {sorted(missing)}; present={df.columns.tolist()}')
    df['group'] = df['Cluster'].map(normalize_cluster)
    df['term_key'] = df['Description'].map(norm_term)
    df['FDR'] = pd.to_numeric(df['FDR'], errors='coerce')
    if not df['FDR'].between(0, 1, inclusive='both').all():
        raise ValueError('GO FDR column contains NA or values outside [0, 1].')
    if 'pvalue' in df:
        df['pvalue'] = pd.to_numeric(df['pvalue'], errors='coerce')
    else:
        df['pvalue'] = np.nan
    chosen = []
    for group in GROUP_ORDER:
        for j, (term, label) in enumerate(TERMS[group]):
            hits = df.loc[(df['group'] == group) & (df['term_key'] == norm_term(term))]
            if hits.empty:
                raise ValueError(f'GO term not found for {group}: {term}\nCheck GO_FILE and Cluster/Description.')
            # If the same GO text occurs more than once, prefer lowest FDR.
            row = hits.sort_values(['FDR', 'pvalue'], na_position='last').iloc[0]
            fdr = float(row['FDR'])
            chosen.append({
                'Group': group,
                'GO_term': term,
                'Display_name': label,
                'plot_order': j,
                'pvalue': float(row['pvalue']) if pd.notna(row['pvalue']) else np.nan,
                'FDR': fdr,
                'neglog10FDR': -math.log10(max(fdr, np.finfo(float).tiny)),
                'FDR_significant': bool(fdr < FDR_THRESHOLD),
            })
    return pd.DataFrame(chosen)
def clean_protein_id(s):
    s = str(s).strip().upper()
    s = re.sub(r'^(?:PROTEIN__|PROTEIN_|PROTEIN:)', '', s)
    return s
def load_time_means():
    df = read_table(MEAN_FILE)
    normalized = {re.sub(r'[^a-z0-9]', '', str(c).casefold()): c for c in df.columns}
    id_col = next((normalized[x] for x in (
        'proteinid', 'protein', 'uniprot', 'moleculeid', 'id', 'featureid')
                   if x in normalized), df.columns[0])
    colmap = {}
    for tp in ALL_TIME_COLS:
        for key in (tp.casefold(), ('mean_' + tp).replace('_', '').casefold()):
            if key in normalized:
                colmap[tp] = normalized[key]
                break
        if tp not in colmap:
            raise ValueError(f'Mean matrix missing {tp} / mean_{tp}: {MEAN_FILE}')
    selected = df[[id_col] + [colmap[t] for t in ALL_TIME_COLS]].copy()
    selected.columns = ['ProteinID'] + ALL_TIME_COLS
    selected['NormalizedID'] = selected['ProteinID'].map(clean_protein_id)
    for tp in ALL_TIME_COLS:
        selected[tp] = pd.to_numeric(selected[tp], errors='coerce')
    if selected['NormalizedID'].duplicated().any():
        # Duplication among nonselected IDs is fine; selected IDs will be checked separately.
        pass
    return selected
def load_selected_trajectories():
    means = load_time_means()
    rows = []
    for group in GROUP_ORDER:
        symbol, uniprot = REPRESENTATIVES[group]
        # Exact UniProt-ID match; a token match is allowed only for wrapped IDs
        # like "sp|P13804|ETFA_HUMAN", never arbitrary substring matching.
        exact = means.loc[means['NormalizedID'] == uniprot]
        if exact.empty:
            pattern = rf'(?<![A-Z0-9]){re.escape(uniprot)}(?![A-Z0-9])'
            exact = means.loc[means['NormalizedID'].str.contains(pattern, na=False, regex=True)]
        if len(exact) != 1:
            raise ValueError(
                f'{symbol} ({uniprot}) matched {len(exact)} rows in {MEAN_FILE}. '
                f'Expected exactly 1; check the protein IDs and mean matrix.'
            )
        row = exact.iloc[0]
        y = row[ALL_TIME_COLS].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f'{symbol}: missing/nonfinite time-point mean(s).')
        std = float(np.std(y, ddof=0))
        if std <= 0:
            raise ValueError(f'{symbol}: zero variance across eight time means.')
        z = (y - float(np.mean(y))) / std
        for tp, original, zs in zip(ALL_TIME_COLS, y, z):
            rows.append({
                'Group': group, 'Gene': symbol, 'UniProt': uniprot,
                'MatchedID': row['ProteinID'], 'Timepoint': tp,
                'Mean_log2': original, 'Zscore_8means': zs,
            })
    return pd.DataFrame(rows)
# 3. PLOT STYLE
def set_style():
    from matplotlib import font_manager
    installed = {ft.name for ft in font_manager.fontManager.ttflist}
    font = next((f for f in ['Arial', 'Liberation Sans', 'DejaVu Sans'] if f in installed),
                'DejaVu Sans')
    plt.rcParams.update({
        'font.family': font,
        'font.size': 7.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.linewidth': 0.65,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#161616',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'text.color': '#161616',
        'xtick.color': '#3B3B3B',
        'ytick.color': '#3B3B3B',
        'axes.unicode_minus': False,
    })
def setup_axes(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_linewidth(.65)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT,
                   width=.65, length=2.3, pad=2)
    ax.grid(False)
def nice_limit(value):
    value = max(float(value), 0.5)
    for step in (0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        if value <= step:
            return float(step)
    return math.ceil(value / 50) * 50
def bar_panel(ax, selected, group):
    color = GROUP_COLORS[group]
    sub = selected.loc[selected.Group == group].sort_values('plot_order')
    ypos = np.arange(len(sub))[::-1]
    vals = sub['neglog10FDR'].to_numpy(float)
    ymax = nice_limit(max(vals) * 1.055)
    setup_axes(ax)
    for y, (_, row) in zip(ypos, sub.iterrows()):
        val = row.neglog10FDR
        # Both enrichment charts follow the same statistical color rule:
        # FDR < 0.05 = class color; FDR >= 0.05 = neutral gray.
        is_sig = bool(row.FDR_significant)
        ax.barh(y, val, height=.57,
                color=to_rgba(color, .18) if is_sig else NONSIG_FILL,
                edgecolor=color if is_sig else NONSIG_EDGE,
                linewidth=.75 if is_sig else .60, zorder=2)
        ax.text(.014, y, row.Display_name, va='center', ha='left',
                fontsize=GO_FONT, color='#161616' if is_sig else NONSIG_TEXT,
                transform=ax.get_yaxis_transform(), zorder=5,
                clip_on=False)
    if SHOW_FDR_REFERENCE:
        ax.axvline(-math.log10(FDR_THRESHOLD), color='#737B83',
                   linewidth=.95, linestyle=(0, (3, 2.5)), zorder=4)
    ax.set_xlim(0, ymax)
    ax.set_ylim(-.55, len(sub)-.45)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:g}'))
    ax.set_xlabel(r'$-\log_{10}(\mathrm{FDR})$', fontsize=LABEL_FONT, labelpad=1.3)
    return ax
def trajectory_panel(ax, trajectories, group):
    sub = trajectories.loc[trajectories.Group == group].copy()
    gene = REPRESENTATIVES[group][0]
    w0 = sub.set_index('Timepoint').loc[W0_COLS, 'Zscore_8means'].to_numpy(float)
    w8 = sub.set_index('Timepoint').loc[W8_COLS, 'Zscore_8means'].to_numpy(float)
    x = np.arange(4)
    setup_axes(ax)
    ax.axhline(0, color='#B9B9B9', linewidth=.60,
               linestyle=(0,(4,3)), zorder=0)
    for y, c in ((w0, W0_COLOR), (w8, W8_COLOR)):
        ax.plot(x, y, '-', marker='o', markersize=3.05, linewidth=1.40,
                color=c, markeredgecolor='white', markeredgewidth=.40,
                zorder=4)
    ax.set_xlim(-.14, 3.14)
    ax.set_ylim(-2.55, 2.55)      # consistent across all proteins
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_LABELS, fontsize=TICK_FONT)
    ax.set_yticks([-2, 0, 2])
    ax.set_ylabel('Z-score', fontsize=7.2, labelpad=0.6)
    ax.tick_params(axis='y', labelsize=TICK_FONT, pad=1)
    ax.set_xlabel('Time (min)', fontsize=LABEL_FONT, labelpad=1.3)
    ax.set_title(gene, fontsize=GENE_FONT, fontweight='medium', pad=4.5)
    return ax
def draw(selected, trajectories):
    set_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), facecolor='white')
    for i, group in enumerate(GROUP_ORDER):
        start = GROUP_STARTS[i]
        color = GROUP_COLORS[group]
        # Rounded, subtly tinted facet strips; no outer framing.
        from matplotlib.patches import FancyBboxPatch
        strip = FancyBboxPatch((start+.002, HEADER_Y), .320, HEADER_H,
                               transform=fig.transFigure, zorder=1,
                               boxstyle='round,pad=0,rounding_size=0.012',
                               facecolor=to_rgba(color, 0.13), edgecolor='none')
        fig.add_artist(strip)
        fig.text(start+.162, HEADER_Y+HEADER_H*.50, group,
                 fontsize=HEADER_FONT, ha='center', va='center',
                 color=color, fontweight='bold')
        a = fig.add_axes((start+BAR_OFFSET, PLOT_BOTTOM, BAR_W, PLOT_HEIGHT))
        b = fig.add_axes((start+TIME_OFFSET, PLOT_BOTTOM, TIME_W, PLOT_HEIGHT))
        bar_panel(a, selected, group)
        trajectory_panel(b, trajectories, group)
        if i != 2:
            fig.add_artist(Line2D([start+.328, start+.328],
                                  [.125, .815], transform=fig.transFigure,
                                  color='#E2E2E2', linewidth=.52, zorder=0))
    # One compact W0 / W8 legend at the top, below facet headers.
    legends = [Line2D([0], [0], color=c, marker='o', linewidth=1.35,
                      markersize=3.3, markerfacecolor=c, markeredgecolor='white',
                      markeredgewidth=.45, label=lab)
               for lab, c in (('W0',W0_COLOR), ('W8',W8_COLOR))]
    if SHOW_LEGEND:
        fig.legend(handles=legends, loc='lower center', bbox_to_anchor=(.5, .02),
                   ncol=2, frameon=False, fontsize=7.8, columnspacing=1.7,
                   handlelength=1.8, handletextpad=.4)
    return fig
# 4. OUTPUT
def main():
    pathways = load_selected_pathways()
    trajectories = load_selected_trajectories()
    output_dir = Path(OUTDIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pathways.to_csv(output_dir / 'Figure3B_selected_GO_terms_FDR.tsv', sep='\t', index=False)
    trajectories.to_csv(output_dir / 'Figure3B_representative_protein_trajectories.tsv', sep='\t', index=False)
    fig = draw(pathways, trajectories)
    pdf = output_dir / OUT_PDF
    fig.savefig(pdf, facecolor='white')       # No bbox_inches; exact 205 mm width
    if SAVE_PNG:
        fig.savefig(pdf.with_suffix('.png'), facecolor='white', dpi=400)
    plt.close(fig)
    print(str(pdf))                            # only print the PDF absolute path
    return pdf
if __name__ == '__main__':
    main()
