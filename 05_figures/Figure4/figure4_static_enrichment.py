#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4: resting-state network module enrichment plot."""
from __future__ import annotations
import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# 0. Settings
ROOT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
INPUT_DIR = ROOT / 'resting_module_enrichment'
OUT_DIR = ROOT / 'Static_module_enrichment_horizontal_palette_143mm'  # NEW output dir
STEM = 'Static_module_enrichment_preserved_unmatched_143mm_module_palette'

FIG_WIDTH_MM = 143.0              # narrower than previous 154.5 mm
FIG_HEIGHT_MM = 86.0               # also used in the single scatter script
FONT_SIZE = 7.0
DPI = 400

COLORS = {'W0': '#2E8AC4', 'W8': '#E64B87'}
FDR_LIMIT = 0.05
JACCARD_LIMIT = 0.40
MIN_HITS = 3
METABOLITE_FDR_COLUMN = 'FDR_within_module'

# EXACT SAME background/outline colors as Figure 3 network module legend.
MODULE_BG_COLORS = {
    'M1': '#F4C7C3', 'M2': '#BFD7F2', 'M3': '#BFE3C0',
    'M4': '#F3E2A7', 'M5': '#D9C8F0', 'M6': '#BFE3EA',
    'M7': '#F7D6BF', 'M8': '#C9D0D8',
    'N1': '#F6D1E6', 'N2': '#D8D1C5', 'N3': '#E6CFC4',
}
MODULE_EDGE_COLORS = {
    'M1': '#E78B84', 'M2': '#7FB0E3', 'M3': '#7FC68A',
    'M4': '#D8B94F', 'M5': '#B497DE', 'M6': '#7DC8D6',
    'M7': '#E5AE7B', 'M8': '#93A0AD',
    'N1': '#E49BC4', 'N2': '#B7AA93', 'N3': '#D0A792',
}

# Original network-module to Figure 3 DISPLAY-ID mapping.
# Do not infer labels by stripping W0/W8: the originals are renumbered!
# Cross-check the displayed unmatched labels against the enrichment tables.
EXPECTED_MODULE_DISPLAY = {
    ('W0', 'M4'): 'M5',
    ('W0', 'M5'): 'M6',
    ('W0', 'M6'): 'M7',
    ('W0', 'M7'): 'M4',
    ('W8', 'M4'): 'N1',
    ('W8', 'M6'): 'N2',
}

# Precise millimeter layout (2 columns)
PANEL_LEFT_X = 2.8
PANEL_RIGHT_X = 72.8
PANEL_WIDTH = 67.6
MODULE_W = 6.7
GAP_AFTER_MODULE = 0.5
TERMS_W = 35.0
GAP_BEFORE_DATA = 0.6
DATA_W = PANEL_WIDTH - MODULE_W - GAP_AFTER_MODULE - TERMS_W - GAP_BEFORE_DATA
PLOT_TOP_MM = 17.0
PLOT_HEIGHT_MM = 54.5
GROUP_GAP_MM = 1.10

# Original fixed selection; display names unchanged.
HUMORAL = ('GO:0006959', 'Humoral immune response')
ECM = ('GO:0030198', 'ECM organization')
ACUTE = ('GO:0006953', 'Acute-phase response')
PROTEOLYSIS = ('GO:0006508', 'Proteolysis')
TRANSLATION = ('GO:0006412', 'Translation')
RNA = ('GO:0006396', 'RNA processing')
CYTOKINESIS = ('GO:0000910', 'Cytokinesis')
LOCALIZATION = ('GO:0008104', 'Protein localization')
DEPHOS = ('GO:0006470', 'Protein dephosphorylation')
RESPIRATION = ('GO:0045333', 'Cellular respiration')
TRANSPORT = ('GO:0055085', 'Transmembrane transport')
ER_STRESS = ('GO:0034976', 'ER stress response')
VESICLE = ('GO:0016192', 'Vesicle-mediated transport')
TRNA = ('GO:0043039', 'tRNA aminoacylation')
CYTO_TRANSLATION = ('GO:0002181', 'Cytoplasmic translation')
NUCLEOSOME = ('GO:0006334', 'Nucleosome assembly')
GPC = ('Glycerophosphocholines', 'Glycerophosphocholines')
CERAMIDES = ('Ceramides', 'Ceramides')
SPHINGO = ('Phosphosphingolipids', 'Phosphosphingolipids')
ESTERS = ('Fatty esters', 'Fatty esters')

PAIRED_SELECTION = {
    ('M1', 'M1'): [HUMORAL, ECM, ACUTE, PROTEOLYSIS],
    ('M2', 'M3'): [TRANSLATION, RNA, CYTOKINESIS, GPC],
    ('M3', 'M2'): [LOCALIZATION, DEPHOS, RESPIRATION, TRANSPORT],
}
UNMATCHED_SELECTION = {
    ('W0', 'M4'): [ESTERS],
    ('W0', 'M5'): [RESPIRATION, TRANSPORT, ER_STRESS],
    ('W0', 'M6'): [VESICLE, CYTOKINESIS, TRNA],
    ('W0', 'M7'): [GPC, CERAMIDES, SPHINGO],
    ('W8', 'M4'): [GPC, ESTERS],
    ('W8', 'M6'): [CYTO_TRANSLATION, RNA, NUCLEOSOME],
}

def set_style():
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in ('Arial', 'Liberation Sans', 'DejaVu Sans') if name in fonts), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': chosen, 'font.size': FONT_SIZE,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.unicode_minus': False, 'axes.linewidth': 0.75,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'text.color': '#20252A',
    })

def exactly_one(pattern):
    found = sorted(INPUT_DIR.glob(pattern))
    if len(found) != 1:
        raise FileNotFoundError(f'Expected one input for {pattern}: {found}')
    return found[0]

def read_inputs():
    pairs = pd.read_csv(exactly_one('01_all_module_pairs.txt'), sep='\t')
    go = pd.read_csv(exactly_one('06_each_module_enrichment_all_results*.txt'), sep='\t')
    met = pd.read_csv(exactly_one('16E_each_module_metabolite_mainclass_ALL_tests*.txt'), sep='\t')
    for table, cols in (
        (pairs, ('module_W0','module_W8','jaccard')),
        (go, ('network','module','ID','display_id','p.adjust','Count')),
        (met, ('network','module','display_id','main_class',METABOLITE_FDR_COLUMN,'count_in_module')),
    ):
        missing = set(cols) - set(table.columns)
        if missing:
            raise ValueError(f'Enrichment input missing columns: {sorted(missing)}')
    for tab in (go, met):
        for c in ('network','module'):
            tab[c] = tab[c].astype(str).str.strip()
    for c in ('module_W0','module_W8'):
        pairs[c] = pairs[c].astype(str).str.strip()
    pairs['jaccard'] = pd.to_numeric(pairs['jaccard'], errors='raise')
    go['p.adjust'] = pd.to_numeric(go['p.adjust'], errors='raise')
    go['Count'] = pd.to_numeric(go['Count'], errors='raise')
    met[METABOLITE_FDR_COLUMN] = pd.to_numeric(met[METABOLITE_FDR_COLUMN], errors='raise')
    met['count_in_module'] = pd.to_numeric(met['count_in_module'], errors='raise')
    for tab, cols in ((go, ['network','module','ID']), (met, ['network','module','main_class'])):
        if tab.duplicated(cols).any():
            raise ValueError(f'Input has duplicated enrichment test keys: {cols}')
    return pairs, go, met

def retrieve_test(go, met, state, module, term):
    is_protein = term.startswith('GO:')
    tab = go if is_protein else met
    key = 'ID' if is_protein else 'main_class'
    fdr_col = 'p.adjust' if is_protein else METABOLITE_FDR_COLUMN
    count_col = 'Count' if is_protein else 'count_in_module'
    hit = tab.loc[(tab['network'] == state) & (tab['module'] == module) & (tab[key].astype(str) == term)]
    if hit.empty:
        return {'status': 'not_reported', 'FDR': np.nan, 'count': np.nan, 'x': np.nan,
                'display': False, 'eligible': False}
    r = hit.iloc[0]
    fdr = float(r[fdr_col])
    count = float(r[count_col])
    if not np.isfinite(fdr) or not (0 <= fdr <= 1):
        raise ValueError(f'Invalid {fdr_col}: {state} {module} {term}: {fdr}')
    if not np.isfinite(count):
        raise ValueError(f'Invalid hit count: {state} {module} {term}')
    # Same drawing rule as original curated script: only q < 0.05,
    # including secondary points that might have fewer than MIN_HITS.
    shown = fdr < FDR_LIMIT
    return {
        'status': 'reported', 'FDR': fdr, 'count': count,
        'x': -math.log10(max(fdr, np.nextafter(0.0, 1.0))) if shown else np.nan,
        'display': shown, 'eligible': shown and count >= MIN_HITS,
    }

def prepare(pairs, go, met):
    qualifying = pairs.loc[pairs['jaccard'] >= JACCARD_LIMIT]
    actual = set(zip(qualifying['module_W0'], qualifying['module_W8']))
    if actual != set(PAIRED_SELECTION):
        raise ValueError(f'Preserved pairs changed. Expected {set(PAIRED_SELECTION)}; got {actual}')
    if qualifying['module_W0'].duplicated().any() or qualifying['module_W8'].duplicated().any():
        raise ValueError('Ambiguous many-to-one module mapping')
    for state, module in UNMATCHED_SELECTION:
        if module in set(qualifying[f'module_{state}']):
            raise ValueError(f'{state} {module} is matched, but appears in unmatched selection')

    groups = {'Preserved modules': [], 'Unmatched modules': []}
    records = []
    for (m0, m8), chosen in PAIRED_SELECTION.items():
        group = {'label': m0, 'modules': [('W0', m0), ('W8', m8)], 'terms': chosen, 'kind': 'preserved'}
        groups['Preserved modules'].append(group)
    # IMPORTANT: use Figure 3's presentation labels, NOT original module IDs.
    # This restores the intended sequence M4-M7 / N1-N2, without week prefixes.
    unmatched_groups = []
    for (state, module), chosen in UNMATCHED_SELECTION.items():
        canonical_label = EXPECTED_MODULE_DISPLAY[(state, module)]
        # Verify the original output's display_id before applying figure colors.
        observed_labels = set()
        for tab in (go, met):
            subset = tab.loc[(tab['network'] == state) & (tab['module'] == module), 'display_id']
            observed_labels.update(subset.dropna().astype(str).str.strip())
        if not observed_labels or observed_labels != {canonical_label}:
            raise ValueError(
                f'Display-ID mismatch for {state} {module}: '
                f'expected {canonical_label}, found {sorted(observed_labels)}. '
                'Check input version and Figure 3 Sankey mapping.'
            )
        unmatched_groups.append({
            'label': canonical_label,
            'modules': [(state, module)],
            'terms': chosen, 'kind': state,
        })
    # Figure 3 module-legend order.
    def label_sort_key(g):
        label = g['label']
        return (0 if label.startswith('M') else 1, int(label[1:]))
    groups['Unmatched modules'] = sorted(unmatched_groups, key=label_sort_key)

    for panel, panel_groups in groups.items():
        for group in panel_groups:
            for term, label in group['terms']:
                present = []
                for state, mod in group['modules']:
                    res = retrieve_test(go, met, state, mod, term)
                    records.append({
                        'panel': panel, 'display_module': group['label'],
                        'network': state, 'original_module': mod,
                        'term_id': term, 'term_label': label,
                        'omics': 'Protein' if term.startswith('GO:') else 'Metabolite',
                        **res,
                    })
                    present.append(res)
                if not any(r['eligible'] for r in present):
                    raise ValueError(f'Curated term no longer passes selection criterion: '
                                     f"{group['label']} / {label}")
    return groups, pd.DataFrame(records)

def mm_box(fig, left, top, width, height):
    return [left / FIG_WIDTH_MM, (FIG_HEIGHT_MM - top - height) / FIG_HEIGHT_MM,
            width / FIG_WIDTH_MM, height / FIG_HEIGHT_MM]

def draw_panel(fig, x0, title, groups, data, xmax, label_artists):
    # A background axis spans the entire group band (but not the x ticks).
    bg = fig.add_axes(mm_box(fig, x0, PLOT_TOP_MM, PANEL_WIDTH, PLOT_HEIGHT_MM), zorder=0)
    bg.set_xlim(0, PANEL_WIDTH)
    bg.set_ylim(PLOT_HEIGHT_MM, 0)
    bg.axis('off')

    mod_x = x0
    term_x = x0 + MODULE_W + GAP_AFTER_MODULE
    dot_x = term_x + TERMS_W + GAP_BEFORE_DATA
    mod_ax = fig.add_axes(mm_box(fig, mod_x, PLOT_TOP_MM, MODULE_W, PLOT_HEIGHT_MM), zorder=2)
    term_ax = fig.add_axes(mm_box(fig, term_x, PLOT_TOP_MM, TERMS_W, PLOT_HEIGHT_MM), zorder=2)
    dot_ax = fig.add_axes(mm_box(fig, dot_x, PLOT_TOP_MM, DATA_W, PLOT_HEIGHT_MM), zorder=3)
    for a in (mod_ax, term_ax, dot_ax):
        a.patch.set_alpha(0)
        a.set_ylim(PLOT_HEIGHT_MM, 0)
    mod_ax.set_xlim(0, MODULE_W); mod_ax.axis('off')
    term_ax.set_xlim(0, TERMS_W); term_ax.axis('off')

    fig.text(x0/FIG_WIDTH_MM, 1 - 12.2/FIG_HEIGHT_MM,
             title, fontsize=8.8, fontweight='bold', ha='left', va='bottom')

    nrows = sum(len(g['terms']) for g in groups)
    row_h = (PLOT_HEIGHT_MM - GROUP_GAP_MM*(len(groups) - 1))/nrows
    if row_h < 3.0:
        raise ValueError('Insufficient vertical space for readable 7pt labels')

    ycursor = 0.0
    for gi, group in enumerate(groups):
        gheight = row_h * len(group['terms'])
        bg.add_patch(Rectangle((0, ycursor), PANEL_WIDTH, gheight,
                               facecolor=('#F3F5F6' if gi % 2 else '#FFFFFF'),
                               edgecolor='none'))
        # The exact same M/N module palette as the network figure legend.
        display_label = group['label']
        if display_label not in MODULE_BG_COLORS:
            raise ValueError(f'Unknown display module: {display_label}')
        stripe = MODULE_BG_COLORS[display_label]
        accent = MODULE_EDGE_COLORS[display_label]
        bg.add_patch(Rectangle((0, ycursor), MODULE_W, gheight,
                               facecolor=stripe, edgecolor='none'))
        bg.add_patch(Rectangle((0, ycursor), .42, gheight,
                               facecolor=accent, edgecolor='none'))
        mod_ax.text(MODULE_W/2+.15, ycursor + gheight/2, display_label,
                    ha='center', va='center', fontsize=7.0, fontweight='bold', color='#20252A')
        for j, (term, term_label) in enumerate(group['terms']):
            yy = ycursor + (j+.5)*row_h
            txt = term_ax.text(.3, yy, term_label, va='center', ha='left',
                               fontsize=7.0, color='#20252A', clip_on=True)
            label_artists.append((txt, term_ax, dot_ax))
            rows = data.loc[(data.panel == title) & (data.display_module == group['label'])
                            & (data.term_id == term) & (data.display)]
            if len(rows) == 2:
                xx = rows.set_index('network')['x']
                dot_ax.plot([xx['W0'], xx['W8']], [yy, yy],
                            color='#A9B6BF', lw=.78, zorder=2)
            for r in rows.itertuples(index=False):
                dot_ax.scatter(r.x, yy, marker=('o' if r.omics == 'Protein' else '^'),
                               s=(21 if r.omics == 'Protein' else 28),
                               c=COLORS[r.network], edgecolors='white', linewidths=.35,
                               zorder=4, clip_on=True)
        ycursor += gheight + (GROUP_GAP_MM if gi < len(groups)-1 else 0)

    dot_ax.set_xlim(0, xmax)
    dot_ax.axvline(-math.log10(FDR_LIMIT), lw=.8, color='#A4AFB8',
                   ls=(0, (2.5, 2.4)), zorder=1)
    tick_step = 4 if xmax <= 20 else int(math.ceil(xmax/5/2)*2)
    dot_ax.set_xticks(np.arange(0, xmax+0.01, tick_step))
    dot_ax.set_yticks([])
    dot_ax.tick_params(axis='x', width=.65, length=2.5, labelsize=7,
                       pad=1.6, colors='#333A40')
    for side in ('left', 'top', 'right'):
        dot_ax.spines[side].set_visible(False)
    dot_ax.spines['bottom'].set_color('#4C5760')
    dot_ax.spines['bottom'].set_linewidth(.8)
    dot_ax.set_xlabel(r'$-\log_{10}(\mathrm{FDR})$', fontsize=7.2, labelpad=2.0)
    return dot_ax

def draw(groups, records, outdir):
    valid = pd.to_numeric(records.loc[records.display, 'x'], errors='coerce').dropna()
    if valid.empty:
        raise ValueError('No significant enrichment point to plot')
    xmax = max(4, int(math.ceil((float(valid.max())+.7)/4))*4)
    fig = plt.figure(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4), facecolor='white')
    labels = []
    draw_panel(fig, PANEL_LEFT_X, 'Preserved modules', groups['Preserved modules'], records, xmax, labels)
    draw_panel(fig, PANEL_RIGHT_X, 'Unmatched modules', groups['Unmatched modules'], records, xmax, labels)

    time_handles = [Line2D([], [], ls='', marker='o', ms=5.0, mfc=COLORS[n],
                           mec='none', label=n) for n in ('W0', 'W8')]
    fig.legend(handles=time_handles, ncol=2, loc='center', bbox_to_anchor=(.89, .948),
               frameon=False, fontsize=7, handletextpad=.35, columnspacing=1.0)

    kind_handles = [
        Line2D([], [], ls='', marker='o', ms=4.6, mfc='#777777', mec='none', label='Protein GO'),
        Line2D([], [], ls='', marker='^', ms=4.8, mfc='#777777', mec='none', label='Metabolite class'),
        Line2D([], [], ls=(0,(2.5,2.4)), lw=.8, color='#A4AFB8', label='FDR = 0.05'),
    ]
    fig.legend(handles=kind_handles, ncol=3, loc='lower center',
               bbox_to_anchor=(.50, .012), frameon=False,
               fontsize=7.0, handlelength=1.8, handletextpad=.43, columnspacing=1.6)

    # This guards the failure that was visible in the previous horizontal plot.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt, term_ax, dot_ax in labels:
        tx = txt.get_window_extent(renderer)
        dx = dot_ax.get_window_extent(renderer)
        if tx.x1 >= dx.x0 - 1:
            plt.close(fig)
            raise RuntimeError(f'Pathway name crosses into the dot area: {txt.get_text()}')
        if tx.x1 >= term_ax.get_window_extent(renderer).x1 - .3:
            plt.close(fig)
            raise RuntimeError(f'Pathway name exceeds its allocated width: {txt.get_text()}')

    pdf = outdir / f'{STEM}.pdf'
    png = outdir / f'{STEM}.png'
    fig.savefig(pdf, dpi=DPI, facecolor='white',
        metadata={'Title': 'Static preserved and unmatched module enrichment'})
    fig.savefig(png, dpi=300, facecolor='white')
    plt.close(fig)
    return pdf.resolve()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    pairs, go, met = read_inputs()
    groups, records = prepare(pairs, go, met)
    records.to_csv(OUT_DIR / f'{STEM}_source_statistics.tsv', sep='\t', index=False)
    print(draw(groups, records, OUT_DIR))

if __name__ == '__main__':
    main()
