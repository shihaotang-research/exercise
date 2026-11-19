#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure S1: metabolite classes and PCA."""

import os
from pathlib import Path
import math
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1) PATHS (change only BASE_DIR if necessary)
BASE_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))

MET_ANNOTATION = BASE_DIR / 'metabolite_annotations.tsv'
MET_LIST       = BASE_DIR / 'metabolite_ids.txt'
MET_RAW        = BASE_DIR / 'metabolite_raw.tsv'
N_MET_QC       = 47                  # original: final 47 columns are metabolite QC

PRO_RAW        = BASE_DIR / 'protein_raw.tsv'
PRO_QC         = BASE_DIR / 'protein_qc.tsv'
PRO_LIST       = BASE_DIR / 'protein_ids.txt'

OUTPUT_DIR = BASE_DIR / 'Supplementary_Figure_S1_Metabolite_PCA'
OUTPUT_PDF = OUTPUT_DIR / 'Figure_S1_Metabolite_classes_PCA_A4.pdf'

# To place just the plots in Illustrator, optionally hide the printed figure legend.
PRINT_FIGURE_LEGEND = True
PAGE_W_MM, PAGE_H_MM = 210.0, 297.0  # A4 portrait, exact size

# Fixed color meaning across both PCA subpanels (weeks = colors, times = shapes)
W0_COLOR = '#2E8AC4'
W8_COLOR = '#E64B87'
QC_COLOR = '#242424'
BAR_COLOR = '#E8C6BD'
BAR_DOT = '#D67B66'
INK = '#202020'

TIME_CODES = ['base', '2min', '30min', '60min']
TIME_LABELS = {'base': 'baseline', '2min': '2 min',
               '30min': '30 min', '60min': '60 min'}
TIME_MARKERS = {'base': 'o', '2min': '^', '30min': 's', '60min': 'D'}
SAMPLE_TO_GROUP = {
    'A': 'W0_base',  'B1': 'W0_2min',  'B2': 'W0_30min', 'B3': 'W0_60min',
    'D': 'W8_base',  'F1': 'W8_2min',  'F2': 'W8_30min', 'F3': 'W8_60min'
}
GROUP_ORDER = [f'{w}_{t}' for w in ['W0', 'W8'] for t in TIME_CODES]

LEGEND_A = (
    '(A). Distribution of identified metabolites across the top 10 metabolite '
    'classes, ranked by the number of annotated metabolites.'
)
LEGEND_B = (
    '(B). Principal component analysis (PCA) of plasma proteomic and metabolomic '
    'profiles at W0 and W8 across four sampling time points: baseline and 2, 30, '
    'and 60 min after cardiopulmonary exercise testing (CPET). Blue and pink '
    'indicate W0 and W8, respectively, while different symbols distinguish '
    'sampling time points. Quality control (QC) samples are shown in black.'
)

def set_style():
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = 'Arial' if 'Arial' in fonts else (
        'Liberation Sans' if 'Liberation Sans' in fonts else 'DejaVu Sans'
    )
    plt.rcParams.update({
        'font.family': font,
        'font.size': 8.2,
        'text.color': INK,
        'axes.labelcolor': '#111111',
        'xtick.color': '#111111',
        'ytick.color': '#111111',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.unicode_minus': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    print('[Font]', font)

def required_files():
    paths = [MET_ANNOTATION, MET_LIST, MET_RAW, PRO_RAW, PRO_QC, PRO_LIST]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required source files:\n' + '\n'.join(missing))

def read_id_set(path):
    with open(path, encoding='utf-8') as fh:
        ids = {line.strip() for line in fh if line.strip()}
    if not ids:
        raise ValueError(f'Empty feature list: {path}')
    return ids

def read_table(path):
    # Source files are typically tab-separated; preserve the original automatic
    # separator detection as a fallback for different metabolite_raw.tsv exports.
    df = pd.read_csv(path, sep='\t', low_memory=False)
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=None, engine='python')
    df.columns = df.columns.astype(str).str.strip()
    return df

# 2) PANEL A: METABOLITE CLASS COUNTS (exact source counting rule)
def load_metabolite_classes():
    keep = read_id_set(MET_LIST)
    ann = read_table(MET_ANNOTATION)
    need = {'Metabolite', 'main_class'}
    if not need.issubset(ann.columns):
        raise ValueError(f'{MET_ANNOTATION.name}: missing columns {need-set(ann.columns)}')

    ann['Metabolite'] = ann['Metabolite'].fillna('').astype(str).str.strip()
    ann = ann[ann['Metabolite'].isin(keep)].drop_duplicates('Metabolite').copy()
    ann['main_class'] = ann['main_class'].fillna('').astype(str).str.strip()
    ann['main_class'] = ann['main_class'].replace(
        ['', 'nan', 'NaN', 'NA', '#N/A', 'None'], 'Unclassified'
    )
    n_annotated = ann['Metabolite'].nunique()
    if n_annotated != len(keep):
        print(f'[Warning] Metabolites in list={len(keep)}, annotated={n_annotated}')

    counts = (ann.groupby('main_class', as_index=False)['Metabolite']
              .nunique().rename(columns={'main_class': 'Class', 'Metabolite': 'Count'})
              .sort_values(['Count', 'Class'], ascending=[False, True])
              .reset_index(drop=True))
    if len(counts) < 10:
        raise ValueError('Fewer than 10 annotated metabolite classes')
    counts['Percent_of_annotated'] = counts['Count'] / n_annotated * 100
    counts.to_csv(OUTPUT_DIR / 'A_all_metabolite_class_counts.tsv',
                  sep='\t', index=False)
    print(f'[A] Metabolites annotated: {n_annotated}; classes: {len(counts)}')
    return counts.head(10).copy()

# 3) PANEL B: ORIGINAL QC PCA DATA PROCESSING
def parse_group(sample):
    s = str(sample).strip()
    if 'QC' in s.upper():
        return 'QC'
    m = re.search(r'(A|B1|B2|B3|D|F1|F2|F3)$', s, flags=re.I)
    return SAMPLE_TO_GROUP.get(m.group(1).upper(), 'Other') if m else 'Other'

def is_target_study_sample(column):
    """Keep only the eight intended biological time-point columns.

    Explicitly excludes B4 (and any other unscheduled columns) before
    constructing the input matrix, so they do not affect imputation or PCA.
    QC samples are selected separately from their dedicated source columns.
    """
    name = str(column).strip()
    return (name.lower().startswith('ae')
            and parse_group(name) in GROUP_ORDER)

def print_study_sample_audit(columns, selected, name):
    candidates = [str(c) for c in columns if str(c).lower().startswith('ae')]
    dropped_b4 = [c for c in candidates if re.search(r'B4$', c, flags=re.I)]
    print(f'[{name}] Selected 8-timepoint biological samples: {len(selected)}; '
          f'excluded B4 columns: {len(dropped_b4)}')
    if any(c in selected for c in dropped_b4):
        raise AssertionError(f'{name}: B4 sample unexpectedly included')

def global_min_half_log2(data, dataset_name):
    # Exactly one imputation constant per omics matrix, not per analyte.
    data = data.apply(pd.to_numeric, errors='coerce')
    arr = data.to_numpy(dtype=float, copy=True)
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        raise ValueError(f'{dataset_name}: no finite positive values')
    fill = float(valid.min()) / 2.0
    if (np.isfinite(arr) & (arr < 0)).any():
        raise ValueError(f'{dataset_name}: negative raw intensities; cannot log2')
    arr[~np.isfinite(arr) | (arr == 0)] = fill
    transformed = np.log2(arr)
    if not np.isfinite(transformed).all():
        raise ValueError(f'{dataset_name}: log2 produced invalid values')
    print(f'[{dataset_name}] Global positive minimum / 2 = {fill:.8g}')
    return pd.DataFrame(transformed, index=data.index, columns=data.columns)

def validate_groups(groups, name):
    other = groups[groups == 'Other']
    if len(other):
        raise ValueError(f'{name}: study/QC names not recognized: {other.index.tolist()[:12]}')
    print(f'[{name}] Samples by group:', groups.value_counts().to_dict())

def load_proteomics():
    original = read_table(PRO_RAW)
    qc = read_table(PRO_QC)
    original.rename(columns={original.columns[0]: 'ProteinAccessions'}, inplace=True)
    qc.rename(columns={qc.columns[0]: 'ProteinAccessions'}, inplace=True)

    keep = read_id_set(PRO_LIST)
    for df in (original, qc):
        df['ProteinAccessions'] = df['ProteinAccessions'].astype(str).str.strip()

    study_cols = [c for c in original.columns if is_target_study_sample(c)]
    print_study_sample_audit(original.columns, study_cols, 'Proteomics')
    meta = {'ProteinAccessions', 'Genes', 'Gene', 'gene',
            'Description', 'description'}
    qc_candidates = [c for c in qc.columns if c not in meta]
    explicit_qc = [c for c in qc_candidates if 'qc' in c.lower()]
    qc_cols = explicit_qc if explicit_qc else qc_candidates
    if not study_cols or not qc_cols:
        raise ValueError('Proteomics: missing study or QC columns')
    if set(study_cols) & set(qc_cols):
        raise ValueError('Proteomics: duplicate sample names between study and QC')

    original = original[original['ProteinAccessions'].isin(keep)]
    qc = qc[qc['ProteinAccessions'].isin(keep)]
    merged = (original[['ProteinAccessions'] + study_cols]
              .merge(qc[['ProteinAccessions'] + qc_cols],
                     on='ProteinAccessions', how='inner')
              .drop_duplicates('ProteinAccessions')
              .set_index('ProteinAccessions'))
    X = merged.T
    groups = pd.Series([parse_group(s) for s in X.index], index=X.index)
    # The original protein QC file is an explicitly defined QC dataset.
    groups.loc[qc_cols] = 'QC'
    validate_groups(groups, 'Proteomics')
    print(f'[Proteomics] Molecular features: {X.shape[1]} (expected 2039)')
    if X.shape[1] != len(keep):
        print('[Warning] Some protein IDs do not occur in both source matrices')
    return global_min_half_log2(X, 'Proteomics'), groups

def load_metabolomics():
    raw = read_table(MET_RAW)
    feature_col = raw.columns[0]
    raw[feature_col] = raw[feature_col].astype(str).str.strip()
    raw = raw[raw[feature_col].isin(read_id_set(MET_LIST))]
    all_cols = list(raw.columns)
    # Exactly as original: the final 47 columns are QC, not biological samples.
    qc_cols = list(all_cols[-N_MET_QC:])
    study_cols = [c for c in all_cols[:-N_MET_QC] if is_target_study_sample(c)]
    print_study_sample_audit(all_cols[:-N_MET_QC], study_cols, 'Metabolomics')
    if set(study_cols) & set(qc_cols):
        raise ValueError('Metabolomics: study/QC sample columns overlap')
    if not study_cols or not qc_cols:
        raise ValueError('Metabolomics: missing study or QC columns')
    raw = raw.drop_duplicates(feature_col).set_index(feature_col)
    X = raw[study_cols + qc_cols].T
    groups = pd.Series([parse_group(s) for s in X.index], index=X.index)
    groups.loc[qc_cols] = 'QC'
    validate_groups(groups, 'Metabolomics')
    print(f'[Metabolomics] Molecular features: {X.shape[1]} (expected 1142)')
    return global_min_half_log2(X, 'Metabolomics'), groups

def do_pca(X, groups, dataset):
    # As in the original: QC samples PARTICIPATE in scaling and PCA fitting.
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, svd_solver='full')
    xy = pca.fit_transform(scaled)
    scores = pd.DataFrame({
        'sample': X.index.astype(str),
        'group': groups.reindex(X.index).values,
        'PC1': xy[:, 0], 'PC2': xy[:, 1]
    })
    scores.to_csv(OUTPUT_DIR / f'B_{dataset}_QC_PCA_scores.tsv',
                  sep='\t', index=False)
    variance = pca.explained_variance_ratio_[:2] * 100.0
    print(f'[B] {dataset}: PC1={variance[0]:.2f}%, PC2={variance[1]:.2f}%')
    return scores, variance

# 4) A4 VECTOR FIGURE: A (WIDE HORIZONTAL BARS), B (TWO PCA PANELS)
def mm_box(fig, x, y, w, h):
    """Axes rectangle in millimeters from page bottom-left."""
    return fig.add_axes([x / PAGE_W_MM, y / PAGE_H_MM,
                         w / PAGE_W_MM, h / PAGE_H_MM])

def mm_text(fig, x, y, string, **kw):
    return fig.text(x / PAGE_W_MM, y / PAGE_H_MM, string, **kw)

def panel_heading(fig, letter, title, y):
    mm_text(fig, 12.0, y, letter, fontsize=13.8, weight='bold',
            ha='left', va='center', color='#111111')
    mm_text(fig, 25.0, y, title, fontsize=11.3, weight='bold',
            ha='left', va='center', color='#111111')

def draw_panel_a(fig, top10):
    panel_heading(fig, 'A', 'Metabolite classes', 281.0)
    ax = mm_box(fig, 82.0, 198.0, 106.0, 69.0)
    ordered = top10.sort_values('Count', ascending=True)
    labels = ordered['Class'].astype(str).tolist()
    values = ordered['Count'].to_numpy(dtype=float)
    yy = np.arange(len(values))
    max_value = int(np.ceil(values.max() * 1.29 / 10.0) * 10)
    # Bar + end dot keeps the original salmon palette but is clearer at A4 size.
    ax.barh(yy, values, height=0.58, color=BAR_COLOR,
            edgecolor='none', zorder=2)
    ax.scatter(values, yy, s=40, marker='o', color=BAR_DOT,
               edgecolors='white', linewidths=0.6, zorder=4)
    for y, val in zip(yy, values):
        ax.text(val + 3.1, y, f'{int(val)}', va='center', ha='left',
                fontsize=8.4, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=8.35)
    ax.set_ylim(-0.65, len(values) - 0.35)
    ax.set_xlim(0, max_value)
    ax.set_xticks(np.linspace(0, max_value - max_value % 4, 5).round().astype(int))
    ax.set_xlabel('Number of metabolites', fontsize=8.6, labelpad=4)
    ax.grid(axis='x', color='#E9EDF0', linewidth=0.65, zorder=0)
    ax.tick_params(axis='y', length=0, pad=10)
    ax.tick_params(axis='x', labelsize=8, length=2.5, width=0.65, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('#B9BEC2')
    ax.spines['bottom'].set_linewidth(0.65)

def add_covariance_ellipse(ax, xy, color):
    xy = xy[['PC1', 'PC2']].dropna().to_numpy(dtype=float)
    if len(xy) < 3:
        return
    cov = np.cov(xy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    major = eigenvectors[:, order[0]]
    theta = np.degrees(np.arctan2(major[1], major[0]))
    factor = math.sqrt(chi2.ppf(.95, df=2))
    ellipse = Ellipse(
        xy.mean(axis=0),
        width=2 * factor * math.sqrt(eigenvalues[0]),
        height=2 * factor * math.sqrt(eigenvalues[1]),
        angle=theta,
        fill=False, edgecolor=color, alpha=0.52,
        linewidth=1.05, zorder=2,
    )
    ax.add_patch(ellipse)

def draw_one_pca(ax, scores, var):
    for week in ('W0', 'W8'):
        col = W0_COLOR if week == 'W0' else W8_COLOR
        for tp in TIME_CODES:
            key = f'{week}_{tp}'
            subset = scores[scores['group'] == key]
            if subset.empty:
                continue
            add_covariance_ellipse(ax, subset, col)
            ax.scatter(subset['PC1'], subset['PC2'], s=13,
                       color=col, marker=TIME_MARKERS[tp],
                       alpha=.79, edgecolors='none', zorder=3,
                       rasterized=False)
    qc = scores[scores['group'] == 'QC']
    if not qc.empty:
        ax.scatter(qc['PC1'], qc['PC2'], s=17, marker='o',
                   color=QC_COLOR, edgecolors='white',
                   linewidths=.25, zorder=6)
    ax.axhline(0, linewidth=.55, color='#E4E7EB', zorder=0)
    ax.axvline(0, linewidth=.55, color='#E4E7EB', zorder=0)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)', fontsize=8.7, labelpad=3)
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)', fontsize=8.7, labelpad=3)
    ax.tick_params(labelsize=7.6, length=3, width=.7, direction='out', pad=2)
    for spine in ax.spines.values():
        spine.set_color('#666666')
        spine.set_linewidth(.75)
    ax.margins(x=.06, y=.08)
    ax.grid(False)

def draw_pca_legend(fig):
    # True two-row legend; color conveys W0/W8 and shape conveys time.
    # Fixed positions avoid unexpected reflow at A4 width.
    left, step = 17.0, 38.0
    rows = [('W0', 60.0), ('W8', 50.4)]
    for week, y in rows:
        color = W0_COLOR if week == 'W0' else W8_COLOR
        for j, tp in enumerate(TIME_CODES):
            x = left + step * j
            marker = Line2D([x / PAGE_W_MM], [y / PAGE_H_MM],
                            transform=fig.transFigure,
                            marker=TIME_MARKERS[tp], linestyle='None',
                            markersize=5.3, markerfacecolor=color,
                            markeredgecolor='none')
            fig.add_artist(marker)
            mm_text(fig, x + 4.4, y, f'{week} {TIME_LABELS[tp]}',
                    ha='left', va='center', fontsize=7.75)
    x, y = left + step * 4, 50.4
    fig.add_artist(Line2D([x / PAGE_W_MM], [y / PAGE_H_MM],
                          transform=fig.transFigure, marker='o',
                          linestyle='None', markersize=5.5,
                          markerfacecolor=QC_COLOR, markeredgecolor='none'))
    mm_text(fig, x + 4.4, y, 'QC', ha='left', va='center', fontsize=7.75)

def draw_legend_text(fig):
    if not PRINT_FIGURE_LEGEND:
        return
    # No colored/shaded caption box; manuscript-ready black text only.
    line_a = textwrap.fill(LEGEND_A, width=120, break_long_words=False)
    line_b = textwrap.fill(LEGEND_B, width=120, break_long_words=False)
    mm_text(fig, 12.0, 40.0, line_a + '\n' + line_b,
            fontsize=7.45, color=INK,
            va='top', ha='left', linespacing=1.32)

def make_pdf(class_counts, pro_scores, pro_var, met_scores, met_var):
    fig = plt.figure(figsize=(PAGE_W_MM / 25.4, PAGE_H_MM / 25.4))
    draw_panel_a(fig, class_counts)

    panel_heading(fig, 'B', 'PCA of multi-omics profiles', 172.2)
    mm_text(fig, 59.5, 160.5, 'Proteomics', fontsize=9.7, weight='bold',
            ha='center', va='center')
    mm_text(fig, 157.4, 160.5, 'Metabolomics', fontsize=9.7, weight='bold',
            ha='center', va='center')
    ax_pro = mm_box(fig, 21.0, 80.0, 77.0, 76.0)
    ax_met = mm_box(fig, 118.5, 80.0, 77.0, 76.0)
    draw_one_pca(ax_pro, pro_scores, pro_var)
    draw_one_pca(ax_met, met_scores, met_var)
    draw_pca_legend(fig)
    draw_legend_text(fig)

    # DO NOT use bbox_inches='tight': it would modify the exact A4 dimensions.
    fig.savefig(OUTPUT_PDF, format='pdf', facecolor='white')
    plt.close(fig)
    print(f'\n[Saved PDF] {OUTPUT_PDF}')

def main():
    set_style()
    required_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = load_metabolite_classes()
    X_pro, groups_pro = load_proteomics()
    X_met, groups_met = load_metabolomics()
    pro_scores, pro_var = do_pca(X_pro, groups_pro, 'Proteomics')
    met_scores, met_var = do_pca(X_met, groups_met, 'Metabolomics')
    make_pdf(counts, pro_scores, pro_var, met_scores, met_var)
    print('[Done] Single vector A4 PDF plus audit TSVs. No PNG output.')

if __name__ == '__main__':
    main()
