#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure S3: protein–disease associations with BH-FDR across 36 comparisons."""
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# Settings
PROJECT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
UKB = Path(os.environ.get("MULTIOMICS_UKB_DIR", str(Path(__file__).resolve().parents[2] / "data" / "ukb")))
HR_FILE = PROJECT / 'ukb_incident_cox' / 'incident_protein_cox_results.tsv'
ANNOTATION_FILES = [PROJECT / 'disease_definitions.tsv', UKB / 'disease_definitions.tsv']
OUT_DIR = PROJECT / 'supplementary_figures' / 'protein_disease_associations'
PDF_NAME = 'Supplementary_Figure_S3_Protein_Disease_Associations.pdf'

CENSOR_DATE = '2023-11-30'
SOURCE_TAG = os.environ.get('MULTIOMICS_COX_SOURCE_TAG', 'USER_SUPPLIED_630')
EXPECTED_DISEASES = 630
SEED = 2026
SHARED_Y_SCALE = True  # same scale across the ten panels

# Data may still use the legacy name Trend-only; standardize display only.
RAW_CLASS_MAP = {'Baseline-only': 'Baseline-only',
                 'Dual-response': 'Dual-response',
                 'Trend-only': 'Dynamic-only',
                 'Dynamic-only': 'Dynamic-only'}
CLASS_ORDER = ['Baseline-only', 'Dual-response', 'Dynamic-only']
CLASS_SHORT = ['Base', 'Dual', 'Dyn']
COLORS = {'Baseline-only': '#4C95BC',
          'Dual-response': '#55A96E',
          'Dynamic-only': '#E57C5E'}

ROOT_MAP = {
    'Chapter I Certain infectious and parasitic diseases': 'Infectious/parasitic',
    'Chapter II Neoplasms': 'Neoplasms',
    'Chapter III Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism': 'Blood/immune',

    'Chapter IV Endocrine, nutritional and metabolic diseases': 'Endocrine/metabolic',
    'Chapter V Mental and behavioural disorders': 'Mental/behavioural',
    'Chapter VI Diseases of the nervous system': 'Nervous system',
    'Chapter VII Diseases of the eye, adnexa, ear and mastoid process': 'Eye/ear',
    'Chapter IX Diseases of the circulatory system': 'Circulatory',
    'Chapter X Diseases of the respiratory system': 'Respiratory',
    'Chapter XI Diseases of the digestive system': 'Digestive',
    'Chapter XII Diseases of the skin and subcutaneous tissue': 'Skin/subcutaneous',
    'Chapter XIII Diseases of the musculoskeletal system and connective tissue': 'Musculoskeletal',
    'Chapter XIV Diseases of the genitourinary system': 'Genitourinary',
}

# The same 10 systems and order as the uploaded analysis script.
SYSTEMS = [
    'Mental/behavioural', 'Respiratory',
    'Blood/immune', 'Digestive',
    'Skin/subcutaneous', 'Musculoskeletal',
    'Nervous system', 'Genitourinary',
    'Neoplasms', 'Eye/ear',
]
EXCLUDED = {'Infectious/parasitic'}
FULL_SYSTEMS = ['Endocrine/metabolic', 'Circulatory'] + SYSTEMS
SYSTEM_LABEL = {'Mental/behavioural': 'Mental/behavioral'}
PAIRS = [(0, 1), (1, 2), (0, 2)]
PAGE_W, PAGE_H = 210, 297  # A4 portrait, millimetres

def read_annotation():
    f = next((p for p in ANNOTATION_FILES if p.is_file()), None)
    if f is None:
        raise FileNotFoundError(f'disease_definitions.tsv not found in {ANNOTATION_FILES}')
    ann = pd.read_csv(f, sep='\t', dtype=str, low_memory=False)
    key = {str(c).strip().lower(): c for c in ann.columns}
    if {'disease code', 'root'} <= set(key):
        ann = ann.rename(columns={key['disease code']: 'disease', key['root']: 'Root'})
    elif {'disease', 'root'} <= set(key):
        ann = ann.rename(columns={key['disease']: 'disease', key['root']: 'Root'})
    else:
        ann = pd.read_csv(f, sep='\t', header=None, dtype=str).iloc[:, :3].copy()
        if ann.shape[1] < 3:
            raise ValueError('disease_definitions.tsv requires disease ID and Root annotation')
        ann.columns = ['disease', 'DiseaseLabel', 'Root']
    ann = ann[['disease', 'Root']].dropna().copy()
    for c in ['disease', 'Root']:
        ann[c] = ann[c].str.strip().str.replace(r'\s+', ' ', regex=True)
    if ann.groupby('disease')['Root'].nunique().gt(1).any():
        raise ValueError('Conflicting ICD chapter annotations')
    ann = ann.drop_duplicates('disease')
    ann['DiseaseSystem'] = ann['Root'].str.lower().map(
        {k.lower(): v for k, v in ROOT_MAP.items()})
    return ann[['disease', 'DiseaseSystem']]

def read_hr():
    if not HR_FILE.is_file():
        raise FileNotFoundError(f'Missing original protein Cox results: {HR_FILE}')
    d = pd.read_csv(HR_FILE, sep='\t', low_memory=False)
    required = {'disease', 'coding', 'MoleculeClass', 'HR', 'p_value',
                'admin_censor_date', 'source_case_control_version'}
    if required - set(d.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(d.columns))}')
    d['disease'] = d['disease'].astype(str).str.strip()
    if d['disease'].nunique() > EXPECTED_DISEASES:
        raise ValueError(f'Expected no more than 630 endpoints; found {d.disease.nunique()}')
    if d['disease'].nunique() < 500:
        raise ValueError(f'Too few fitted endpoints to reproduce the supplementary figure: {d.disease.nunique()}')
    for col, expected in [('admin_censor_date', CENSOR_DATE),
                          ('source_case_control_version', SOURCE_TAG)]:
        found = set(d[col].dropna().astype(str).str.strip())
        if found != {expected}:
            raise ValueError(f'{col}: {found} (expected {expected})')

    d = d[d['MoleculeClass'].isin(RAW_CLASS_MAP)].copy()
    d['MoleculeClass'] = d['MoleculeClass'].map(RAW_CLASS_MAP)
    d['coding'] = d['coding'].astype('string').str.strip()
    if d['coding'].isna().any() or d['coding'].eq('').any():
        raise ValueError('Missing protein Olink coding')
    if d.duplicated(['disease', 'coding']).any():
        raise ValueError('Repeated disease x protein coding')

    for col in ['HR', 'p_value']:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    # Same validity criterion as the original script: all finite HR and p.
    valid = (np.isfinite(d['HR']) & d['HR'].gt(0)
             & np.isfinite(d['p_value']) & d['p_value'].between(0, 1))
    d['valid'] = valid
    d['abs_logHR'] = np.log(d['HR'].where(valid)).abs()
    d = d.merge(read_annotation(), on='disease', how='left', validate='many_to_one')
    if d['DiseaseSystem'].isna().any():
        unknown = d.loc[d['DiseaseSystem'].isna(), 'disease'].unique()[:8]
        raise ValueError(f'Missing/unrecognized ICD chapters for {unknown}')
    actual = set(d.DiseaseSystem) - EXCLUDED
    if actual != set(FULL_SYSTEMS):
        raise ValueError(f'Unexpected system set: {sorted(actual ^ set(FULL_SYSTEMS))}')
    return d.loc[d.DiseaseSystem.isin(FULL_SYSTEMS)].copy()

def disease_level_means(d):
    grouped = (d.groupby(['DiseaseSystem', 'disease', 'MoleculeClass'], as_index=False)
               .agg(n_tested=('valid', 'sum'),
                    mean_abs_logHR_all=('abs_logHR', 'mean')))
    universe = d[['DiseaseSystem', 'disease']].drop_duplicates()
    complete = universe.merge(pd.DataFrame({'MoleculeClass': CLASS_ORDER}), how='cross')
    complete = complete.merge(grouped,
                              on=['DiseaseSystem', 'disease', 'MoleculeClass'],
                              how='left', validate='one_to_one')
    complete['n_tested'] = complete['n_tested'].fillna(0).astype(int)
    return complete

def wilcoxon_36(values):
    rows = []
    for system in FULL_SYSTEMS:
        wide = (values.loc[values.DiseaseSystem.eq(system)]
                .pivot(index='disease', columns='MoleculeClass', values='mean_abs_logHR_all')
                .reindex(columns=CLASS_ORDER))
        for i, j in PAIRS:
            a, b = CLASS_ORDER[i], CLASS_ORDER[j]
            pset = wide[[a, b]].dropna()
            p = stat = np.nan
            if len(pset) >= 2:
                diff = pset[b].to_numpy(float) - pset[a].to_numpy(float)
                if np.allclose(diff, 0, atol=1e-14, rtol=1e-12):
                    p, stat = 1., 0.
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        try:
                            fit = wilcoxon(pset[a].to_numpy(float), pset[b].to_numpy(float),
                                           alternative='two-sided', zero_method='wilcox')
                            p, stat = float(fit.pvalue), float(fit.statistic)
                        except ValueError:
                            pass
            rows.append(dict(DiseaseSystem=system, class_a=a, class_b=b,
                             n_paired_diseases=len(pset), wilcoxon_W=stat,
                             p_two_sided=p))
    stats = pd.DataFrame(rows)
    assert len(stats) == 36, 'Exactly thirty-six tests required'
    good = np.isfinite(stats.p_two_sided.to_numpy(float))
    stats['FDR_BH_all_tests'] = np.nan
    if good.any():
        stats.loc[good, 'FDR_BH_all_tests'] = multipletests(
            stats.loc[good, 'p_two_sided'], method='fdr_bh')[1]
    return stats

def style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 7.5,
        'text.color': '#000000',
        'axes.labelcolor': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.unicode_minus': False,
        'savefig.facecolor': 'white',
    })

def q_label(q):
    return f'FDR={q:.1e}' if q < .001 else (f'FDR={q:.3f}' if q < .01 else f'FDR={q:.2f}')

def draw_panel(ax, vals, tests, system, ytop, rng):
    subset = vals.loc[vals.DiseaseSystem.eq(system)]
    means = []
    for i, cls in enumerate(CLASS_ORDER):
        y = subset.loc[subset.MoleculeClass.eq(cls), 'mean_abs_logHR_all'].dropna().to_numpy(float)
        if len(y) > 1 and np.std(y) > 1e-12:
            v = ax.violinplot([y], positions=[i - .11], widths=.58,
                              showmeans=False, showmedians=False,
                              showextrema=False, bw_method=.28)
            for part in v['bodies']:
                part.set_facecolor(COLORS[cls])
                part.set_edgecolor(COLORS[cls])
                part.set_linewidth(.60)
                part.set_alpha(.22)
        if len(y):
            jitter = i + .11 + rng.uniform(-.095, .095, size=len(y))
            ax.scatter(jitter, y, c=COLORS[cls], s=6, alpha=.56,
                       edgecolors='none', zorder=4)
            ax.boxplot([y], positions=[i - .11], widths=.18,
                       patch_artist=True, manage_ticks=False,
                       showfliers=False, whis=(10, 90),
                       boxprops=dict(facecolor='white', edgecolor='#40505B', lw=.7),
                       medianprops=dict(color='none', lw=0),
                       whiskerprops=dict(color='#6B7880', lw=.6),
                       capprops=dict(color='#6B7880', lw=.6))
            ax.scatter([i - .11], [float(np.mean(y))], s=18,
                       facecolors='white', edgecolors='#25343D',
                       linewidths=.8, marker='D', zorder=6)
            means.append(f'{np.mean(y):.3f}')
        else:
            means.append('NA')

    ax.set_xlim(-.47, 2.47)
    ax.set_ylim(0, ytop)
    ax.set_xticks([0, 1, 2], CLASS_SHORT)
    ax.tick_params(axis='x', labelsize=7.1, length=2.2, pad=2)
    ax.tick_params(axis='y', labelsize=7.0, length=2.2, pad=2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.grid(False)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#596772')
        ax.spines[spine].set_linewidth(.75)

    for tick, cls in zip(ax.get_xticklabels(), CLASS_ORDER):
        tick.set_color(COLORS[cls])
    for i, (cls, txt) in enumerate(zip(CLASS_ORDER, means)):
        ax.text(i, -.30, txt, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7.0, fontweight='bold',
                color=COLORS[cls], clip_on=False)

    lookup = tests.loc[tests.DiseaseSystem.eq(system)].set_index(['class_a', 'class_b'])
    sig = []
    for i, j in PAIRS:
        q = lookup.loc[(CLASS_ORDER[i], CLASS_ORDER[j]), 'FDR_BH_all_tests']
        if pd.notna(q) and q < .05:
            sig.append((i, j, float(q)))
    sig.sort(key=lambda v: (v[1]-v[0], v[0]))
    # Brackets are in axis-fraction coordinates and do not hide observations.
    for r, (i, j, q) in enumerate(sig):
        h = 1.09 + r * .16
        t = ax.get_xaxis_transform()
        ax.plot([i, i, j, j], [h-.035, h, h, h-.035],
                transform=t, color='#53636D', lw=.63, clip_on=False)
        ax.text((i+j)/2, h+.020, q_label(q), transform=t,
                ha='center', va='bottom', fontsize=7.0,
                color='black', clip_on=False)

    title = SYSTEM_LABEL.get(system, system)
    n_diseases = subset['disease'].nunique()
    ax.text(.5, 1.79, f'{title} (n={n_diseases})', transform=ax.transAxes,
            ha='center', va='bottom', fontweight='bold', fontsize=8.0,
            color='black', clip_on=False)

def draw_pdf(values, stats, pdf):
    style()
    rng = np.random.default_rng(SEED)
    obs = values.mean_abs_logHR_all.to_numpy(float)
    obs = obs[np.isfinite(obs)]
    if not len(obs):
        raise ValueError('No valid per-disease mean |log(HR)| data')
    top = max(.020, float(obs.max()) * 1.08)
    fig = plt.figure(figsize=(PAGE_W/25.4, PAGE_H/25.4), facecolor='white')
    X = lambda mm: mm/PAGE_W
    Y = lambda mm: mm/PAGE_H
    mm_axes = lambda l, b, w, h: [X(l), Y(b), X(w), Y(h)]

    # Editorial header, legend and thin divider.
    fig.text(X(14), Y(281), 'SUPPLEMENTARY FIGURE',
             fontsize=7.5, fontweight='bold', va='center')
    fig.text(X(14), Y(272), 'Protein–disease associations across ten systems',
             fontsize=11.2, fontweight='bold', va='center')
    fig.text(X(14), Y(264), 'Distribution of mean |log(HR)| for all valid proteins',
             fontsize=8.0, va='center')
    handles = [Patch(facecolor=COLORS[c], edgecolor='none', label=c)
               for c in CLASS_ORDER]
    fig.legend(handles=handles, loc='lower left', frameon=False, ncol=3,
               bbox_to_anchor=(X(14), Y(253)), bbox_transform=fig.transFigure,
               fontsize=7.5, handlelength=1.35, handletextpad=.55,
               columnspacing=1.55, borderaxespad=0)
    fig.add_artist(plt.Line2D([X(14), X(196)], [Y(250), Y(250)],
                              transform=fig.transFigure,
                              color='#CFD5D8', lw=.75))

    # 2 columns x 5 rows, identical axes widths and heights.
    # The extra vertical space above axes holds all significant brackets.
    # Axes top positions in mm: 230,189,148,107,66.
    for k, system in enumerate(SYSTEMS):
        row, col = divmod(k, 2)
        left = [16, 113][col]
        bottom = 214 - row * 41
        ax = fig.add_axes(mm_axes(left, bottom, 76, 16))
        if SHARED_Y_SCALE:
            yt = top
        else:
            arr = values.loc[values.DiseaseSystem.eq(system), 'mean_abs_logHR_all'].to_numpy(float)
            arr = arr[np.isfinite(arr)]
            yt = max(.020, float(arr.max()) * 1.08) if len(arr) else .020
        draw_panel(ax, values, stats, system, yt, rng)
        if col == 0:
            ax.set_ylabel(r'Mean $|\log(\mathrm{HR})|$', fontsize=7.2, labelpad=2)

    # A4 manuscript-style caption (only significant FDR brackets are shown).
    fig.add_artist(plt.Line2D([X(14), X(196)], [Y(35), Y(35)],
                              transform=fig.transFigure,
                              color='#CFD5D8', lw=.70))
    fig.text(X(14), Y(31),
             'Supplementary Figure S3. Protein-level associations with incident disease.',
             ha='left', va='top', fontsize=7.2, fontweight='bold')
    lines = [
        'Each point represents a disease; values are means of |log(HR)| across all valid proteins in each class, without protein-FDR selection.',

        'Violins show the distributions; boxes show IQR (whiskers: 10th–90th percentiles); white diamonds and colored numbers denote means.',

        'Two-sided paired Wilcoxon tests compare classes across diseases; BH-FDR was applied over 36 tests (12 systems × 3 pairs). Only FDR < 0.05 is labeled.',

        'Endocrine/metabolic and cardiovascular systems appear in the main figure; infections are excluded.',
    ]
    for j, line in enumerate(lines):
        fig.text(X(14), Y(26.5 - 4.3*j), line,
                 ha='left', va='top', fontsize=7.0, color='black')
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, format='pdf', pad_inches=0)
    plt.close(fig)

def main():
    raw = read_hr()
    values_all12 = disease_level_means(raw)
    stats = wilcoxon_36(values_all12)
    # Restrict only the display to ten systems; retain 36-test FDR results.
    values = values_all12.loc[values_all12.DiseaseSystem.isin(SYSTEMS)].copy()
    pdf = OUT_DIR / PDF_NAME
    draw_pdf(values, stats, pdf)
    print('Systems:', len(SYSTEMS), '| comparisons:', len(stats),
          '| FDR < 0.05:', int(stats.FDR_BH_all_tests.lt(.05).sum()))
    print('PDF:', pdf.resolve())

if __name__ == '__main__':
    main()
