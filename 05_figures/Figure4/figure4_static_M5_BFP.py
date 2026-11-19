#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4: resting-state module M5 and body fat percentage association."""
from __future__ import annotations
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 0. USER SETTINGS
PROJECT_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
HEATMAP_DIR = PROJECT_DIR / 'resting_module_phenotype'

MODULE_SCORE_FILE = HEATMAP_DIR / 'subject_module_scores.tsv'
PHENO_FILE = HEATMAP_DIR / '00_8phenotype_values_used_FatMass.tsv'
RESULT_FILE = HEATMAP_DIR / 'module_phenotype_all_tests.tsv'

OUT_DIR = PROJECT_DIR / 'Static_M5_BFP_scatter_single_clean'
PDF_NAME = 'Static_M5_BFP_scatter_single_clean.pdf'

MODULE_ID = 'M5'
PHENOTYPE = 'BFP'
FIG_WIDTH_MM = 70.0
FIG_HEIGHT_MM = 86.0
MIN_N = 10
CHECK_ATOL = 1e-6
N_BOOT = 3000
BOOT_SEED = 20261020

# Original scatter styling (kept unchanged).
POINT_FACE = '#A8C8E8'
POINT_EDGE = '#2E6EA6'
LINE_COLOR = '#2E6EA6'
BAND_COLOR = '#DCEAF7'
AXIS_COLOR = '#788590'
ZERO_COLOR = '#D8DEE5'
TEXT_COLOR = '#26333E'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 7.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

# 1. READ INPUTS
def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing input file: {path}')
    t = pd.read_csv(path, sep='\t', dtype=str, low_memory=False)
    if t.shape[1] == 1:
        t = pd.read_csv(path, sep=None, engine='python', dtype=str)
    t.columns = [str(c).replace('\ufeff', '').strip() for c in t.columns]
    return t

def load_values(path: Path, value_column: str, new_column: str) -> pd.DataFrame:
    t = read_table(path)
    aliases = {str(col).strip().lower(): col for col in t.columns}
    id_col = next((aliases[c] for c in ('subject', 'id', 'individual') if c in aliases), None)
    if id_col is None:
        raise ValueError(f'{path}: missing subject/ID/individual column')
    if value_column not in t:
        raise ValueError(f'{path}: missing {value_column}; columns={list(t.columns)}')

    out = t[[id_col, value_column]].rename(
        columns={id_col: 'subject', value_column: new_column}
    ).copy()
    out['subject'] = out['subject'].astype('string').str.strip().str.lower()
    if (out['subject'].isna().any() or out['subject'].eq('').any()
            or out['subject'].duplicated().any()):
        raise ValueError(f'{path}: invalid/duplicate participant IDs')
    out[new_column] = pd.to_numeric(out[new_column], errors='coerce')
    out[new_column] = out[new_column].replace([np.inf, -np.inf], np.nan)
    return out

def load_saved_statistics(path: Path) -> dict:
    t = read_table(path)
    mod_col = next((c for c in ('module', 'pair_id', 'module_id') if c in t), None)
    if mod_col is None or 'phenotype' not in t:
        raise ValueError(f'{path}: missing module or phenotype column')
    hit = t.loc[(t[mod_col].str.strip() == MODULE_ID) &
                (t['phenotype'].str.strip() == PHENOTYPE)]
    if len(hit) != 1:
        raise ValueError(f'{path}: expected one {MODULE_ID}/{PHENOTYPE} row, got {len(hit)}')
    row = hit.iloc[0]
    fields = ('rho', 'n', 'pvalue', 'empirical_p', 'empirical_fdr_global',
              'empirical_fdr_within_phenotype', 'perm_maxT_p')
    return {name: float(pd.to_numeric(row[name], errors='coerce'))
            for name in fields if name in t.columns}

def load_data_and_statistics():
    score = load_values(MODULE_SCORE_FILE, MODULE_ID, 'module_score')
    pheno = load_values(PHENO_FILE, f'Delta_{PHENOTYPE}', 'phenotype_delta')

    # No hidden subject intersection: every module-score subject must have a BFP row.
    missing = sorted(set(score['subject']) - set(pheno['subject']))
    if missing:
        raise ValueError(f'Phenotype data missing score subjects: {missing[:12]}')

    data = score.merge(pheno, how='left', on='subject', validate='one_to_one')
    data = data.loc[np.isfinite(data['module_score']) &
                    np.isfinite(data['phenotype_delta'])].copy()
    data = data.sort_values('subject', kind='stable').reset_index(drop=True)
    if len(data) < MIN_N:
        raise ValueError(f'Only {len(data)} complete observations; need {MIN_N}')

    x = data['phenotype_delta'].to_numpy(dtype=float)
    y = data['module_score'].to_numpy(dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        raise ValueError('BFP change or module score is constant; cannot correlate')
    rho, raw_p = spearmanr(x, y)
    rho, raw_p = float(rho), float(raw_p)

    saved = load_saved_statistics(RESULT_FILE)
    if 'n' in saved and np.isfinite(saved['n']) and int(saved['n']) != len(data):
        raise RuntimeError(f'Heatmap N={saved["n"]}, scatter N={len(data)}; input mismatch')
    if 'rho' in saved and np.isfinite(saved['rho']) and not np.isclose(
            rho, saved['rho'], atol=CHECK_ATOL, rtol=0):
        raise RuntimeError(f'Heatmap rho={saved["rho"]:.8f}; '
                           f'scatter rho={rho:.8f}; input mismatch')
    if 'pvalue' in saved and np.isfinite(saved['pvalue']) and not np.isclose(
            raw_p, saved['pvalue'], atol=CHECK_ATOL, rtol=0):
        raise RuntimeError(f'Heatmap raw P={saved["pvalue"]:.8g}; '
                           f'scatter raw P={raw_p:.8g}; input mismatch')
    stats = {
        'module_id': MODULE_ID, 'phenotype': PHENOTYPE, 'n': len(data),
        'rho': rho, 'raw_p': raw_p,
        'empirical_p': saved.get('empirical_p', np.nan),
        'empirical_fdr_global': saved.get('empirical_fdr_global', np.nan),
        'empirical_fdr_within_phenotype': saved.get(
            'empirical_fdr_within_phenotype', np.nan),
        'perm_maxT_p': saved.get('perm_maxT_p', np.nan),
        'module_score_source': str(MODULE_SCORE_FILE),
        'phenotype_source': str(PHENO_FILE),
        'statistic_source': str(RESULT_FILE),
    }
    return data, stats

# 2. LINEAR TREND AND BOOTSTRAP CONFIDENCE BAND
def bootstrap_line_ci(x, y):
    if len(x) < 3 or np.std(x) <= 1e-12:
        return None, None, None
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    rng = np.random.default_rng(BOOT_SEED)
    predictions = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = x[idx], y[idx]
        if np.std(xb) <= 1e-12:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            predictions.append(np.polyval(np.polyfit(xb, yb, 1), xx))

    fitted = np.polyval(np.polyfit(x, y, 1), xx)
    band = None
    if predictions:
        arr = np.asarray(predictions, dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr):
            band = (np.percentile(arr, 2.5, axis=0),
                    np.percentile(arr, 97.5, axis=0))
    return xx, fitted, band

# 3. DRAW: THE SAME 70 x 86 mm STYLE AS THE ORIGINAL
def draw(data: pd.DataFrame, stats: dict) -> Path:
    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4),
                     facecolor='white')

    # Three lines in the reserved header; do not overlap the axes.
    title = fig.text(.16, .966, 'M5 vs Body fat percentage',
                     fontsize=8.7, fontweight='bold', color=TEXT_COLOR,
                     ha='left', va='top')
    line1 = fig.text(.16, .917,
                     f'ρ = {stats["rho"]:+.3f}    N = {stats["n"]}',
                     fontsize=7.2, color=TEXT_COLOR, ha='left', va='top')
    empirical_p = stats['empirical_p']
    line2_str = f'Raw P = {stats["raw_p"]:.4g}'
    if np.isfinite(empirical_p):
        line2_str += f'    Perm. P = {empirical_p:.4g}'
    line2 = fig.text(.16, .868, line2_str, fontsize=7.0,
                     color=TEXT_COLOR, ha='left', va='top')

    ax = fig.add_axes([.16, .15, .80, .66])
    x = data['phenotype_delta'].to_numpy(dtype=float)
    y = data['module_score'].to_numpy(dtype=float)
    xx, yy, band = bootstrap_line_ci(x, y)
    if band is not None:
        ax.fill_between(xx, band[0], band[1], color=BAND_COLOR,
                        alpha=.72, linewidth=0, zorder=1)
    if xx is not None:
        ax.plot(xx, yy, color=LINE_COLOR, linewidth=1.3, zorder=3)
    ax.axhline(0, color=ZERO_COLOR, linewidth=.8, zorder=0)
    ax.axvline(0, color=ZERO_COLOR, linewidth=.8, zorder=0)
    ax.scatter(x, y, s=22, marker='o', facecolor=POINT_FACE,
               edgecolor=POINT_EDGE, linewidth=.7, alpha=.96,
               rasterized=False, zorder=4)

    spanx = max(float(np.ptp(x)), .2)
    spany = max(float(np.ptp(y)), .2)
    ax.set_xlim(float(x.min()) - .10 * spanx, float(x.max()) + .10 * spanx)
    ax.set_ylim(float(y.min()) - .10 * spany, float(y.max()) + .10 * spany)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(AXIS_COLOR)
        ax.spines[side].set_linewidth(.8)
    ax.tick_params(axis='both', labelsize=7, width=.7, length=3, color=AXIS_COLOR)
    ax.set_xlabel('ΔBody fat percentage (%)', fontsize=7.2,
                  color=TEXT_COLOR, labelpad=3)
    ax.set_ylabel('ΔModule expression', fontsize=7.2,
                  color=TEXT_COLOR, labelpad=3)

    # Prevent header text from entering the data region.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    header_bottom = min(t.get_window_extent(renderer).y0 for t in (title, line1, line2))
    plot_top = ax.get_window_extent(renderer).y1
    if header_bottom <= plot_top:
        plt.close(fig)
        raise RuntimeError('The header overlaps the data axes')

    pdf = OUT_DIR / PDF_NAME
    fig.savefig(pdf, format='pdf', facecolor='white',
                metadata={'Title': 'Static M5 and change in body fat percentage'})
    fig.savefig(pdf.with_suffix('.png'), dpi=300, facecolor='white')
    plt.close(fig)
    return pdf.resolve()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, stats = load_data_and_statistics()
    data.to_csv(OUT_DIR / 'Static_M5_BFP_subject_values.tsv',
                sep='\t', index=False)
    pd.DataFrame([stats]).to_csv(OUT_DIR / 'Static_M5_BFP_correlation_summary.tsv',
                                 sep='\t', index=False)
    print(draw(data, stats))  # Only print the absolute PDF path.

if __name__ == '__main__':
    main()
