#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5: acute exercise-response module M3 and fitness association."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr

# 1. USER SETTINGS
PROJECT_DIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
MAIN_DIR = PROJECT_DIR / 'acute_module_phenotype'
OUT_DIR = PROJECT_DIR / 'Dynamic_M3_VO2_scatter_Gamma1p2_MAIN'
MODULE = 'M3'
PHENOTYPE = 'VO2'

# Original M5 reference was 70 x 86 mm; slightly narrower / taller here.
FIG_WIDTH_MM = 68.0
FIG_HEIGHT_MM = 90.0
DPI = 300
BOOTSTRAP_REPS = 3000
BOOTSTRAP_SEED = 20261026
MIN_SUBJECTS = 10

# Very strict consistency with saved full-precision MAIN heatmap values.
MATCH_RHO_TOL = 1e-7
MATCH_RAW_P_TOL = 1e-6
# Visual screenshot showed M3 rho rounded to +0.31.
EXPECTED_HEATMAP_RHO = 0.31
CHECK_SCREENSHOT = True
SCREENSHOT_ROUND_TOL = 0.006

POINT_COLOR = '#A8C8E8'
POINT_EDGE = '#2E6EA6'
LINE_COLOR = '#2E6EA6'
CI_COLOR = '#DCEAF7'
ZERO_COLOR = '#D8DEE5'
SPINE_COLOR = '#788590'
TEXT_COLOR = '#26333E'

def configure_style():
    # Arial is preferred. Fallback for headless Linux installs lacking Arial.
    available = {f.name for f in font_manager.fontManager.ttflist}
    face = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans')
                 if f in available), 'DejaVu Sans')
    plt.rcParams.update({
        'font.family': face,
        'font.size': 7.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.unicode_minus': False,
    })

def read_main(filename):
    path = MAIN_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'MAIN analysis input missing: {path}\n'
            'Run the gamma=1.2 MAIN heatmap script and do not replace it '
            'with CORRECTED_results files.'
        )
    data = pd.read_csv(path, sep='\t', low_memory=False)
    data.columns = [str(col).replace('\ufeff', '').strip() for col in data.columns]
    return data

def require_columns(data, names, label):
    missing = sorted(set(names) - set(data.columns))
    if missing:
        raise ValueError(f'{label}: missing columns: {missing}')

def norm_ids(series):
    return series.astype('string').str.strip().str.lower()

def validate_ids(series, label):
    if series.isna().any() or series.eq('').any() or series.duplicated().any():
        raise ValueError(f'{label} contains blank/missing/duplicate subjects')

def load_and_validate():
    mapping = read_main('module_scoring_groups.tsv')
    scores = read_main('subject_module_scores.tsv')
    phenotype = read_main('00_8phenotype_values_used_FatMass.tsv')
    tests = read_main('module_phenotype_all_tests.tsv')

    require_columns(mapping, ['module', 'kind', 'scoring_nodes'], 'module mapping')
    require_columns(scores, ['subject', MODULE], 'module subject scores')
    require_columns(phenotype, ['subject', f'Delta_{PHENOTYPE}'], 'phenotype')
    require_columns(tests,
                    ['module', 'phenotype', 'rho', 'pvalue', 'n', 'empirical_p'],
                    'MAIN heatmap tests')

    mapping['module'] = mapping['module'].astype(str).str.strip()
    tests['module'] = tests['module'].astype(str).str.strip()
    tests['phenotype'] = tests['phenotype'].astype(str).str.strip()
    if mapping['module'].duplicated().any():
        raise ValueError('Duplicated MAIN module IDs in module_scoring_groups.tsv')
    if tests.duplicated(['module', 'phenotype']).any():
        raise ValueError('Duplicated tests for the same module and phenotype')

    group = mapping.loc[mapping['module'].eq(MODULE)]
    if len(group) != 1:
        raise ValueError(f'Expected exactly one MAIN module mapping for {MODULE}')
    group = group.iloc[0]
    if str(group['kind']).strip() != 'Matched':
        raise ValueError(f'{MODULE} is kind={group["kind"]!r}, not Matched')

    stat_rows = tests.loc[tests['module'].eq(MODULE)
                          & tests['phenotype'].eq(PHENOTYPE)]
    if len(stat_rows) != 1:
        raise ValueError(f'Expected exactly one MAIN result for {MODULE} vs {PHENOTYPE}')
    saved = stat_rows.iloc[0]

    # Validate MAIN settings from saved metadata, if present.
    for field, expected in [('resolution', 1.2), ('min_size', 10),
                            ('weight_mode', 'weighted')]:
        if field not in tests.columns:
            continue
        values = tests[field].dropna()
        if len(values) == 0:
            continue
        if field == 'weight_mode':
            okay = values.astype(str).str.strip().str.lower().eq(expected).all()
        else:
            nums = pd.to_numeric(values, errors='coerce').to_numpy(float)
            okay = np.isfinite(nums).all() and np.isclose(nums, expected).all()
        if not okay:
            raise RuntimeError(f'MAIN {field} differs from {expected!r}')

    scores['subject'] = norm_ids(scores['subject'])
    phenotype['subject'] = norm_ids(phenotype['subject'])
    validate_ids(scores['subject'], 'module scores')
    validate_ids(phenotype['subject'], 'phenotype values')

    # Same subject matching and phenotype validity mask as MAIN heatmap.
    x = phenotype[['subject', f'Delta_{PHENOTYPE}']].copy()
    y = scores[['subject', MODULE]].copy()
    merged = y.merge(x, on='subject', how='left', validate='one_to_one', indicator=True)
    if not merged['_merge'].eq('both').all():
        raise ValueError('Missing MAIN phenotype IDs for one or more module-scored subjects')
    merged = merged.drop(columns='_merge')
    merged[MODULE] = pd.to_numeric(merged[MODULE], errors='coerce')
    merged[f'Delta_{PHENOTYPE}'] = pd.to_numeric(
        merged[f'Delta_{PHENOTYPE}'], errors='coerce')
    merged = merged.loc[np.isfinite(merged[f'Delta_{PHENOTYPE}'])].copy()
    if not np.isfinite(merged[MODULE]).all():
        raise ValueError('Nonfinite M3 scores in subjects with finite phenotype')
    if len(merged) < MIN_SUBJECTS:
        raise ValueError(f'Only {len(merged)} valid subjects (minimum {MIN_SUBJECTS})')

    merged = merged.rename(columns={
        MODULE: 'module_score', f'Delta_{PHENOTYPE}': 'phenotype_delta'
    }).sort_values('subject').reset_index(drop=True)
    x_vals = merged['phenotype_delta'].to_numpy(float)
    y_vals = merged['module_score'].to_numpy(float)
    if np.std(x_vals) < 1e-12 or np.std(y_vals) < 1e-12:
        raise ValueError('Cannot calculate Spearman: near-constant scores or phenotype')

    rho, raw_p = spearmanr(y_vals, x_vals)
    rho = float(rho)
    raw_p = float(raw_p)
    ref_rho = float(saved['rho'])
    ref_p = float(saved['pvalue'])
    ref_n = int(saved['n'])
    if (not np.isfinite(rho)
            or abs(rho - ref_rho) > MATCH_RHO_TOL
            or abs(raw_p - ref_p) > MATCH_RAW_P_TOL
            or len(merged) != ref_n):
        raise RuntimeError(
            f'M3 values DO NOT match MAIN heatmap: recalculated '
            f'rho={rho:+.6f}, raw P={raw_p:.6g}, N={len(merged)}; '
            f'saved rho={ref_rho:+.6f}, P={ref_p:.6g}, N={ref_n}. '
            'Ensure files are from the SAME MAIN run.'
        )
    if CHECK_SCREENSHOT and abs(rho - EXPECTED_HEATMAP_RHO) > SCREENSHOT_ROUND_TOL:
        raise RuntimeError(
            f'MAIN rho={rho:+.6f} does not match screenshot M3≈'
            f'{EXPECTED_HEATMAP_RHO:+.2f}. Check which heatmap is final; '
            'do not override the actual rho just to match a figure.'
        )

    stats = {
        'module': MODULE,
        'phenotype': PHENOTYPE,
        'n': len(merged),
        'rho': rho,
        'raw_p_recomputed': raw_p,
        'raw_p_saved': ref_p,
        'empirical_p': float(saved['empirical_p']),
        'empirical_fdr_within_phenotype': pd.to_numeric(
            pd.Series([saved.get('empirical_fdr_within_phenotype', np.nan)]),
            errors='coerce').iloc[0],
        'empirical_fdr_global': pd.to_numeric(
            pd.Series([saved.get('empirical_fdr_global', np.nan)]),
            errors='coerce').iloc[0],
        'kind': str(group['kind']),
        'W0_module': str(group.get('W0_module', '')),
        'W8_module': str(group.get('W8_module', '')),
        'n_scoring_nodes': len([v for v in str(group['scoring_nodes']).split(';') if v.strip()]),
        'score_file': str(MAIN_DIR / 'subject_module_scores.tsv'),
        'result_file': str(MAIN_DIR / 'module_phenotype_all_tests.tsv'),
    }
    return merged, stats

def bootstrap_ols_ci(x, y):
    grid = np.linspace(float(x.min()), float(x.max()), 160)
    fitted = np.polyval(np.polyfit(x, y, deg=1), grid)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    predictions = []
    for _ in range(BOOTSTRAP_REPS):
        ix = rng.integers(0, len(x), len(x))
        xx, yy = x[ix], y[ix]
        if np.std(xx) < 1e-12:
            continue
        predictions.append(np.polyval(np.polyfit(xx, yy, deg=1), grid))
    if len(predictions) < 0.90 * BOOTSTRAP_REPS:
        raise RuntimeError('Too few valid bootstrap resamples to calculate 95% CI')
    ci_low, ci_high = np.percentile(np.asarray(predictions), [2.5, 97.5], axis=0)
    return grid, fitted, ci_low, ci_high

def plot_one(data, stat, pdf, png):
    x = data['phenotype_delta'].to_numpy(float)
    y = data['module_score'].to_numpy(float)
    grid, fitted, low, high = bootstrap_ols_ci(x, y)
    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4),
                     facecolor='white')
    # Reserve three separate header lines above the plot, no overlapping points.
    title = fig.text(.16, .968, 'M3 vs ΔVO₂max', fontsize=8.7,
                     fontweight='bold', color=TEXT_COLOR, va='top', ha='left')
    line1 = fig.text(.16, .919, f'ρ = {stat["rho"]:+.3f}    N = {stat["n"]}',
                     fontsize=7.3, color=TEXT_COLOR, va='top', ha='left')
    line2 = fig.text(.16, .871,
                     f'Raw P = {stat["raw_p_recomputed"]:.4g}    '
                     f'Perm. P = {stat["empirical_p"]:.4g}',
                     fontsize=7.0, color=TEXT_COLOR, va='top', ha='left')

    ax = fig.add_axes([.16, .15, .80, .66])
    ax.axhline(0, color=ZERO_COLOR, linewidth=.8, zorder=0)
    ax.axvline(0, color=ZERO_COLOR, linewidth=.8, zorder=0)
    ax.fill_between(grid, low, high, color=CI_COLOR,
                    alpha=.72, edgecolor='none', zorder=1)
    ax.plot(grid, fitted, color=LINE_COLOR, linewidth=1.3, zorder=2)
    ax.scatter(x, y, s=22, marker='o', facecolor=POINT_COLOR,
               edgecolor=POINT_EDGE, linewidth=.7, alpha=.96,
               rasterized=False, zorder=4)

    x_span = max(float(np.ptp(x)), .2)
    y_span = max(float(np.ptp(y)), .2)
    ax.set_xlim(x.min() - .10 * x_span, x.max() + .10 * x_span)
    ax.set_ylim(y.min() - .10 * y_span, y.max() + .10 * y_span)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_linewidth(.8)
        ax.spines[side].set_color(SPINE_COLOR)
    ax.tick_params(axis='both', labelsize=7.0, width=.7, length=3,
                   color=SPINE_COLOR)
    ax.set_xlabel('ΔVO₂max', fontsize=7.2, color=TEXT_COLOR, labelpad=3)
    ax.set_ylabel('Mean module ΔAbsAUC', fontsize=7.2,
                  color=TEXT_COLOR, labelpad=3)

    # Check that statistics stay outside the data axes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bottom_of_header = min(t.get_window_extent(renderer).y0
                           for t in (title, line1, line2))
    top_of_axes = ax.get_window_extent(renderer).y1
    if bottom_of_header <= top_of_axes:
        plt.close(fig)
        raise RuntimeError('Header overlaps the axes: adjust layout spacing')
    fig.savefig(pdf, format='pdf', facecolor='white',
                metadata={'Title': 'Dynamic M3 vs Delta VO2, gamma1p2 MAIN'})
    fig.savefig(png, format='png', dpi=DPI, facecolor='white')
    plt.close(fig)

def main():
    configure_style()
    data, stat = load_and_validate()
    # Write outputs only after all consistency checks have passed.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DIR / '01_M3_VO2_subject_values.tsv', sep='\t', index=False)
    pd.DataFrame([stat]).to_csv(
        OUT_DIR / '02_M3_VO2_MAIN_statistics.tsv', sep='\t', index=False)
    pdf = OUT_DIR / 'Dynamic_M3_DeltaAbsAUC_vs_DeltaVO2_MAIN_68x90.pdf'
    png = pdf.with_suffix('.png')
    plot_one(data, stat, pdf, png)
    print(pdf.resolve())

if __name__ == '__main__':
    main()
