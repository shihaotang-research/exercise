#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 2A: paired week 0/week 8 phenotype raincloud plots."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde, ttest_rel

# 1. User configuration
INPUT_FILE = Path('Total_before_after.xlsx')
OUTPUT_PDF = Path('W0_W8_RainCloud_8phenotypes_A4width_final.pdf')

COLORS = {'week0': '#2E8AC4', 'week8': '#E64B87'}
ORDER = ('week0', 'week8')

PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 124.5  # ~ original 4.90 inches; A4-wide Figure 2A

FONT_SIZE = 8.0
Y_LABEL_SIZE = 8.3
TICK_SIZE = 7.7
P_VALUE_SIZE = 8.0

VIOLIN_MAX_WIDTH = 0.265
VIOLIN_RIGHT_OFFSET = 0.135
VIOLIN_ALPHA = 0.82
VIOLIN_OUTLINE_COLOR = '#4A5056'
VIOLIN_LINE_WIDTH = 0.52

BOX_WIDTH = 0.145
BOX_LINE_WIDTH = 0.62
POINT_AREA = 8.5             # points^2 (area of each dot)
POINT_ALPHA = 0.88
JITTER_CENTER = 0.155        # place cloud points to right of box
JITTER_HALF_WIDTH = 0.072
RANDOM_SEED = 20261020

SPINE_COLOR = '#5B6269'
SPINE_LINE_WIDTH = 0.60
SHOW_FIGURE = False  # True to display in a Notebook; PDFs always saved

VARIABLES = [
    ('BMI', 'BMI'),
    ('waist', 'Waist circumference'),
    ('TOTAL_FAT_MASS', 'Total body fat mass'),
    ('BFP', 'Body fat percentage'),
    ('VO2_kg_peak', 'Peak VO₂max/kg'),
    ('O2pluse_peak', 'Peak O₂ pulse'),
    ('Rest_HR', 'Resting heart rate'),
    ('VCMAX', 'Vital capacity max'),
]

# 2. Style and inputs
def configure_style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Arial', 'Liberation Sans', 'DejaVu Sans')
                 if f in installed), 'DejaVu Sans')

    plt.rcParams.update({
        'font.family': font,
        'font.size': FONT_SIZE,
        'text.color': '#000000',
        'axes.labelcolor': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'axes.edgecolor': SPINE_COLOR,
        'axes.linewidth': SPINE_LINE_WIDTH,
        'figure.facecolor': 'white',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
    })

def load_data(path):
    if not path.is_file():
        raise FileNotFoundError(f'Input Excel file not found: {path.resolve()}')

    df = pd.read_excel(path)
    required = {'ID', 'group'} | {var for var, _ in VARIABLES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    df = df.copy()
    df['ID'] = df['ID'].astype('string').str.strip().str.lower()
    df['group'] = df['group'].astype('string').str.strip().str.lower()
    df = df.loc[df['group'].isin(ORDER)].copy()

    if df.empty:
        raise ValueError('No W0/W8 rows left after filtering.')
    if df['ID'].isna().any() or df['ID'].eq('').any():
        raise ValueError('Missing participant IDs in input data.')

    # Paired analysis expects at most one row per subject and week.
    duplicated = df.duplicated(['ID', 'group'], keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, ['ID', 'group']].head(8).to_dict('records')
        raise ValueError(f'Duplicate subject/week rows: {examples}')

    for var, _ in VARIABLES:
        df[var] = pd.to_numeric(df[var], errors='coerce')
        # Treat +/- infinity as invalid rather than passing it to KDE / t-test.
        df[var] = df[var].replace([np.inf, -np.inf], np.nan)

    # Keep all eight measurements in their ORIGINAL units and scales.
    # In particular, TOTAL_FAT_MASS is the Excel column, not WBTOT_FAT.
    return df

# 3. Statistics
def paired_pvalue(df, variable):
    paired = (df[['ID', 'group', variable]]
              .dropna(subset=[variable])
              .pivot(index='ID', columns='group', values=variable))
    if not set(ORDER).issubset(paired.columns):
        return np.nan

    paired = paired.loc[:, list(ORDER)].dropna()
    if len(paired) < 2:
        return np.nan
    if np.array_equal(paired['week0'].to_numpy(), paired['week8'].to_numpy()):
        return 1.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(ttest_rel(paired['week0'], paired['week8']).pvalue)

def format_pvalue(p):
    if not np.isfinite(p):
        return 'P = NA'
    return f'P = {p:.2e}'

# 4. RainCloud primitives
def draw_half_violin(ax, samples, center, color):
    """True one-sided Gaussian KDE; no full-violin overlap with the box."""
    v = np.asarray(samples, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3 or np.ptp(v) < 1e-12:
        return

    spread = float(np.ptp(v))
    y_grid = np.linspace(v.min() - 0.05 * spread,
                         v.max() + 0.05 * spread, 240)
    try:
        # Moderate smoothing: avoid overly bumpy empirical KDE silhouettes.
        kde = gaussian_kde(v, bw_method=lambda obj: obj.scotts_factor() * 0.94)
        density = kde(y_grid)
    except np.linalg.LinAlgError:
        return

    maximum = float(np.max(density))
    if not np.isfinite(maximum) or maximum <= 0:
        return
    widths = VIOLIN_MAX_WIDTH * density / maximum
    right_x = center - VIOLIN_RIGHT_OFFSET
    left_x = right_x - widths
    polygon_x = np.r_[left_x, np.full_like(y_grid, right_x)[::-1]]
    polygon_y = np.r_[y_grid, y_grid[::-1]]

    ax.fill(polygon_x, polygon_y,
            facecolor=color, alpha=VIOLIN_ALPHA,
            edgecolor=VIOLIN_OUTLINE_COLOR,
            linewidth=VIOLIN_LINE_WIDTH,
            joinstyle='round', zorder=2)

def draw_iqr_box(ax, samples, center):
    """White box, visible median and conventional 1.5*IQR whiskers."""
    plot = ax.boxplot(
        samples, positions=[center], widths=BOX_WIDTH,
        patch_artist=True, showfliers=False, manage_ticks=False,
        whis=1.5,
        boxprops={'facecolor': 'white', 'edgecolor': '#51575C',
                  'linewidth': BOX_LINE_WIDTH},
        whiskerprops={'color': '#51575C', 'linewidth': BOX_LINE_WIDTH},
        capprops={'color': '#51575C', 'linewidth': BOX_LINE_WIDTH},
        medianprops={'color': '#222222', 'linewidth': 1.0},
        zorder=5,
    )
    for patch in plot['boxes']:
        patch.set_zorder(5)
    for median in plot['medians']:
        median.set_zorder(7)

def draw_observations(ax, samples, center, color, rng):
    """All individuals are shown; jitter is horizontal only."""
    samples = np.asarray(samples, dtype=float)
    x_offsets = JITTER_CENTER + rng.uniform(-JITTER_HALF_WIDTH,
                                             JITTER_HALF_WIDTH, size=len(samples))
    ax.scatter(np.full(len(samples), center) + x_offsets, samples,
               s=POINT_AREA, marker='o', color=color,
               alpha=POINT_ALPHA, linewidths=0, rasterized=False,
               zorder=6)

def set_axis_range(ax, values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    lo, hi = float(np.min(vals)), float(np.max(vals))
    spread = hi - lo
    if spread <= 0:
        spread = max(abs(lo) * 0.12, 1.0)
    ax.set_ylim(lo - 0.105 * spread, hi + 0.105 * spread)
    ax.set_xlim(-0.52, 1.46)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))

# 5. Each phenotype panel
def draw_panel(ax, df, variable, ylabel, panel_seed):
    rng = np.random.default_rng(panel_seed)
    values_all = []

    for x, week in enumerate(ORDER):
        v = df.loc[df['group'].eq(week), variable].dropna().to_numpy(dtype=float)
        if len(v) == 0:
            raise ValueError(f'No valid values for {variable}, {week}')
        values_all.append(v)
        draw_half_violin(ax, v, x, COLORS[week])
        draw_iqr_box(ax, v, x)
        draw_observations(ax, v, x, COLORS[week], rng)

    set_axis_range(ax, np.concatenate(values_all))

    # Two-sided paired t-test; W0/W8 observations are matched by ID.
    ax.text(.5, 1.035, format_pvalue(paired_pvalue(df, variable)),
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=P_VALUE_SIZE, fontweight='normal',
            color='black', clip_on=False)

    ax.set_xticks([0, 1], ['W0', 'W8'])
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=Y_LABEL_SIZE, color='black', labelpad=3)
    ax.tick_params(axis='x', labelsize=TICK_SIZE,
                   length=2.7, width=.65, pad=2.0, colors='black')
    ax.tick_params(axis='y', labelsize=TICK_SIZE - .2,
                   length=2.7, width=.65, pad=1.7, colors='black')

    for spine in ax.spines.values():
        spine.set_visible(True)  # preserve the reference's four-sided frames
        spine.set_linewidth(SPINE_LINE_WIDTH)
        spine.set_color(SPINE_COLOR)
    ax.grid(False)

# 6. Draw and export
def main():
    configure_style()
    df = load_data(INPUT_FILE)

    figure = plt.figure(figsize=(PAGE_WIDTH_MM / 25.4,
                                 PAGE_HEIGHT_MM / 25.4), facecolor='white')
    axes = figure.subplots(2, 4)

    for i, (ax, (variable, label)) in enumerate(zip(axes.flat, VARIABLES)):
        draw_panel(ax, df, variable, label, RANDOM_SEED + i * 17)

    # Spacing tuned for A4-width Figure 2A while leaving room for P labels.
    figure.subplots_adjust(
        left=0.083, right=0.982,
        bottom=0.105, top=0.920,
        wspace=0.67, hspace=0.70,
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PDF, format='pdf', dpi=300,
                   facecolor='white', metadata={
                       'Title': 'W0 vs W8: eight physiological phenotypes',
                       'Creator': 'Matplotlib',
                   })
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)
    print(OUTPUT_PDF.resolve())

if __name__ == '__main__':
    main()
