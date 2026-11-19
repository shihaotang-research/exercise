"""Figure 6B/C: disease-level protein breadth and |log(HR)| comparisons.
Use protein FDR_global and 36 system-level paired Wilcoxon comparisons."""
from __future__ import annotations
import re
import warnings
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
PROJECT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
UKB_DIR = Path(os.environ.get("MULTIOMICS_UKB_DIR", str(Path(__file__).resolve().parents[2] / "data" / "ukb")))
HR_FILE = PROJECT / 'ukb_incident_cox' / 'incident_protein_cox_results.tsv'
ANNOTATION_FILES = [PROJECT / 'disease_definitions.tsv', UKB_DIR / 'disease_definitions.tsv']
OUT_DIR = PROJECT / 'ukb_prediction_analysis' / '52_Figure6BC_DiseaseLevel_Boxplot_200x105mm'
EXPECTED_NONINFECTIOUS_ENDPOINTS = 630
EXPECTED_CENSOR = '2023-11-30'
EXPECTED_SOURCE = 'USER_SUPPLIED_630'
GLOBAL_PROTEIN_FDR = 0.05
LOGHR_MODE = 'absolute'
WHISKER_IQR = 1.5
SHOW_DISEASE_POINTS = False
DOT_ALPHA = 0.35
FIG_W_MM, FIG_H_MM = (200.0, 105.0)
PDF_NAME = 'Figure6BC_DiseaseLevel_Boxplot_200x105mm.pdf'
CLASSES = ('Baseline-only', 'Dual-response', 'Dynamic-only')
COLORS = {'Baseline-only': '#4C95BC', 'Dual-response': '#55A96E', 'Dynamic-only': '#E57C5E'}
SYSTEM_ORDER = ('Endocrine/metabolic', 'Circulatory', 'Respiratory', 'Mental/behavioural', 'Blood/immune',
    'Digestive', 'Skin/subcutaneous', 'Musculoskeletal', 'Nervous system', 'Genitourinary', 'Neoplasms', 'Eye/ear')
SYSTEM_DISPLAY = {'Endocrine/metabolic': 'Endocrine/metabolic', 'Circulatory': 'Cardiovascular',
    'Respiratory': 'Respiratory', 'Mental/behavioural': 'Mental/behavioral', 'Blood/immune': 'Blood/immune',
    'Digestive': 'Digestive', 'Skin/subcutaneous': 'Skin/subcutaneous', 'Musculoskeletal': 'Musculoskeletal',
    'Nervous system': 'Nervous system', 'Genitourinary': 'Genitourinary', 'Neoplasms': 'Neoplasms', 'Eye/ear': 'Eye/ear'}
ROOT_TO_SYSTEM = {'Chapter I Certain infectious and parasitic diseases': 'Infectious/parasitic',
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
    'Chapter XIV Diseases of the genitourinary system': 'Genitourinary'}

def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans',
        'DejaVu Sans'], 'font.size': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'text.color': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.58,
        'axes.edgecolor': '#5A6972', 'axes.unicode_minus': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def clean(s):
    return re.sub('\\s+', ' ', str(s).strip())

def read_annotation():
    path = next((p for p in ANNOTATION_FILES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('Cannot locate disease_definitions.tsv: ' + str(ANNOTATION_FILES))
    a = pd.read_csv(path, sep='\t', dtype=str, low_memory=False)
    col = {clean(k).lower(): k for k in a.columns}
    if {'disease code', 'disease', 'root'}.issubset(col):
        a = a.rename(columns={col['disease code']: 'disease', col['disease']: 'DiseaseLabel', col['root']: 'Root'})
    elif {'disease', 'diseaselabel', 'root'}.issubset(col):
        a = a.rename(columns={col['disease']: 'disease', col['diseaselabel']: 'DiseaseLabel', col['root']: 'Root'})
    else:
        a = pd.read_csv(path, sep='\t', header=None, dtype=str, low_memory=False)
        if a.shape[1] < 3:
            raise ValueError('disease_definitions.tsv must have at least three columns')
        a = a.iloc[:, :3].copy()
        a.columns = ['disease', 'DiseaseLabel', 'Root']
    a = a[['disease', 'DiseaseLabel', 'Root']].dropna(subset=['disease', 'Root']).copy()
    for colname in ('disease', 'DiseaseLabel', 'Root'):
        a[colname] = a[colname].map(clean)
    if a.groupby('disease').Root.nunique().gt(1).any():
        raise ValueError('Conflicting ICD chapters for same disease')
    a = a.drop_duplicates('disease')
    a['DiseaseSystem'] = a.Root.str.lower().map({k.lower(): v for k, v in ROOT_TO_SYSTEM.items()})
    return a[['disease', 'DiseaseLabel', 'DiseaseSystem']]

def read_hr():
    if not HR_FILE.is_file():
        raise FileNotFoundError('HR source missing: ' + str(HR_FILE))
    d = pd.read_csv(HR_FILE, sep='\t', low_memory=False)
    required = {'disease', 'coding', 'MoleculeClass', 'HR', 'p_value', 'FDR_global'}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError('HR table missing columns: ' + ', '.join(missing))
    for c, wanted in [('admin_censor_date', EXPECTED_CENSOR), ('source_case_control_version', EXPECTED_SOURCE)]:
        if wanted is not None:
            if c not in d:
                raise ValueError(f'Missing source validation column: {c}')
            vals = set(d[c].dropna().astype(str).str.strip())
            if vals != {wanted}:
                raise ValueError(f'{c}: {vals}; expected only {wanted}')
    d['disease'] = d['disease'].astype(str).str.strip()
    d['coding'] = d['coding'].astype('string').str.strip()
    d['MoleculeClass'] = d['MoleculeClass'].replace({'Trend-only': 'Dynamic-only'})
    if d['MoleculeClass'].isna().any() or (~d['MoleculeClass'].isin(CLASSES)).any():
        bad = d.loc[~d['MoleculeClass'].isin(CLASSES), 'MoleculeClass'].unique()
        raise ValueError(f'Unexpected molecule class labels: {bad}')
    d = d.merge(read_annotation(), on='disease', how='left', validate='many_to_one')
    if d.DiseaseSystem.isna().any():
        ex = d.loc[d.DiseaseSystem.isna(), 'disease'].unique()[:8]
        raise ValueError(f'Unknown disease-system assignments: {ex}')
    orig_n = d.disease.nunique()
    n_infection = d.loc[d.DiseaseSystem.eq('Infectious/parasitic'), 'disease'].nunique()
    d = d.loc[d.DiseaseSystem.isin(SYSTEM_ORDER)].copy()
    if d.disease.nunique() != EXPECTED_NONINFECTIOUS_ENDPOINTS:
        raise ValueError(f'Expected {EXPECTED_NONINFECTIOUS_ENDPOINTS} noninfectious endpoints; found {d.disease.nunique()} (source {orig_n}, infections {n_infection})')
    if set(d.DiseaseSystem) != set(SYSTEM_ORDER):
        raise ValueError('Not all 12 noninfectious disease systems are present')
    if d.coding.isna().any() or d.coding.eq('').any():
        raise ValueError('Empty coding in HR table')
    if d.duplicated(['disease', 'coding']).any():
        raise ValueError('Duplicate disease × coding records; check upstream source')
    for c in ('HR', 'p_value', 'FDR_global'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    p, q, hr = [d[c].to_numpy(float) for c in ('p_value', 'FDR_global', 'HR')]
    valid_p = np.isfinite(p) & (p >= 0) & (p <= 1)
    valid_q = np.isfinite(q) & (q >= 0) & (q <= 1)
    if (np.isfinite(p) & ~valid_p).any() or (np.isfinite(q) & ~valid_q).any():
        raise ValueError('Invalid p_value / FDR_global outside [0,1]')
    d['valid'] = np.isfinite(hr) & (hr > 0) & valid_p
    d['significant'] = d['valid'] & valid_q & (q < GLOBAL_PROTEIN_FDR)
    d['logHR'] = np.log(d['HR'].where(d['valid']))
    d['magnitude'] = d.logHR.abs() if LOGHR_MODE == 'absolute' else d.logHR
    print(f'[Input] {orig_n} source endpoints; {n_infection} infection endpoints excluded; {d.disease.nunique()} retained')
    print(f'[QC] Valid fits: {d.valid.sum():,}; original FDR < 0.05 fits: {d.significant.sum():,}')
    return d

def aggregate_data(d):
    key = ['DiseaseSystem', 'disease', 'MoleculeClass']
    agg = d.groupby(key, as_index=False).agg(n_tested=('valid', 'sum'), n_significant=('significant', 'sum'),
        disease_mean_magnitude=('magnitude', 'mean'))
    universe = d[['DiseaseSystem', 'disease']].drop_duplicates().merge(pd.DataFrame({'MoleculeClass': CLASSES}),
        how='cross')
    dis = universe.merge(agg, on=key, how='left', validate='one_to_one')
    dis[['n_tested', 'n_significant']] = dis[['n_tested', 'n_significant']].fillna(0).astype(int)
    dis['significant_pct'] = np.where(dis.n_tested.gt(0), 100 * dis.n_significant / dis.n_tested, np.nan)
    bsys = dis.groupby(['DiseaseSystem', 'MoleculeClass'], as_index=False).agg(n_diseases=('disease', 'nunique'),
        n_valid_diseases=('significant_pct', 'count'), median_significant_pct=('significant_pct', 'median'))
    return (dis, bsys)

def summarize_c(dis):
    """Describe across-disease distributions for each system and protein class.

    Each observation is the mean of valid protein-specific |log(HR)| values
    in ONE disease/class. All diseases get equal weight in the distribution.
    Unlike the previous pooled-mean plot, no weighting by the number of
    successfully fitted protein-disease pairs is applied at the system level.
    """
    rows = []
    for system in SYSTEM_ORDER:
        for cls in CLASSES:
            x = dis.loc[dis.DiseaseSystem.eq(system) & dis.MoleculeClass.eq(cls) & dis.n_tested.gt(0), ['disease',
                'n_tested', 'disease_mean_magnitude']].copy()
            vals = x['disease_mean_magnitude'].to_numpy(float)
            vals = np.sort(vals[np.isfinite(vals)])
            if not len(vals):
                rows.append(dict(DiseaseSystem=system, MoleculeClass=cls, n_valid_diseases=0,
                    n_valid_protein_disease_pairs=0, mean_disease_level=np.nan, median=np.nan, q1=np.nan, q3=np.nan,
                    whisker_low=np.nan, whisker_high=np.nan, n_outliers=0))
                continue
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            inside = vals[(vals >= q1 - WHISKER_IQR * iqr) & (vals <= q3 + WHISKER_IQR * iqr)]
            whislo = float(inside.min()) if len(inside) else float(q1)
            whishi = float(inside.max()) if len(inside) else float(q3)
            rows.append(dict(DiseaseSystem=system, MoleculeClass=cls, n_valid_diseases=len(vals),
                n_valid_protein_disease_pairs=int(x['n_tested'].sum()), mean_disease_level=float(vals.mean()),
                median=float(median), q1=float(q1), q3=float(q3), whisker_low=whislo, whisker_high=whishi,
                n_outliers=int(((vals < whislo) | (vals > whishi)).sum())))
    return pd.DataFrame(rows)

def paired_tests(dis):
    pairs = [(CLASSES[0], CLASSES[1]), (CLASSES[1], CLASSES[2]), (CLASSES[0], CLASSES[2])]
    records = []
    for sys in SYSTEM_ORDER:
        wide = dis.loc[dis.DiseaseSystem.eq(sys)].pivot(index='disease', columns='MoleculeClass',
            values='disease_mean_magnitude').reindex(columns=CLASSES)
        for a, b in pairs:
            pair = wide[[a, b]].dropna()
            n = len(pair)
            delta = pair[b].to_numpy(float) - pair[a].to_numpy(float)
            p, w = (np.nan, np.nan)
            if n >= 2:
                if np.allclose(delta, 0, atol=1e-14, rtol=1e-12):
                    p, w = (1.0, 0.0)
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            res = wilcoxon(pair[a], pair[b], alternative='two-sided', zero_method='wilcox')
                        w, p = (float(res.statistic), float(res.pvalue))
                    except ValueError:
                        pass
            records.append(dict(DiseaseSystem=sys, class_a=a, class_b=b, n_paired_diseases=n, W=w, p_two_sided=p,
                median_difference_b_minus_a=float(np.median(delta)) if n else np.nan))
    tests = pd.DataFrame(records)
    assert len(tests) == 36
    p = tests.p_two_sided.to_numpy(float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p) & (p >= 0) & (p <= 1)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    tests['FDR_BH_36'] = q
    tests['significant'] = tests.FDR_BH_36.lt(0.05)
    return tests

def row_order(bsys):
    mat = bsys.pivot(index='DiseaseSystem', columns='MoleculeClass',
        values='median_significant_pct').reindex(index=SYSTEM_ORDER, columns=CLASSES)
    return mat['Dynamic-only'].sort_values(ascending=False, kind='stable', na_position='last').index.tolist()
PALETTE = {'text': '#172C3D', 'subtext': '#566B79', 'tick': '#263A48', 'grid': '#E3E9ED', 'zebra': '#FAFBFC',
    'baseline': '#788894', 'nonsig': '#A1AAB2', 'lower': '#526470'}

def fdr_symbol(q):
    if not np.isfinite(q):
        return '-'
    if q < 0.001:
        return '***'
    if q < 0.01:
        return '**'
    if q < 0.05:
        return '*'
    return 'ns'

def paired_color(q, diff):
    if not np.isfinite(q) or q >= 0.05:
        return PALETTE['nonsig']
    return COLORS['Dynamic-only'] if np.isfinite(diff) and diff > 0 else PALETTE['lower']

def setup_axis(ax, order, labelpad=8.0):
    n = len(order)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([SYSTEM_DISPLAY[s] for s in order], fontsize=8.1, color=PALETTE['tick'])
    ax.tick_params(axis='y', length=0, pad=labelpad)
    ax.tick_params(axis='x', length=2.7, width=0.5, labelsize=7.8, pad=4, colors=PALETTE['tick'])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#86939B')
    ax.spines['bottom'].set_linewidth(0.58)
    ax.set_facecolor('none')
    ax.set_axisbelow(True)

def shade_rows(fig, order, left, right, bottom, height):
    n = len(order)
    for i, system in enumerate(order):
        if i % 2 == 0:
            color = PALETTE['zebra']
        else:
            continue
        row_bottom = bottom + height * (n - 1 - i) / n
        fig.add_artist(Rectangle((left, row_bottom), right - left, height / n, transform=fig.transFigure,
            facecolor=color, edgecolor='none', zorder=-1))

def plot_bc(bsys, csys, tests, order, outfile, disease):
    """Compact 200 x 105 mm, B and C independent labels, one vector PDF.

    Do not scale an A4-size canvas down: each element is designed to remain
    legible at 200 mm physical print width. The y-axis range is identical.
    """
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), facecolor='white')
    plot_bottom, plot_height = (0.193, 0.61)
    b_left, b_width = (0.208, 0.234)
    c_left, c_width = (0.632, 0.347)
    ax_b = fig.add_axes([b_left, plot_bottom, b_width, plot_height])
    ax_c = fig.add_axes([c_left, plot_bottom, c_width, plot_height])
    shade_rows(fig, order, 0.019, b_left + b_width + 0.002, plot_bottom, plot_height)
    shade_rows(fig, order, 0.469, 0.983, plot_bottom, plot_height)
    setup_axis(ax_b, order, labelpad=6.5)
    setup_axis(ax_c, order, labelpad=6.5)
    for ax in (ax_b, ax_c):
        for text_label in ax.get_yticklabels():
            text_label.set_fontsize(7.0)
        ax.tick_params(axis='x', labelsize=7.0, pad=2.8)
    fig.text(0.022, 0.973, 'Protein-disease associations across disease systems', ha='left', va='top',
        fontsize=10.25, fontweight='bold', color=PALETTE['text'])
    patches = [Patch(facecolor=COLORS[c], label=c, edgecolor='none') for c in CLASSES]
    fig.legend(handles=patches, loc='center', bbox_to_anchor=(0.53, 0.92), ncol=3, frameon=False, fontsize=7.35,
        handlelength=1.4, handleheight=0.85, columnspacing=1.3, handletextpad=0.4)
    fig.text(b_left, 0.87, 'B', ha='left', va='baseline', fontsize=11.0, fontweight='bold', color=PALETTE['text'])
    fig.text(b_left + 0.027, 0.87, 'Association breadth', ha='left', va='baseline', fontsize=8.5,
        fontweight='bold', color=PALETTE['text'])
    fig.text(c_left, 0.87, 'C', ha='left', va='baseline', fontsize=11.0, fontweight='bold', color=PALETTE['text'])
    fig.text(c_left + 0.027, 0.87, 'Association magnitude', ha='left', va='baseline', fontsize=8.5,
        fontweight='bold', color=PALETTE['text'])
    for x0, x1 in ((b_left, b_left + b_width), (c_left, c_left + 0.245)):
        fig.add_artist(plt.Line2D([x0, x1], [0.855, 0.855], transform=fig.transFigure, lw=0.48, color='#DDE5EA'))
    breadth = bsys.pivot(index='DiseaseSystem', columns='MoleculeClass',
        values='median_significant_pct').reindex(index=order, columns=CLASSES)
    values = breadth.to_numpy(float)
    finite = values[np.isfinite(values)]
    max_pct = float(finite.max()) if finite.size else 5.0
    xmax_b = max(5.0, np.ceil(max_pct * 1.16 / 5.0) * 5.0)
    ax_b.set_xlim(0, xmax_b)
    ax_b.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax_b.xaxis.grid(True, color=PALETTE['grid'], lw=0.35, ls=(0, (2, 3)), zorder=0)
    ax_b.axvline(0, lw=0.52, color=PALETTE['baseline'], zorder=1)
    offsets = (-0.28, 0.0, 0.28)
    for i, system in enumerate(order):
        for j, cls in enumerate(CLASSES):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            yy = i + offsets[j]
            ax_b.barh(yy, v, height=0.13, color=COLORS[cls], edgecolor='none', linewidth=0, zorder=3)
            if v == 0:
                ax_b.scatter(0, yy, s=7, color=COLORS[cls], zorder=4)
    ax_b.set_xlabel('Median significant proteins (%)', fontsize=7.25, labelpad=5.3, color=PALETTE['tick'])
    wide = csys.set_index(['DiseaseSystem', 'MoleculeClass'])
    limits = csys[['whisker_low', 'whisker_high']].to_numpy(float)
    finite = limits[np.isfinite(limits)]
    if finite.size == 0:
        raise ValueError('No usable values for panel C')
    vmin, vmax = (float(finite.min()), float(finite.max()))
    spread = max(vmax - vmin, 0.025)
    xmin = max(0.0, vmin - 0.13 * spread) if LOGHR_MODE == 'absolute' else vmin - 0.13 * spread
    effect_end = vmax + 0.13 * spread
    effect_width = effect_end - xmin
    col_base = xmin + 1.06 * effect_width
    col_dual = xmin + 1.265 * effect_width
    xmax_c = xmin + 1.365 * effect_width
    ax_c.set_xlim(xmin, xmax_c)
    ax_c.spines['bottom'].set_visible(False)
    ticks = np.linspace(xmin, effect_end, 4)
    ax_c.set_xticks(ticks)
    ax_c.xaxis.set_major_formatter(FuncFormatter(lambda xx, _: f'{xx:.2f}'))
    for tick in ticks:
        ax_c.axvline(tick, color=PALETTE['grid'], lw=0.35, ls=(0, (2, 3)), zorder=0)
    ax_c.hlines(len(order) - 0.497, xmin, effect_end, lw=0.52, color='#86939B', clip_on=False, zorder=2)
    tr = ax_c.get_xaxis_transform()
    ax_c.text((xmin + effect_end) / 2, 1.024, 'Across-disease distribution', transform=tr, fontsize=7.0,
        color=PALETTE['subtext'], ha='center', va='bottom')
    ax_c.text((col_base + col_dual) / 2, 1.085, 'Dynamic-only vs.', transform=tr, fontsize=7.05, fontweight='bold',
        color=PALETTE['text'], ha='center', va='bottom')
    ax_c.text(col_base, 1.024, 'Baseline', transform=tr, fontsize=7.0, color=PALETTE['subtext'], ha='center',
        va='bottom')
    ax_c.text(col_dual, 1.024, 'Dual', transform=tr, fontsize=7.0, color=PALETTE['subtext'], ha='center', va='bottom')
    stats = tests.set_index(['DiseaseSystem', 'class_a', 'class_b'])
    for i, system in enumerate(order):
        for j, cls in enumerate(CLASSES):
            idx = (system, cls)
            if idx not in wide.index:
                continue
            row = wide.loc[idx]
            q1, med, q3, wl, wh = [float(row[c]) for c in ('q1', 'median', 'q3', 'whisker_low', 'whisker_high')]
            if not np.all(np.isfinite([q1, med, q3, wl, wh])):
                continue
            yy = i + offsets[j]
            color = COLORS[cls]
            artists = ax_c.bxp([dict(med=med, q1=q1, q3=q3, whislo=wl, whishi=wh)], positions=[yy], widths=0.155,
                vert=False, showfliers=False, patch_artist=True, manage_ticks=False, boxprops=dict(facecolor=color,
                edgecolor=color, linewidth=0.54, alpha=0.55), whiskerprops=dict(color=color, linewidth=0.58),
                capprops=dict(color=color, linewidth=0.52), medianprops=dict(color=PALETTE['text'], linewidth=0.8), zorder=3)
            if SHOW_DISEASE_POINTS:
                dx = disease.loc[disease.DiseaseSystem.eq(system) & disease.MoleculeClass.eq(cls),
                    'disease_mean_magnitude'].to_numpy(float)
                dx = dx[np.isfinite(dx)]
                if len(dx):
                    ydot = yy + np.linspace(-0.058, 0.058, len(dx))
                    ax_c.scatter(dx, ydot, s=2.5, color=color, alpha=DOT_ALPHA, zorder=4, linewidths=0, clip_on=True)
        for xc, comparator in ((col_base, 'Baseline-only'), (col_dual, 'Dual-response')):
            r = stats.loc[system, comparator, 'Dynamic-only']
            q = float(r['FDR_BH_36'])
            diff = float(r['median_difference_b_minus_a'])
            label = fdr_symbol(q)
            ax_c.text(xc, i + offsets[2], label, ha='center', va='center', fontsize=7.6 if label not in ('ns',
                '-') else 7.0, fontweight='bold' if label not in ('ns', '-') else 'normal', color=paired_color(q, diff), zorder=6)
    abs_label = '|log(HR)|' if LOGHR_MODE == 'absolute' else 'log(HR)'
    ax_c.set_xlabel(f'Disease-level mean {abs_label}', fontsize=7.25, color=PALETTE['tick'], labelpad=5.3)
    fig.text(0.025, 0.072,
        '630 noninfectious endpoints  |  B: median significant (%)  |  C: disease-level means (box: IQR)', ha='left',
        va='center', fontsize=7.0, color=PALETTE['subtext'])
    fig.text(0.025, 0.033,
        'C: whiskers 1.5 x IQR; no outliers  |  paired Wilcoxon, BH-FDR (36)  |  * q < .05  ** q < .01  *** q < .001',
        ha='left', va='center', fontsize=7.0, color=PALETTE['subtext'])
    fig.savefig(outfile, format='pdf', facecolor='white')
    plt.close(fig)
    print('[PDF]', outfile)

def main():
    if LOGHR_MODE not in ('absolute', 'signed'):
        raise ValueError('LOGHR_MODE must be absolute or signed')
    style()
    d = read_hr()
    disease, bsys = aggregate_data(d)
    csys = summarize_c(disease)
    tests = paired_tests(disease)
    order = row_order(bsys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {'01_disease_level_breadth_magnitude.tsv': disease,
        '02_system_median_significant_percentage.tsv': bsys, '03_system_disease_level_boxplot_summary.tsv': csys,
        '04_paired_Wilcoxon_BH_FDR_all36.tsv': tests}
    for filename, tbl in tables.items():
        tbl.to_csv(OUT_DIR / filename, sep='\t', index=False)
    plot_bc(bsys, csys, tests, order, OUT_DIR / PDF_NAME, disease)
    print('[12 systems in identical order across B/C]')
    print(' | '.join((SYSTEM_DISPLAY[x] for x in order)))
    print('[Statistics] B: median significant proportions; C: distributions of disease-level mean absolute log(HR).')
    print('[Intervals] Boxplot quartiles and Tukey whiskers; no resampling and no confidence intervals.')
    print('[Comparisons] Paired Wilcoxon on within-disease class means; 36 p-values jointly BH-FDR corrected.')
    print('[Audit] Original protein global FDR reused as-is; no refitting.')
    print('[Output] A single 200 x 105 mm vector PDF (fits A4 page width).')
    print('[Result directory]', OUT_DIR.resolve())
if __name__ == '__main__':
    main()
