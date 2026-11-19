#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5: acute exercise-response module correspondence Sankey plot."""
from __future__ import annotations
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyBboxPatch
from matplotlib.colors import to_rgba
from matplotlib import font_manager

# 0. CONFIG
WORKDIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
MAIN_DIR = WORKDIR / 'acute_module_phenotype'
SWEEP_DIR = (WORKDIR / 'acute_module_resolution_check' /
             'by_parameter' / 'gamma_1.20_min_10_weighted')
NETWORK_DIR = (WORKDIR /
    'acute_response_network')
OUT_DIR = MAIN_DIR / '07_Sankey_module_composition'

# Three molecule-class annotation styles. Each style exports one PDF only.
# "bar" = small horizontal 3-class composition bars with rounded percentages.
# "dots" = three colored dots + percentages on one line (compact).
# "sidecar" = narrow vertical 3-class strips outside the original module nodes.
# If desired, set PLOT_STYLES = ("bar",) to create only the bar version.
PLOT_STYLES = ("bar", "dots", "sidecar")

# Percentage base. Default: among molecules assigned to one of three classes.
# Every module's unclassified count and coverage are audited in TSV output.
# If class coverage is not close to 100%, interpret conditional fractions carefully.
COMPOSITION_DENOMINATOR = "classified"   # "classified" or "all"

# In "all" mode a fourth segment, Other/unclassified, is drawn in gray.
CLASS_COLORS = {
    "Baseline-only": "#4C95BC",
    "Dual-response": "#55A96E",
    "Dynamic-only": "#E57C5E",
    "Other": "#CDD2D8",
}
CLASS_NAMES = ("Baseline-only", "Dual-response", "Dynamic-only")

# List files can be 1-column, TSV, or multiple columns containing protein IDs.
# The matching parser selects columns based on overlaps with saved network node IDs.
PROTEIN_LIST_FILES = {
    "Baseline-only": WORKDIR / "protein_baseline_only.tsv",
    "Dual-response": WORKDIR / "protein_dual_response.tsv",
    "Dynamic-only": WORKDIR / "protein_dynamic_only.tsv",
}
METABOLITE_GROUP_FILE = WORKDIR / "metabolite_response_classes.tsv"

# Automatic metabolite column matching uses network node IDs and response-group labels.
# To override auto-detection, supply literal column names here.
METABOLITE_ID_COLUMN = None
METABOLITE_GROUP_COLUMN = None

# Fail rather than silently render almost all nodes as "Other" if IDs mismatch.
MIN_CLASSIFIED_MOLECULES = 5
MIN_CLASSIFIED_COVERAGE = 0.05

# Keep the original Sankey at 40 mm wide. The two wide annotation variants
# provide additional space OUTSIDE the diagram without squeezing ribbons.
ORIGINAL_SANKEY_WIDTH_MM = 40.0
OUTER_FIG_WIDTH_MM = 103.0
SIDECAR_FIG_WIDTH_MM = 44.0
LEGEND_IN_WIDE_FIGURES = True

MIN_MODULE_SIZE = 10
MATCH_J_STRICTLY_ABOVE = 0.40
DRAW_J_THRESHOLD = 0.10
MIN_FLOW_N = 5
LABEL_J_THRESHOLD = 0.40
VERIFY_AGAINST_DYNAMIC_NETWORK = True
# If False, missing network files are skipped; if present, disagreements still fail.
REQUIRE_NETWORK_FILES_FOR_VERIFICATION = True

# Match the static Sankey artboard exactly (do not use bbox_inches='tight').
FIG_WIDTH_MM = 40.0
FIG_HEIGHT_MM = 74.0
NODE_X_W0 = 0.174
NODE_X_W8 = 0.777
NODE_WIDTH = 0.054
TOP = 0.842
BOTTOM = 0.120
NODE_GAP = 0.014
CURVE_STRENGTH = 0.43

MODULE_FONT_PT = 7.0
WEEK_FONT_PT = 8.3
JACCARD_FONT_PT = 7.0

W0_COLOR = '#2E8AC4'
W8_COLOR = '#E64B87'
W0_PALE = '#E7F1F8'
W8_PALE = '#FAE9F0'

# Same module colors as the static module legend. These are display labels
# supplied by the SAVED dynamic heatmap mapping; never reassign modules.
MODULE_BG_COLORS = {
    'M1': '#F4C7C3', 'M2': '#BFD7F2', 'M3': '#BFE3C0',
    'M4': '#F3E2A7', 'M5': '#D9C8F0', 'M6': '#BFE3EA',
    'M7': '#F7D6BF', 'M8': '#C9D0D8', 'M9': '#D6E8C5',
    'N1': '#F6D1E6', 'N2': '#D8D1C5', 'N3': '#E6CFC4',
}
MODULE_EDGE_COLORS = {
    'M1': '#E78B84', 'M2': '#7FB0E3', 'M3': '#7FC68A',
    'M4': '#D8B94F', 'M5': '#B497DE', 'M6': '#7DC8D6',
    'M7': '#E5AE7B', 'M8': '#93A0AD', 'M9': '#9FC37A',
    'N1': '#E49BC4', 'N2': '#B7AA93', 'N3': '#D0A792',
}
LINK_MATCHED_ALPHA = 0.85
LINK_CROSS_ALPHA = 0.54
RIBBON_OUTLINE_ALPHA = 0.14
TEXT_COLOR = '#273747'
LABEL_COLOR = '#47596C'

DRAW_TITLE = False
TITLE = 'Dynamic module remodeling'
SHOW_MODULE_SIZES = False
SHOW_JACCARD_LABELS = True
SHOW_FIGURE = False
DPI = 320

# Arial ONLY. Do not list Liberation Sans as a fallback.
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.unicode_minus': False,
})

# 1. READ SAVED DYNAMIC HEATMAP MAPPING -- NO NEW MATCHING
def need(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'Required input file not found: {path}')

def natural_key(value):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', str(value))]

def read_tsv(path: Path) -> pd.DataFrame:
    need(path)
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=True)
    df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]
    return df

def opt_id(value) -> str | None:
    if pd.isna(value) or str(value).strip().lower() in ('', 'none', 'na', 'nan'):
        return None
    return str(value).strip()

def standard_mapping(df: pd.DataFrame) -> pd.DataFrame:
    required = {'module', 'kind', 'W0_module', 'W8_module', 'scoring_nodes'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('Dynamic heatmap mapping lacks columns: '
                         + ', '.join(sorted(missing)))
    return df.copy()

def normalized_mapping_signature(df: pd.DataFrame):
    keys = ['module', 'kind', 'W0_module', 'W8_module', 'scoring_nodes']
    rows = []
    for row in df[keys].itertuples(index=False, name=None):
        cleaned = []
        for i, value in enumerate(row):
            if i in (2, 3):
                cleaned.append(opt_id(value))
            elif i == 4:
                cleaned.append(frozenset(s.strip() for s in
                                         str(value).split(';') if s.strip()))
            else:
                cleaned.append(str(value).strip())
        rows.append(tuple(cleaned))
    return {row[0]: row[1:] for row in rows}

def locate_inputs():
    # MAIN mapping is authoritative. Never silently substitute a different run.
    main_mapping_path = MAIN_DIR / 'module_scoring_groups.tsv'
    if not main_mapping_path.is_file():
        raise FileNotFoundError(
            f'Dynamic MAIN heatmap mapping is missing: {main_mapping_path}\n'
            'Please run the weighted gamma=1.2 dynamic MAIN phenotype heatmap '
            'first; no alternate matching will be invented.')
    mapping = standard_mapping(read_tsv(main_mapping_path))
    main_assignments = [MAIN_DIR / f'{w}_module_assignment.tsv'
                        for w in ('W0', 'W8')]
    if all(p.is_file() for p in main_assignments):
        source = MAIN_DIR
        print('[INPUT] Dynamic MAIN mapping and assignments')
    else:
        sweep_assignments = [SWEEP_DIR / f'{w}_module_assignment.tsv'
                             for w in ('W0', 'W8')]
        if not all(p.is_file() for p in sweep_assignments):
            raise FileNotFoundError(
                'Dynamic MAIN assignments are missing and matching saved '
                'gamma=1.2 sweep assignments are unavailable.')
        sweep_mapping = standard_mapping(read_tsv(
            SWEEP_DIR / 'module_scoring_groups.tsv'))
        if normalized_mapping_signature(mapping) != normalized_mapping_signature(sweep_mapping):
            raise ValueError('Dynamic MAIN vs saved sweep mapping differ; cannot '
                             'mix these module IDs or scoring molecules.')
        source = SWEEP_DIR
        print('[INPUT] MAIN mapping matches the sweep mapping; verified fallback')
    return source, mapping, main_mapping_path

def get_sets(week: str, source: Path) -> dict[str, set[str]]:
    path = source / f'{week}_module_assignment.tsv'
    df = read_tsv(path)
    if not {'node', 'module'}.issubset(df.columns):
        raise ValueError(f'{path}: expected node/module columns')
    df = df[['node', 'module']].copy()
    df['node'] = df['node'].astype('string').str.strip()
    df['module'] = df['module'].astype('string').str.strip()
    if df['node'].isna().any() or df['node'].eq('').any() or df['node'].duplicated().any():
        raise ValueError(f'{path}: missing or duplicated molecule IDs')
    valid = df.loc[df['module'].notna() &
                   ~df['module'].str.lower().isin(['', 'unassigned','na','nan','none'])]
    sets = {str(mid): set(part['node'].astype(str))
            for mid, part in valid.groupby('module')}
    if not sets:
        raise ValueError(f'{path}: no retained modules')
    small = {m:len(v) for m,v in sets.items() if len(v) < MIN_MODULE_SIZE}
    if small:
        raise ValueError(f'{path}: retained modules < {MIN_MODULE_SIZE}: {small}')
    return dict(sorted(sets.items(), key=lambda kv: natural_key(kv[0])))

def validate_main_mapping(df, sets0, sets8):
    """Validate dynamic heatmap IDs and scoring sets, without rematching."""
    df = standard_mapping(df)
    if df.empty or df['module'].isna().any() or df['module'].duplicated().any():
        raise ValueError('Dynamic mapping has missing/duplicated display IDs')
    seen0, seen8 = set(), set()
    for row in df.to_dict('records'):
        label, kind = str(row['module']).strip(), str(row['kind']).strip()
        m0, m8 = opt_id(row['W0_module']), opt_id(row['W8_module'])
        scoring = {s.strip() for s in str(row['scoring_nodes']).split(';') if s.strip()}
        if kind == 'Matched':
            if m0 not in sets0 or m8 not in sets8 or not re.fullmatch(r'M\d+', label):
                raise ValueError(f'Invalid dynamic Matched mapping row: {row}')
            common = sets0[m0] & sets8[m8]
            union = sets0[m0] | sets8[m8]
            j = len(common)/len(union) if union else 0.0
            if j <= MATCH_J_STRICTLY_ABOVE:
                raise ValueError(f'{label}: saved Matched Jaccard={j:.4f} '
                                 f'<= {MATCH_J_STRICTLY_ABOVE:.2f}')
            expected = common
            if 'jaccard' in df and opt_id(row.get('jaccard')) is not None:
                if abs(float(row['jaccard'])-j) > 1e-6:
                    raise ValueError(f'{label}: saved Jaccard != recomputed overlap')
        elif kind == 'W0_only':
            if m0 not in sets0 or m8 is not None or not re.fullmatch(r'M\d+', label):
                raise ValueError(f'Invalid dynamic W0_only mapping row: {row}')
            expected = sets0[m0]
        elif kind == 'W8_new':
            if m0 is not None or m8 not in sets8 or not re.fullmatch(r'N\d+', label):
                raise ValueError(f'Invalid dynamic W8_new mapping row: {row}')
            expected = sets8[m8]
        else:
            raise ValueError(f'Unknown dynamic heatmap mapping kind: {kind}')
        if scoring != expected:
            raise ValueError(f'{label}: saved scoring_nodes differ from module '
                             f'composition (saved={len(scoring)}, expected={len(expected)})')
        if m0 is not None:
            if m0 in seen0:
                raise ValueError(f'Duplicate W0 raw module assignment in mapping: {m0}')
            seen0.add(m0)
        if m8 is not None:
            if m8 in seen8:
                raise ValueError(f'Duplicate W8 raw module assignment in mapping: {m8}')
            seen8.add(m8)
    if seen0 != set(sets0) or seen8 != set(sets8):
        raise ValueError('Heatmap mapping does not cover every saved module: '
                         f'W0 unrepresented={set(sets0)-seen0}, '
                         f'W8 unrepresented={set(sets8)-seen8}')
    return df

def verify_current_network(week, heatmap_sets):
    if not VERIFY_AGAINST_DYNAMIC_NETWORK:
        return
    path = NETWORK_DIR / f'03_{week}_module_assignment.txt'
    if not path.is_file():
        if REQUIRE_NETWORK_FILES_FOR_VERIFICATION:
            raise FileNotFoundError(
                f'Cannot verify against current dynamic network: {path}')
        print(f'[WARN] {week}: current network missing; verification skipped')
        return
    df = read_tsv(path)
    if not {'node', 'module'}.issubset(df.columns):
        raise ValueError(f'{path}: node/module columns required')
    df['node'] = df['node'].astype('string').str.strip()
    df['module'] = df['module'].astype('string').str.strip()
    if df['node'].isna().any() or df['node'].duplicated().any():
        raise ValueError(f'{path}: missing/duplicate node IDs')
    assigned = df.loc[df['module'].notna() &
                      ~df['module'].str.lower().isin(['','unassigned','na','nan','none'])]
    current = {frozenset(group['node'].astype(str))
               for _, group in assigned.groupby('module')}
    from_heatmap = {frozenset(group) for group in heatmap_sets.values()}
    if current != from_heatmap:
        raise ValueError(
            f'{week}: dynamic MAIN gamma1.2 heatmap module memberships differ '
            f'from the current network file {path}. This diagram must not '
            'silently combine incompatible modules.')
    print(f'[CHECK] {week}: all saved heatmap modules agree with dynamic network')

def overlap_table(sets0, sets8, labels0, labels8, matched):
    rows = []
    for m0, s0 in sets0.items():
        for m8, s8 in sets8.items():
            common = len(s0 & s8)
            union = len(s0 | s8)
            j = common/union if union else 0.0
            rows.append({
                'W0_original_module': m0, 'W8_original_module': m8,
                'W0_display': labels0[m0], 'W8_display': labels8[m8],
                'W0_size': len(s0), 'W8_size': len(s8),
                'overlap_n': common, 'union_n': union, 'jaccard': j,
                'w0_retained_fraction': common/len(s0),
                'w8_retained_fraction': common/len(s8),
                'is_primary_matched': matched.get(m0) == m8,
                'draw_link': common >= MIN_FLOW_N and j >= DRAW_J_THRESHOLD,
                'label_link': common >= MIN_FLOW_N and
                              j >= max(DRAW_J_THRESHOLD, LABEL_J_THRESHOLD),
            })
    return pd.DataFrame(rows)

# 1B. MATCH SAVED MODULE NODES TO THE THREE EXISTING MOLECULE-CLASS LISTS
def canon_id(value):
    """Only conservative text normalization; never infer proteins from genes."""
    if value is None or pd.isna(value):
        return ""
    token = str(value).strip().strip('\"').strip("'").replace('\ufeff', '')
    if token.lower().endswith('.txt'):
        token = token[:-4]
    return token.casefold()

def read_rows_loose(path):
    """Read tab, comma, or whitespace delimited data as raw rows."""
    need(path)
    rows = []
    with path.open('r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '\t' in line:
                cells = line.split('\t')
            elif ',' in line:
                import csv
                cells = next(csv.reader([line]))
            else:
                cells = line.split()
            rows.append([s.strip().strip('\"') for s in cells])
    if not rows:
        raise ValueError(f'Empty molecule-class file: {path}')
    return rows

def read_protein_list(path, node_ids):
    """Identify the protein-ID column by maximum overlap with network nodes."""
    rows = read_rows_loose(path)
    ncol = max(map(len, rows))
    hits = []
    for j in range(ncol):
        ids = {canon_id(row[j]) for row in rows if j < len(row)}
        ids.discard('')
        hits.append((len(ids & node_ids), j, ids))
    hits.sort(key=lambda x: (-x[0], x[1]))
    if not hits or hits[0][0] == 0:
        raise ValueError(f'No protein IDs in {path.name} match module node IDs. '
                         f'Sample network IDs: {sorted(node_ids)[:8]}')
    print(f'[CLASS] {path.name}: column={hits[0][1]+1}, '
          f'matching node IDs={hits[0][0]}')
    return hits[0][2]

def response_group(value):
    """Recognize response classes, not metabolite chemical main classes."""
    s = str(value).strip().casefold().replace('_', '-').replace(' ', '-')
    s = re.sub(r'-+', '-', s)
    labels = {
        'baseline-only': 'Baseline-only', 'baseline': 'Baseline-only',
        'baselineonly': 'Baseline-only', 'baseline-specific': 'Baseline-only',
        'dual-response': 'Dual-response', 'dual': 'Dual-response',
        'both': 'Dual-response', 'both-response': 'Dual-response',
        'both-responsive': 'Dual-response', 'dualresponse': 'Dual-response',
        'dynamic-only': 'Dynamic-only', 'dynamic': 'Dynamic-only',
        'trend-only': 'Dynamic-only', 'trend': 'Dynamic-only',
        'trendonly': 'Dynamic-only', 'dynamiconly': 'Dynamic-only',
    }
    return labels.get(s)

def read_metabolite_classes(path, node_ids):
    """Strictly identify group column by response labels, ID column by overlap."""
    rows = read_rows_loose(path)
    ncol = max(map(len, rows))
    names = rows[0]
    if METABOLITE_ID_COLUMN is not None or METABOLITE_GROUP_COLUMN is not None:
        if not (METABOLITE_ID_COLUMN and METABOLITE_GROUP_COLUMN):
            raise ValueError('Set both METABOLITE_ID_COLUMN and METABOLITE_GROUP_COLUMN.')
        if METABOLITE_ID_COLUMN not in names or METABOLITE_GROUP_COLUMN not in names:
            raise ValueError(f'Metabolite headers: {names}')
        id_j = names.index(METABOLITE_ID_COLUMN)
        group_j = names.index(METABOLITE_GROUP_COLUMN)
        data = rows[1:]
    else:
        group_candidates = []
        for j in range(ncol):
            c = [response_group(r[j]) for r in rows if j < len(r)]
            group_candidates.append((sum(x is not None for x in c), j))
        group_candidates.sort(reverse=True)
        if group_candidates[0][0] == 0:
            raise ValueError(f'Cannot locate 3-class response-group column in '
                             f'{path.name}. First row: {names}')
        group_j = group_candidates[0][1]
        id_candidates = []
        for j in range(ncol):
            if j == group_j:
                continue
            ids = {canon_id(r[j]) for r in rows if j < len(r)}
            id_candidates.append((len(ids & node_ids), j))
        id_candidates.sort(reverse=True)
        if not id_candidates or id_candidates[0][0] == 0:
            raise ValueError(f'Cannot match metabolite IDs from {path.name} '
                             f'to module nodes. First row: {names}')
        id_j = id_candidates[0][1]
        data = rows
    print(f'[CLASS] {path.name}: ID column={id_j+1}, '
          f'response-group column={group_j+1}')
    result = {}
    for row in data:
        if max(id_j, group_j) >= len(row):
            continue
        mid = canon_id(row[id_j])
        cl = response_group(row[group_j])
        if not mid or cl is None:
            continue
        if mid in result and result[mid] != cl:
            raise ValueError(f'Conflicting metabolite groups for {mid}')
        result[mid] = cl
    if not result:
        raise ValueError('No recognized metabolite class entries')
    return result

def build_class_map(sets0, sets8):
    node_ids = {canon_id(n) for groups in (sets0, sets8)
                for members in groups.values() for n in members}
    if '' in node_ids:
        raise ValueError('Missing normalized node IDs')
    classes = {}
    for cls, path in PROTEIN_LIST_FILES.items():
        ids = read_protein_list(path, node_ids)
        for mid in ids & node_ids:
            if mid in classes and classes[mid] != cls:
                raise ValueError(f'Protein ID occurs in multiple classes: {mid}')
            classes[mid] = cls
    for mid, cls in read_metabolite_classes(METABOLITE_GROUP_FILE, node_ids).items():
        if mid not in node_ids:
            continue
        if mid in classes and classes[mid] != cls:
            raise ValueError(f'Molecule {mid} has conflicting protein/metabolite class')
        classes[mid] = cls
    matched = len(classes)
    coverage = matched / len(node_ids)
    print(f'[CLASS QC] unique network nodes={len(node_ids)}, '
          f'matched to 3 classes={matched} ({coverage:.1%})')
    if matched < MIN_CLASSIFIED_MOLECULES or coverage < MIN_CLASSIFIED_COVERAGE:
        raise ValueError('Too few module nodes matched the 3-class lists. '
                         'Check input IDs and metabolite columns instead of '
                         'plotting possibly incorrect composition percentages.')
    return classes

def make_composition(sets0, sets8, labels0, labels8, classes):
    rows = []
    for week, sets, labels in [('W0', sets0, labels0), ('W8', sets8, labels8)]:
        for raw_id, members in sets.items():
            values = [classes.get(canon_id(node), 'Other') for node in members]
            totals = {cl: values.count(cl) for cl in (*CLASS_NAMES, 'Other')}
            n = len(members)
            class_n = n - totals['Other']
            denom = class_n if COMPOSITION_DENOMINATOR == 'classified' else n
            row = {'week': week, 'module_original': raw_id,
                   'display_id': labels[raw_id], 'module_total_n': n,
                   'classified_n': class_n, 'unclassified_n': totals['Other'],
                   'classified_coverage_pct': 100.0*class_n/n}
            for cl, prefix in [('Baseline-only','baseline'),
                               ('Dual-response','dual'),
                               ('Dynamic-only','dynamic'), ('Other','other')]:
                row[prefix + '_n'] = totals[cl]
                row[prefix + '_pct_all'] = 100*totals[cl]/n
                row[prefix + '_pct_display'] = (100*totals[cl]/denom if denom else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

def get_comp(row):
    classes = list(CLASS_NAMES)
    if COMPOSITION_DENOMINATOR == 'all':
        classes.append('Other')
    prefixes = {'Baseline-only':'baseline', 'Dual-response':'dual',
                'Dynamic-only':'dynamic', 'Other':'other'}
    return [(cl, float(row[prefixes[cl]+'_pct_display'])) for cl in classes]

def composition_annotation(ax, side, geo, row, style):
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    comp = get_comp(row)
    comp = [(cl, pct) for cl, pct in comp if np.isfinite(pct)]
    y = geo['mid']
    if style == 'bar':
        left = -0.584 if side == 'W0' else 0.971
        width, height = 0.654, 0.036
        # Outline holds the nominal 100%, even if one class is absent.
        ax.add_patch(Rectangle((left,y-height/2),width,height,
                               facecolor='#EFF1F3',edgecolor='#D6DADE',lw=.28,
                               zorder=7))
        cursor = left
        for cl,pct in comp:
            w = width * pct / 100
            if w <= 0:
                continue
            ax.add_patch(Rectangle((cursor,y-height/2),w,height,
                                   facecolor=CLASS_COLORS[cl],edgecolor='none',
                                   zorder=8))
            # Segments too small for text remain visible without unreadable labels.
            if w >= 0.105:
                ax.text(cursor+w/2,y,f'{pct:.0f}%',ha='center',va='center',
                        fontsize=7,color='white' if cl != 'Other' else '#263745',
                        fontweight='bold',zorder=9)
            cursor += w
    elif style == 'dots':
        start = -0.57 if side == 'W0' else 0.973
        spacing = 0.205 if COMPOSITION_DENOMINATOR == 'classified' else .16
        for idx, (cl,pct) in enumerate(comp):
            x = start+idx*spacing
            ax.scatter([x],[y],s=8.5,marker='o',color=CLASS_COLORS[cl],
                       edgecolors='none',zorder=9)
            ax.text(x+0.028,y,f'{pct:.0f}%',fontsize=7,ha='left',va='center',
                    color='#1F2A36',zorder=9)
    elif style == 'sidecar':
        # Separate attribute strip; never recolor the module block itself.
        left = 0.133 if side == 'W0' else 0.851
        width = 0.022
        height = max(geo['height'],0.023)
        bottom = y-height/2
        cumulative = bottom
        for cl,pct in reversed(comp):
            h = height*pct/100
            if h > 0:
                ax.add_patch(Rectangle((left,cumulative),width,h,
                                       color=CLASS_COLORS[cl],ec='none',zorder=9))
                cumulative += h
        ax.add_patch(Rectangle((left,bottom),width,height,facecolor='none',
                               edgecolor='#FFFFFF',lw=0.22,zorder=10))
    else:
        raise ValueError(f'Unknown style: {style}')

def plot_comp_legend(ax, style):
    if style == 'sidecar' or not LEGEND_IN_WIDE_FIGURES:
        return
    # Legend row is independent of Sankey modules / Jaccard ribbons.
    items = list(CLASS_NAMES) + (['Other'] if COMPOSITION_DENOMINATOR == 'all' else [])
    x_positions = [-0.57,0.09,0.74] if len(items)==3 else [-0.6,-.08,.48,1.05]
    for x,cl in zip(x_positions,items):
        ax.scatter([x],[0.040],s=12,color=CLASS_COLORS[cl],zorder=9)
        ax.text(x+0.055,0.040,cl,ha='left',va='center',fontsize=7,
                color='#273747')
    ax.text(-.57,.080,
            ('Composition among classified molecules'
             if COMPOSITION_DENOMINATOR=='classified'
             else 'Composition among all module molecules'),
            color='#64717D',ha='left',va='center',fontsize=7)

# 2. CURVED RIBBON LAYOUT -- VECTOR PATHS WITH CLIPPED GRADIENT
def module_layout(order: list[str], sets: dict, unit_height: float) -> dict:
    blocks={}
    cursor=TOP
    for mid in order:
        size=len(sets[mid]); height=size*unit_height
        blocks[mid]={'top':cursor, 'bottom':cursor-height,
                     'height':height, 'size':size, 'mid':cursor-height/2}
        cursor-=height+NODE_GAP
    return blocks

def distribute_ribbons(flows: pd.DataFrame, lo0: dict, lo8: dict, unit: float) -> pd.DataFrame:
    if flows.empty:
        return flows.copy()
    flows=flows.copy()
    pos0={name:i for i,name in enumerate(lo0)}
    pos8={name:i for i,name in enumerate(lo8)}
    flows['_a']=flows['W0_original_module'].map(pos0)
    flows['_b']=flows['W8_original_module'].map(pos8)
    for side, block, own, other, sort_cols in [
        ('source',lo0,'W0_original_module','W8_original_module',['_a','_b']),
        ('target',lo8,'W8_original_module','W0_original_module',['_b','_a'])
    ]:
        for module, info in block.items():
            part=flows.loc[flows[own]==module].sort_values(sort_cols, kind='mergesort')
            total=int(part['overlap_n'].sum())
            if total>info['size']:
                raise ValueError(f'{side}: displayed overlap exceeds module size {module}')
            cursor=info['mid']+(total*unit)/2.0
            for ix,rec in part.iterrows():
                height=int(rec['overlap_n'])*unit
                flows.loc[ix, side+'_top']=cursor
                flows.loc[ix, side+'_bottom']=cursor-height
                flows.loc[ix, side+'_mid']=cursor-height/2
                cursor-=height
    return flows.sort_values(['_a','_b']).reset_index(drop=True)

def ribbon_path(xleft, xright, st, sb, tt, tb):
    """Continuous cubics have flat horizontal tangents at both endpoints."""
    dx=xright-xleft
    xctrl1=xleft+CURVE_STRENGTH*dx
    xctrl2=xright-CURVE_STRENGTH*dx
    xy=[(xleft,st), (xctrl1,st),(xctrl2,tt),(xright,tt),
        (xright,tb), (xctrl2,tb),(xctrl1,sb),(xleft,sb), (xleft,st)]
    codes=[MplPath.MOVETO, MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
           MplPath.LINETO, MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
           MplPath.CLOSEPOLY]
    return MplPath(xy,codes)

def cubic_at(p0,p1,p2,p3,t):
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3

def curve_center(st, sb, tt, tb, t):
    sm=(st+sb)/2; tm=(tt+tb)/2
    return cubic_at(sm,sm,tm,tm,t)

def make_horizontal_rgba_gradient(color_left, color_right, alpha_scale=1.0, width=900):
    """Return a 1 x width x 4 RGBA array for clipping into a ribbon path."""
    c0=np.array(to_rgba(color_left), dtype=float)
    c1=np.array(to_rgba(color_right), dtype=float)
    c0[3]*=float(alpha_scale)
    c1[3]*=float(alpha_scale)
    t=np.linspace(0.0, 1.0, int(width), dtype=float)[:,None]
    arr=(1.0-t)*c0 + t*c1
    return arr[None, :, :]

def draw_gradient_ribbon(ax, path, x0, x1, y0, y1,
                         color_left, color_right,
                         alpha_scale=1.0, zorder=2):
    """Draw one left-to-right gradient ribbon clipped to a Bezier path."""
    clip_patch=PathPatch(path, facecolor='none', edgecolor='none', lw=0, transform=ax.transData)
    ax.add_patch(clip_patch)
    grad=make_horizontal_rgba_gradient(color_left, color_right, alpha_scale=alpha_scale, width=350)
    ax.imshow(grad, extent=(x0, x1, y0, y1), origin='lower', aspect='auto',
              interpolation='bicubic', clip_path=clip_patch, clip_on=True, zorder=zorder)
    # faint outline so thin flows remain readable
    ax.add_patch(PathPatch(path, facecolor='none', edgecolor=to_rgba('#6E7E8E', RIBBON_OUTLINE_ALPHA*alpha_scale),
                           lw=0.25, joinstyle='round', capstyle='round', zorder=zorder+0.1))

def y_to_font(y_range, nmax):
    return 10 if nmax<=8 else 9 if nmax<=12 else 8.3

def draw_figure(order0,order8,sets0,sets8,labels0,labels8,flows,
                composition,style):
    nmax=max(len(order0),len(order8))
    if style == 'sidecar':
        fig_width = SIDECAR_FIG_WIDTH_MM
        xlim = (-.05,1.05)
    else:
        fig_width = OUTER_FIG_WIDTH_MM
        extra = (fig_width / ORIGINAL_SANKEY_WIDTH_MM - 1.0) / 2.0
        xlim = (-extra, 1.0+extra)
    fig, ax=plt.subplots(figsize=(fig_width/25.4, FIG_HEIGHT_MM/25.4),
                         facecolor='white')
    # Fill the physical page; padding is built into the x/y limits above.
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(*xlim);ax.set_ylim(0,1);ax.axis('off')
    available=min(TOP-BOTTOM-(len(order0)-1)*NODE_GAP,
                  TOP-BOTTOM-(len(order8)-1)*NODE_GAP)
    if available<=0:
        raise ValueError('Not enough vertical area for this number of modules')
    unit=min((TOP-BOTTOM-(len(order0)-1)*NODE_GAP)/sum(map(lambda m:len(sets0[m]),order0)),
             (TOP-BOTTOM-(len(order8)-1)*NODE_GAP)/sum(map(lambda m:len(sets8[m]),order8)))
    lo0=module_layout(order0,sets0,unit)
    lo8=module_layout(order8,sets8,unit)
    segments=distribute_ribbons(flows,lo0,lo8,unit)
    xa=NODE_X_W0+NODE_WIDTH; xb=NODE_X_W8

    # BANDS: smooth left->right W0-to-W8 color gradient, clipped to each ribbon.
    # Matched flows use slightly higher opacity; secondary flows remain lighter.
    if not segments.empty:
        for rec in segments.sort_values(['is_primary_matched','overlap_n'],
                                        ascending=[True,False]).to_dict('records'):
            match=bool(rec['is_primary_matched'])
            path=ribbon_path(xa,xb,rec['source_top'],rec['source_bottom'],
                             rec['target_top'],rec['target_bottom'])
            ymin=min(rec['source_bottom'], rec['target_bottom'])
            ymax=max(rec['source_top'], rec['target_top'])
            # Ribbon fades from source-module background to target-module background.
            # Labels are from saved heatmap mapping, never inferred/rematched.
            left_id = labels0[rec['W0_original_module']]
            right_id = labels8[rec['W8_original_module']]
            draw_gradient_ribbon(
                ax, path, xa, xb, ymin, ymax,
                color_left=MODULE_BG_COLORS[left_id],
                color_right=MODULE_BG_COLORS[right_id],
                alpha_scale=(LINK_MATCHED_ALPHA if match else LINK_CROSS_ALPHA),
                zorder=2,
            )

    # NODE BLOCKS: module-specific color on BOTH sides of Sankey.
    def nodes(layout, x, display_ids):
        for original_module, geo in layout.items():
            display_id = display_ids[original_module]
            ax.add_patch(FancyBboxPatch(
                (x, geo['bottom']), NODE_WIDTH, geo['height'],
                boxstyle='round,pad=0.00001,rounding_size=0.005',
                linewidth=0.85,
                edgecolor=MODULE_EDGE_COLORS[display_id],
                facecolor=MODULE_BG_COLORS[display_id],
                zorder=5))
    nodes(lo0, NODE_X_W0, labels0)
    nodes(lo8, NODE_X_W8, labels8)

    # Labels on outside; when W8 modules match, display the W0-anchored M ID.
    for mid,geo in lo0.items():
        label=labels0[mid]+(f' ({geo["size"]})' if SHOW_MODULE_SIZES else '')
        ax.text((0.118 if style == 'sidecar' else NODE_X_W0-0.012), geo['mid'], label,
                fontsize=MODULE_FONT_PT, color=TEXT_COLOR,
                va='center', ha='right', zorder=8)
    for mid,geo in lo8.items():
        label=labels8[mid]+(f' ({geo["size"]})' if SHOW_MODULE_SIZES else '')
        ax.text((0.886 if style == 'sidecar' else NODE_X_W8+NODE_WIDTH+0.012), geo['mid'], label,
                fontsize=MODULE_FONT_PT, color=TEXT_COLOR,
                va='center', ha='left', zorder=8)

    ax.text(NODE_X_W0+NODE_WIDTH/2, 0.916, 'W0', ha='center', va='center',
            color=W0_COLOR, fontsize=WEEK_FONT_PT, weight='bold')
    ax.text(NODE_X_W8+NODE_WIDTH/2, 0.916, 'W8', ha='center', va='center',
            color=W8_COLOR, fontsize=WEEK_FONT_PT, weight='bold')

    # Annotate strongest links. Slightly stagger x position when labels overlap.
    if SHOW_JACCARD_LABELS and not segments.empty:
        significant=segments.loc[segments['label_link']].copy()
        significant=significant.sort_values(['source_mid','target_mid'],ascending=False)
        placed=[]
        for i,rec in enumerate(significant.to_dict('records')):
            t=0.49 + (i%3-1)*0.10
            xx=xa+(xb-xa)*t
            yy=curve_center(rec['source_top'],rec['source_bottom'],
                            rec['target_top'],rec['target_bottom'],t)
            # If two labels collide, try alternate horizontal placements on their flow.
            for trial in [t,0.34,0.66,0.46,0.56]:
                newx=xa+(xb-xa)*trial
                newy=curve_center(rec['source_top'],rec['source_bottom'],
                                  rec['target_top'],rec['target_bottom'],trial)
                if all(abs(newy-y)>0.031 or abs(newx-x)>0.11 for x,y in placed):
                    xx,yy=newx,newy
                    break
            placed.append((xx,yy))
            ax.text(xx,yy,f'{float(rec["jaccard"]):.2f}',fontsize=JACCARD_FONT_PT,
                    ha='center',va='center',color=LABEL_COLOR,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':0.85,'pad':0.35},
                    zorder=10)
    for side, layout in [('W0',lo0),('W8',lo8)]:
        for module,geo in layout.items():
            row = composition.loc[(composition['week']==side) &
                                  (composition['module_original']==module)]
            if len(row)!=1:
                raise ValueError(f'Missing/duplicated composition for {side} {module}')
            composition_annotation(ax,side,geo,row.iloc[0],style)
    plot_comp_legend(ax,style)
    if DRAW_TITLE:
        fig.suptitle(TITLE,x=0.06,y=0.995,ha='left',fontsize=15,
                     color=TEXT_COLOR,weight='bold')
    return fig,segments,lo0,lo8

# 3. RUN / EXPORT
def main():
    # Never silently substitute another font for Arial.
    try:
        font_manager.findfont('Arial', fallback_to_default=False)
    except ValueError as exc:
        raise RuntimeError(
            'Arial is not installed or not visible to Matplotlib. '
            'Install/register Arial, refresh the Matplotlib font cache, and rerun.'
        ) from exc
    print('='*88)
    print('DYNAMIC Delta3TP W0/W8 SANKEY -- gamma=1.2, MAIN heatmap IDs')
    print('Read saved module assignments and matching; NO Louvain/rematching')
    print(f'Ribbons: overlap >= {MIN_FLOW_N}, Jaccard >= {DRAW_J_THRESHOLD:.2f}')
    print('='*88)

    source_dir, mapping, mapping_path = locate_inputs()
    sets0 = get_sets('W0', source_dir)
    sets8 = get_sets('W8', source_dir)
    mapping = validate_main_mapping(mapping, sets0, sets8)

    for week, groups in (('W0', sets0), ('W8', sets8)):
        print(f'[{week}] retained modules={len(groups)}; '
              f'assigned molecules={sum(map(len,groups.values())):,}')
        verify_current_network(week, groups)

    labels0, labels8, matched = {}, {}, {}
    for row in mapping.to_dict('records'):
        m0, m8 = opt_id(row['W0_module']), opt_id(row['W8_module'])
        label = str(row['module']).strip()
        if m0 is not None:
            labels0[m0] = label
        if m8 is not None:
            labels8[m8] = label
        if str(row['kind']) == 'Matched':
            matched[m0] = m8
    if len(set(labels0.values())) != len(labels0) or len(set(labels8.values())) != len(labels8):
        raise ValueError('Repeated display ID on the same side')

    # Order is exactly the heatmap module display order, not raw Louvain order.
    records = mapping.to_dict('records')
    order0 = [opt_id(row['W0_module']) for row in records
              if opt_id(row['W0_module']) is not None]
    order8 = [opt_id(row['W8_module']) for row in records
              if opt_id(row['W8_module']) is not None]

    if COMPOSITION_DENOMINATOR not in ('classified','all'):
        raise ValueError('COMPOSITION_DENOMINATOR must be classified or all')
    classes = build_class_map(sets0, sets8)
    composition = make_composition(sets0, sets8, labels0, labels8, classes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    composition.to_csv(OUT_DIR/'07_module_class_composition.tsv',sep='\t',index=False,
                       float_format='%.5f')
    # Record exactly which saved module-node IDs were classified for auditing.
    audit=[]
    for wk,sets,lab in (('W0',sets0,labels0),('W8',sets8,labels8)):
        for mid,nodes in sets.items():
            for node in sorted(nodes):
                audit.append((wk,mid,lab[mid],node,classes.get(canon_id(node),'Other')))
    pd.DataFrame(audit,columns=['week','module_original','display_id','node','molecule_class'])\
        .to_csv(OUT_DIR/'07_module_molecule_class_audit.tsv',sep='\t',index=False)

    print('\n[MODULE COMPOSITION]')
    print(composition[['week','display_id','module_total_n','classified_n',
                       'unclassified_n','baseline_pct_display','dual_pct_display',
                       'dynamic_pct_display']].to_string(index=False,
                       float_format=lambda x:f'{x:.1f}'))
    pairs = overlap_table(sets0, sets8, labels0, labels8, matched)
    drawn = pairs.loc[pairs['draw_link']].copy()
    paths={}
    for style in PLOT_STYLES:
        fig, segments, geo0, geo8 = draw_figure(
            order0, order8, sets0, sets8, labels0, labels8, drawn,
            composition, style)
        path=OUT_DIR / f'07_dynamic_W0_W8_module_sankey_composition_{style}.pdf'
        fig.savefig(path,dpi=DPI,facecolor='white')
        print(f'[PDF] {style}: {path.resolve()}')
        paths[style]=path
        plt.close(fig)
    pairs.to_csv(OUT_DIR/'07_all_W0_W8_overlap_Jaccard.tsv', sep='\t', index=False)
    drawn.to_csv(OUT_DIR/'07_drawn_ribbon_pairs.tsv', sep='\t', index=False)
    segments.to_csv(OUT_DIR/'07_drawn_ribbon_vector_coordinates.tsv',
                    sep='\t', index=False)
    mapping.to_csv(OUT_DIR/'07_mapping_USED_UNCHANGED.tsv', sep='\t', index=False)

    covered0 = {m: sum(int(r['overlap_n']) for r in drawn.to_dict('records')
                       if r['W0_original_module'] == m) for m in order0}
    covered8 = {m: sum(int(r['overlap_n']) for r in drawn.to_dict('records')
                       if r['W8_original_module'] == m) for m in order8}
    coverage = pd.DataFrame(
        [{'week':'W0', 'module_original':m, 'display_id':labels0[m],
          'total_n':len(sets0[m]), 'ribbon_n':covered0[m],
          'unshown_n':len(sets0[m])-covered0[m]} for m in order0]
        + [{'week':'W8', 'module_original':m, 'display_id':labels8[m],
            'total_n':len(sets8[m]), 'ribbon_n':covered8[m],
            'unshown_n':len(sets8[m])-covered8[m]} for m in order8])
    coverage.to_csv(OUT_DIR/'07_module_ribbon_coverage.tsv', sep='\t', index=False)

    parameters = {
        'mapping_source': str(mapping_path),
        'assignment_source': str(source_dir),
        'network_validation': str(NETWORK_DIR) if VERIFY_AGAINST_DYNAMIC_NETWORK else 'off',
        'network': 'Delta3TP W0 vs W8 acute molecular responses',
        'module_method': 'weighted positive-edge Louvain gamma=1.2 min_size=10',
        'matching': 'saved dynamic phenotype heatmap greedy Jaccard >0.40 (unchanged)',
        'W0_modules': len(sets0), 'W8_modules': len(sets8),
        'matched_pairs': len(matched),
        'draw_min_overlap': MIN_FLOW_N,
        'draw_min_jaccard': DRAW_J_THRESHOLD,
        'label_min_jaccard': LABEL_J_THRESHOLD,
        'ribbon_width': 'overlap molecule count on common W0/W8 scale',
        'gradient': 'source module color -> target module color',
        'figure_width_mm': FIG_WIDTH_MM,
        'figure_height_mm': FIG_HEIGHT_MM,
        'gradient_format': 'rasterized color gradient clipped to vector bezier shapes',
        'composition_denominator': COMPOSITION_DENOMINATOR,
        'composition_styles': ','.join(PLOT_STYLES),
        'composition_classes': ';'.join(CLASS_NAMES),
        'unclassified_policy': ('exclude from percentage, retain in QC TSV'
                                if COMPOSITION_DENOMINATOR=='classified' else
                                'include as gray Other segment'),
    }
    with open(OUT_DIR/'07_plot_parameters.txt', 'w', encoding='utf8') as stream:
        for key, value in parameters.items():
            stream.write(f'{key}\t{value}\n')

    print('\nDYNAMIC HEATMAP MODULE MAPPING (unchanged):')
    print(mapping[['module','kind','W0_module','W8_module']].to_string(index=False))
    print(f'\nMatched pairs: {len(matched)}')
    print(f'Drawn ribbons: {len(drawn)}/{len(pairs)}; '
          f'displayed overlap total={int(drawn.overlap_n.sum()):,}')
    print('\n[PDF ONLY] Files:',*[str(path.resolve()) for path in paths.values()],sep='\n  ')
    print('[DATA] module composition, node audit, and original ribbon tables:',OUT_DIR)
    return pairs, mapping, composition

if __name__ == '__main__':
    main()
