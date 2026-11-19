#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 4: resting-state module correspondence Sankey plot."""
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

# 0. CONFIG — FIXED MAIN ANALYSIS

WORKDIR = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
MAIN_DIR = WORKDIR / 'resting_module_phenotype'
SWEEP_DIR = (WORKDIR / 'resting_module_resolution_check'
             / 'by_parameter' / 'gamma_1.20_min_10_weighted')
VERIFY_DIR = WORKDIR / 'resting_network'
OUT_DIR = MAIN_DIR / '07_Sankey_heatmap_consistent'

# These are validation assertions based on the three-network comparison.
# They DO NOT affect community detection or force a module count.
EXPECTED_MODULE_COUNTS = {'W0': 8, 'W8': 6}
EXPECTED_RETAINED_NODES = {'W0': 1104, 'W8': 1102}
MIN_MODULE_SIZE = 10
MATCH_J_STRICTLY_ABOVE = 0.40  # validate saved Hungarian matched mapping
DRAW_J_THRESHOLD = 0.10
MIN_FLOW_N = 5
LABEL_J_THRESHOLD = 0.40
VERIFY_AGAINST_FINAL_SIMPLE_WHEN_AVAILABLE = True

# Physical output dimensions. The exported PDF/SVG page is EXACTLY 40 x 74 mm.
FIG_WIDTH_MM = 40.0
FIG_HEIGHT_MM = 74.0
NODE_X_W0 = 0.174
NODE_X_W8 = 0.777
NODE_WIDTH = 0.054
TOP = 0.842
BOTTOM = 0.120
NODE_GAP = 0.014
CURVE_STRENGTH = 0.43

# Typography/layout for a 40-mm narrow figure (74-mm height)
MODULE_FONT_PT = 7.0
WEEK_FONT_PT = 8.3
JACCARD_FONT_PT = 7.0

W0_COLOR = '#2E8AC4'
W8_COLOR = '#E64B87'
W0_PALE = '#E7F1F8'
W8_PALE = '#FAE9F0'
# Consistent module colors: EXACT palette from the figure legend.
MODULE_BG_COLORS = {
    'M1': '#F4C7C3', 'M2': '#BFD7F2', 'M3': '#BFE3C0',
    'M4': '#F3E2A7', 'M5': '#D9C8F0', 'M6': '#BFE3EA',
    'M7': '#F7D6BF', 'M8': '#C9D0D8', 'N1': '#F6D1E6',
    'N2': '#D8D1C5', 'N3': '#E6CFC4',
}
MODULE_EDGE_COLORS = {
    'M1': '#E78B84', 'M2': '#7FB0E3', 'M3': '#7FC68A',
    'M4': '#D8B94F', 'M5': '#B497DE', 'M6': '#7DC8D6',
    'M7': '#E5AE7B', 'M8': '#93A0AD', 'N1': '#E49BC4',
    'N2': '#B7AA93', 'N3': '#D0A792',
}
# Use pale module fills for ribbons and stronger matching module outlines.
LINK_MATCHED_ALPHA = 0.85
LINK_CROSS_ALPHA = 0.54
RIBBON_OUTLINE_ALPHA = 0.14
TEXT_COLOR = '#273747'
LABEL_COLOR = '#47596C'

DRAW_TITLE = False
TITLE = 'Static module remodeling'
SHOW_MODULE_SIZES = False
SHOW_JACCARD_LABELS = True
SHOW_FIGURE = False
DPI = 320

plt.rcParams.update({
    'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.facecolor': 'white', 'axes.unicode_minus': False,
})

# 1. READ SAVED MAIN MODULES AND HEATMAP MAPPING (NO REMATCHING)

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
        raise ValueError('Main phenotype heatmap mapping lacks columns: '
                         + ', '.join(sorted(missing)))
    return df.copy()

def normalized_mapping_signature(df: pd.DataFrame):
    # Require ALL identity, membership and statistics-defining scoring columns.
    keys = ['module', 'kind', 'W0_module', 'W8_module', 'scoring_nodes']
    rows=[]
    for row in df[keys].itertuples(index=False, name=None):
        cleaned=[]
        for i,v in enumerate(row):
            if i in (2,3):
                cleaned.append(opt_id(v))
            elif i == 4:
                cleaned.append(frozenset(s.strip() for s in str(v).split(';') if s.strip()))
            else:
                cleaned.append(str(v).strip())
        rows.append(tuple(cleaned))
    return {r[0]: r[1:] for r in rows}

def locate_inputs():
    # Heatmap mapping must be sourced from the current MAIN directory.
    main_mapping = MAIN_DIR / 'module_scoring_groups.tsv'
    if not main_mapping.is_file():
        raise FileNotFoundError(
            f'Current main-heatmap mapping is missing: {main_mapping}\n'
            'Run the fixed gamma=1.2 MAIN phenotype heatmap script first. '
            'Do not silently substitute the old 6/6 mapping.'
        )
    mapping = standard_mapping(read_tsv(main_mapping))
    main_assign = [MAIN_DIR / f'{w}_module_assignment.tsv' for w in ('W0', 'W8')]
    if all(p.is_file() for p in main_assign):
        source = MAIN_DIR
        # No status logging: main prints only the finished PDF path.
    else:
        sweep_assign = [SWEEP_DIR / f'{w}_module_assignment.tsv' for w in ('W0','W8')]
        if not all(p.is_file() for p in sweep_assign):
            missing = [str(p) for p in main_assign+sweep_assign if not p.is_file()]
            raise FileNotFoundError(
                'Both MAIN and saved gamma=1.2 sweep assignments are incomplete. '
                'Missing candidates:\n  ' + '\n  '.join(missing)
            )
        sweep_mapping_path = SWEEP_DIR / 'module_scoring_groups.tsv'
        sweep_mapping = standard_mapping(read_tsv(sweep_mapping_path))
        if normalized_mapping_signature(mapping) != normalized_mapping_signature(sweep_mapping):
            raise ValueError('MAIN and gamma=1.2 sweep mapping differ; stop rather '
                             'than mixing incompatible module identifiers.')
        source = SWEEP_DIR
        # Verified in the comparison above.
    return source, mapping, main_mapping

def get_sets(week: str, assignment_dir: Path) -> dict[str, set[str]]:
    path = assignment_dir / f'{week}_module_assignment.tsv'
    df = read_tsv(path)
    if not {'node','module'}.issubset(df.columns):
        raise ValueError(f'{path}: expected node and module columns')
    df = df[['node', 'module']].copy()
    df['node'] = df['node'].astype('string').str.strip()
    df['module'] = df['module'].astype('string').str.strip()
    if df['node'].isna().any() or df['node'].eq('').any() or df['node'].duplicated().any():
        raise ValueError(f'{path}: blank or duplicated molecule IDs')
    df = df.loc[df['module'].notna() &
                ~df['module'].str.lower().isin(['', 'unassigned', 'na','nan','none'])]
    sets = {str(mid): set(part['node'].astype(str)) for mid,part in df.groupby('module')}
    if not sets:
        raise ValueError(f'{path}: no retained modules')
    small = {mid: len(nodes) for mid,nodes in sets.items() if len(nodes)<MIN_MODULE_SIZE}
    if small:
        raise ValueError(f'{path}: retained module smaller than {MIN_MODULE_SIZE}: {small}')
    return dict(sorted(sets.items(), key=lambda kv: natural_key(kv[0])))

def validate_main_mapping(df: pd.DataFrame, sets0, sets8) -> pd.DataFrame:
    """Validate saved heatmap identities and scoring sets; DO NOT rematch."""
    df = standard_mapping(df)
    if df.empty or df['module'].isna().any() or df['module'].duplicated().any():
        raise ValueError('Main mapping has missing/duplicated heatmap module IDs.')
    w0_seen, w8_seen = set(), set()
    for rec in df.to_dict('records'):
        label = str(rec['module']).strip()
        kind = str(rec['kind']).strip()
        m0 = opt_id(rec['W0_module'])
        m8 = opt_id(rec['W8_module'])
        scoring = {s.strip() for s in str(rec['scoring_nodes']).split(';') if s.strip()}
        if kind == 'Matched':
            if m0 not in sets0 or m8 not in sets8 or label!=m0:
                raise ValueError(f'Malformed W0-anchored matched mapping: {rec}')
            common = sets0[m0] & sets8[m8]
            union = sets0[m0] | sets8[m8]
            j = len(common)/len(union) if union else 0.
            if j <= MATCH_J_STRICTLY_ABOVE:
                raise ValueError(f'{label}: saved Matched pair has Jaccard {j:.4f} <= 0.40')
            expected_score = common
            if 'jaccard' in df and opt_id(rec.get('jaccard')) is not None:
                if abs(float(rec['jaccard'])-j)>1e-6:
                    raise ValueError(f'{label}: saved Jaccard disagrees with module sets')
        elif kind == 'W0_only':
            if m0 not in sets0 or m8 is not None or label!=m0:
                raise ValueError(f'Malformed W0-only mapping: {rec}')
            expected_score = sets0[m0]
        elif kind == 'W8_new':
            if m0 is not None or m8 not in sets8 or not re.fullmatch(r'N\d+', label):
                raise ValueError(f'Malformed W8-new mapping: {rec}')
            expected_score = sets8[m8]
        else:
            raise ValueError(f'Unknown heatmap mapping kind: {kind}')
        if scoring!=expected_score:
            raise ValueError(f'{label}: scoring_nodes do not match main module members '
                             f'(score {len(scoring)}, expected {len(expected_score)})')
        if m0 is not None:
            if m0 in w0_seen:
                raise ValueError(f'W0 original module mapped twice: {m0}')
            w0_seen.add(m0)
        if m8 is not None:
            if m8 in w8_seen:
                raise ValueError(f'W8 original module mapped twice: {m8}')
            w8_seen.add(m8)
    if w0_seen!=set(sets0) or w8_seen!=set(sets8):
        raise ValueError(f'Heatmap mapping does not cover all main modules. '
                         f'W0 missing={set(sets0)-w0_seen}, W8 missing={set(sets8)-w8_seen}')
    return df

def verify_final_simple(week: str, actual_sets):
    if not VERIFY_AGAINST_FINAL_SIMPLE_WHEN_AVAILABLE:
        return
    path = VERIFY_DIR / f'03_{week}_module_assignment.txt'
    if not path.is_file():
        pass  # optional verification file is absent
        return
    df = read_tsv(path)
    if not {'node','module'}.issubset(df.columns):
        raise ValueError(f'{path}: cannot verify module member sets')
    valid = df.loc[~df['module'].astype('string').str.lower().isin(
        ['unassigned','nan','na','none',''])].dropna(subset=['node','module'])
    comparison={frozenset(g['node'].astype(str).str.strip())
                for _,g in valid.groupby('module')}
    current={frozenset(members) for members in actual_sets.values()}
    if comparison != current:
        raise ValueError(f'{week}: gamma=1.2 MAIN module members differ from '
                         f'{path}; do not mix this Sankey with the Cytoscape network')
    # All module members verified.

def overlap_table(sets0, sets8, labels0, labels8, matched):
    rows=[]
    for m0,s0 in sets0.items():
        for m8,s8 in sets8.items():
            common=len(s0&s8)
            union=len(s0|s8)
            j = common/union if union else 0.
            rows.append({
                'W0_original_module': m0, 'W8_original_module': m8,
                'W0_display': labels0[m0], 'W8_display': labels8[m8],
                'W0_size':len(s0), 'W8_size':len(s8),
                'overlap_n':common, 'union_n':union, 'jaccard':j,
                'w0_retained_fraction':common/len(s0),
                'w8_retained_fraction':common/len(s8),
                'is_primary_matched': matched.get(m0)==m8,
                'draw_link': common>=MIN_FLOW_N and j>=DRAW_J_THRESHOLD,
                'label_link': common>=MIN_FLOW_N and j>=max(DRAW_J_THRESHOLD, LABEL_J_THRESHOLD),
            })
    return pd.DataFrame(rows)

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

def draw_figure(order0,order8,sets0,sets8,labels0,labels8,flows):
    nmax=max(len(order0),len(order8))
    fig, ax=plt.subplots(figsize=(FIG_WIDTH_MM/25.4, FIG_HEIGHT_MM/25.4),
                         facecolor='white')
    # Fill the physical page; padding is built into the x/y limits above.
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
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
        ax.text(NODE_X_W0-0.012, geo['mid'], label,
                fontsize=MODULE_FONT_PT, color=TEXT_COLOR,
                va='center', ha='right', zorder=8)
    for mid,geo in lo8.items():
        label=labels8[mid]+(f' ({geo["size"]})' if SHOW_MODULE_SIZES else '')
        ax.text(NODE_X_W8+NODE_WIDTH+0.012, geo['mid'], label,
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
    if DRAW_TITLE:
        fig.suptitle(TITLE,x=0.06,y=0.995,ha='left',fontsize=15,
                     color=TEXT_COLOR,weight='bold')
    return fig,segments,lo0,lo8

# 3. RUN / EXPORT
def main():

    assignment_dir, mapping, mapping_path = locate_inputs()
    sets0 = get_sets('W0', assignment_dir)
    sets8 = get_sets('W8', assignment_dir)
    mapping = validate_main_mapping(mapping, sets0, sets8)
    for week, sets in (('W0',sets0),('W8',sets8)):
        n = len(sets)
        retained = sum(len(s) for s in sets.values())
        if n != EXPECTED_MODULE_COUNTS[week]:
            raise ValueError(f'{week}: main currently has {n} modules, '
                             f'expected {EXPECTED_MODULE_COUNTS[week]}; '
                             'check if the MAIN dataset changed.')
        if retained != EXPECTED_RETAINED_NODES[week]:
            raise ValueError(f'{week}: main currently has {retained} retained nodes, '
                             f'expected {EXPECTED_RETAINED_NODES[week]}; '
                             'check if the MAIN dataset changed.')
        # Validation passed.
        verify_final_simple(week,sets)

    labels0,labels8,matched={}, {}, {}
    for rec in mapping.to_dict('records'):
        m0=opt_id(rec['W0_module'])
        m8=opt_id(rec['W8_module'])
        label=str(rec['module']).strip()
        if m0 is not None: labels0[m0]=label
        if m8 is not None: labels8[m8]=label
        if str(rec['kind'])=='Matched': matched[m0]=m8

    unknown_ids = (set(labels0.values()) | set(labels8.values())) - set(MODULE_BG_COLORS)
    if unknown_ids:
        raise ValueError(f'Missing module palette colors for IDs: {sorted(unknown_ids)}')

    # Follow the exact module row ORDER used by the main phenotype heatmap.
    order0=[opt_id(r['W0_module']) for r in mapping.to_dict('records')
            if opt_id(r['W0_module']) is not None]
    order8=[opt_id(r['W8_module']) for r in mapping.to_dict('records')
            if opt_id(r['W8_module']) is not None]
    if len(set(labels0.values()))!=len(labels0) or len(set(labels8.values()))!=len(labels8):
        raise ValueError('Repeated display IDs on one side')

    all_pairs=overlap_table(sets0,sets8,labels0,labels8,matched)
    drawn=all_pairs.loc[all_pairs['draw_link']].copy()
    fig,segments,geo0,geo8=draw_figure(order0,order8,sets0,sets8,labels0,labels8,drawn)

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    basename='07_static_W0_W8_module_sankey_HEATMAP_MATCHED'
    paths={ext: OUT_DIR/f'{basename}.{ext}' for ext in ('pdf','svg','png')}
    for ext,p in paths.items():
        fig.savefig(p,dpi=DPI,facecolor='white')
    all_pairs.to_csv(OUT_DIR/'07_all_W0_W8_overlap_Jaccard.tsv',sep='\t',index=False)
    drawn.to_csv(OUT_DIR/'07_drawn_ribbon_pairs.tsv',sep='\t',index=False)
    segments.to_csv(OUT_DIR/'07_drawn_ribbon_vector_coordinates.tsv',sep='\t',index=False)

    # Saved heatmap mapping is copied with its ORIGINAL column names and values.
    # It is never changed/reconstructed; W0/W8 graph/module files are untouched.
    mapping.to_csv(OUT_DIR/'07_mapping_USED_UNCHANGED.tsv',sep='\t',index=False)

    covered0={m:sum(int(r['overlap_n']) for r in drawn.to_dict('records')
                    if r['W0_original_module']==m) for m in order0}
    covered8={m:sum(int(r['overlap_n']) for r in drawn.to_dict('records')
                    if r['W8_original_module']==m) for m in order8}
    coverage=pd.DataFrame(
        [{'week':'W0','module_original':m,'display_id':labels0[m],
          'total_n':len(sets0[m]),'ribbon_n':covered0[m],
          'unshown_n':len(sets0[m])-covered0[m]} for m in order0]
        +[{'week':'W8','module_original':m,'display_id':labels8[m],
          'total_n':len(sets8[m]),'ribbon_n':covered8[m],
          'unshown_n':len(sets8[m])-covered8[m]} for m in order8])
    coverage.to_csv(OUT_DIR/'07_module_ribbon_coverage.tsv',sep='\t',index=False)
    with open(OUT_DIR/'07_plot_parameters.txt','w',encoding='utf8') as f:
        for key,val in {
            'network_dir':str(assignment_dir),
            'heatmap_mapping_file':str(mapping_path),
            'source': 'MAIN gamma=1.2, weighted positive-edge Louvain, minimum size=10',
            'module_counts': f'W0={len(sets0)},W8={len(sets8)}',
            'matching':'EXISTING main phenotype heatmap Hungarian-Jaccard mapping >0.40, unchanged',
            'draw_min_overlap':MIN_FLOW_N,'draw_min_Jaccard':DRAW_J_THRESHOLD,
            'label_min_Jaccard':LABEL_J_THRESHOLD,
            'ribbon_width':'shared-node count / SAME vertical scale on W0,W8',
            'ribbon_gradient': 'source display-module background to target display-module background',
             'figure_width_mm': FIG_WIDTH_MM, 'figure_height_mm': FIG_HEIGHT_MM,
            'gradient_format':'Rasterized gradient clipped to editable Bezier paths inside PDF/SVG',
        }.items(): f.write(f'{key}\t{val}\n')
    print(paths['pdf'].resolve())
    if SHOW_FIGURE:
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import display
                display(fig)
            elif 'agg' not in plt.get_backend().lower():
                plt.show()
        except ImportError:
            if 'agg' not in plt.get_backend().lower():
                plt.show()
    plt.close(fig)
    return all_pairs,mapping

if __name__=='__main__':
    main()
