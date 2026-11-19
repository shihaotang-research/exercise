#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 5: acute exercise-response module enrichment plot."""
from __future__ import annotations
import argparse
import sys
from collections import OrderedDict
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib import font_manager

ROOT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
INPUT_DIR = ROOT / 'acute_module_enrichment'
OUTPUT_DIR = ROOT / 'Dynamic_Delta3TP_module_enrichment_horizontal_143mm'
OUTPUT_STEM = 'Dynamic_Delta3TP_module_enrichment_horizontal_143mm_module_palette_FIXED'
FIGURE_MM = (143.0, 86.0)  # Identical static plot physical dimensions
FONT_SIZE = 7.0
FDR_CUTOFF = 0.05
MIN_HIT = 3
DOT_AREA = 21
WEEK_COLORS = {'W0':'#2E8AC4', 'W8':'#E64B87'}
STRIP_FILL = {'A':'#E8EFF4', 'W0':'#E7F1F8', 'W8':'#FAE9F0'}
STRIP_ACCENT = {'A':'#6F879B', 'W0':'#2E8AC4', 'W8':'#E64B87'}

# All term IDs below were verified against the newly uploaded MAIN/FINAL tables.
# Each entry: (feature type, [(ID, short label), ...]).
MATCHED = OrderedDict([
    ('M1', ('Protein', [
        ('GO:0006959', 'Humoral immune response'),
        ('GO:0030198', 'ECM organization'),
        ('GO:0006953', 'Acute-phase response')])),
    ('M2', ('Protein', [
        ('GO:0030168', 'Platelet activation'),
        ('GO:0008104', 'Protein localization'),
        ('GO:0006836', 'Neurotransmitter transport')])),
    ('M3', ('Protein', [
        ('GO:0006412', 'Translation'),
        ('GO:0022613', 'RNP complex biogenesis'),
        ('GO:0007264', 'Small GTPase signaling')])),
    ('M4', ('Metabolite', [
        ('Amino acids and peptides', 'Amino acids and peptides'),
        ('Purines', 'Purines')])),
    ('M5', ('Metabolite', [
        ('Fatty esters', 'Fatty esters'),
        ('Fatty acids', 'Fatty acids')])),
    ('M6', ('Protein', [
        ('GO:0006396', 'RNA processing'),
        ('GO:0002181', 'Cytoplasmic translation'),
        ('GO:0006334', 'Nucleosome assembly')])),
    ('M7', ('Metabolite', [
        ('Glycerophosphocholines', 'Glycerophosphocholines'),
        ('Glycerophosphoethanolamines', 'Glycerophosphoethanolamines'),
        ('Phosphosphingolipids', 'Phosphosphingolipids')])),
])
UNMATCHED = OrderedDict([
    (('W0','M8'), [
        ('GO:0000910','Cytokinesis'),
        ('GO:0016197','Endosomal transport')]),
    (('W0','M9'), [
        ('GO:0045333','Cellular respiration'),
        ('GO:0007005','Mitochondrion organization'),
        ('GO:0006635','Fatty acid β-oxidation')]),
    (('W8','N1'), [
        ('GO:0070371','ERK1/2 cascade'),
        ('GO:0019221','Cytokine signaling'),
        ('GO:0071456','Hypoxia response')]),
])

def file_path(folder:Path, pattern:str)->Path:
    matches=sorted(folder.glob(pattern))
    if len(matches)!=1:
        raise FileNotFoundError(f'Expected one file matching {pattern!r} in {folder}, found {matches}')
    return matches[0]

def read_required(folder, pattern, fields):
    path=file_path(folder,pattern)
    d=pd.read_csv(path,sep='\t',low_memory=False)
    d.columns=d.columns.astype(str).str.strip()
    missing=set(fields)-set(d.columns)
    if missing:raise ValueError(f'{path}: missing {sorted(missing)}')
    for col in ['network','module','display_id','module_W0','module_W8','ID','main_class']:
        if col in d.columns:d[col]=d[col].astype('string').str.strip()
    return d

def prepare(folder:Path):
    pairs=read_required(folder,'02B_primary_one_to_one_matched_pairs*.txt',
                        ['display_id','module_W0','module_W8','jaccard'])
    protein=read_required(folder,'08_each_module_enrichment_FDR005.txt',
                          ['network','module','display_id','ID','Description','p.adjust','Count'])
    prot_both=read_required(folder,'10_primary_matched_pairs_shared_terms_BOTH_FDR005.txt',
                            ['display_id','ID','W0_p.adjust','W8_p.adjust','W0_Count','W8_Count'])
    nonprimary=read_required(folder,'14_nonprimary_module_enrichment_FDR005.txt',
                             ['network','module','display_id','ID','p.adjust','Count'])
    metab=read_required(folder,'16G_each_module_metabolite_mainclass_FDR005_count3.txt',
                        ['network','module','display_id','main_class','FDR_within_module','count_in_module'])
    metab_both=read_required(folder,'17B_primary_pairs_metabolite_mainclass_BOTH_FDR005.txt',
                             ['display_id','main_class','W0_FDR','W8_FDR','W0_count','W8_count'])
    if set(pairs.display_id)!=set(MATCHED) or len(pairs)!=7:
        raise ValueError(f'MAIN mapping mismatch: expect exactly M1-M7, got {list(pairs.display_id)}')
    if pairs.duplicated(['display_id']).any():raise ValueError('Duplicated display IDs')
    if any(pairs['module_'+week].duplicated().any() for week in ['W0','W8']):
        raise ValueError('Primary pairs are not one-to-one')
    # Validate full displayed ID vs actual original week-module membership.
    for week in ['W0','W8']:
        src=pd.concat([protein,metab],ignore_index=True)
        for r in pairs.itertuples(index=False):
            block=src.loc[(src.network==week)&(src.display_id==r.display_id)]
            if not block.empty and not block.module.eq(getattr(r,'module_'+week)).all():
                raise ValueError(f'Inconsistent original module for {week}/{r.display_id}')
    if set(nonprimary.display_id)!={'M8','M9','N1'}:
        raise ValueError('Nonprimary modules not expected W0 M8, W0 M9, W8 N1')

    records=[]
    def one(d,cond,what):
        x=d.loc[cond]
        if len(x)!=1:raise ValueError(f'{what}: expected exactly one result, got {len(x)}')
        return x.iloc[0]
    def add(panel,group,omics,term,label,w,q,hits,reason):
        if not (np.isfinite(q) and 0 < q < FDR_CUTOFF and int(hits)>=MIN_HIT):
            raise ValueError(f'Not significant or missing min-hit: {panel}/{group}/{w}/{term}: q={q}, hits={hits}')
        records.append(dict(panel=panel,group=group,omics=omics,term=term,label=label,
                            week=w,q=float(q),hit=int(hits),reason=reason))

    for display,(omics,terms) in MATCHED.items():
        pair=one(pairs,pairs.display_id.eq(display),f'pair {display}')
        for term,label in terms:
            if omics=='Protein':
                both=prot_both.loc[prot_both.display_id.eq(display)&prot_both.ID.eq(term)]
                if not both.empty:
                    r=one(prot_both,prot_both.display_id.eq(display)&prot_both.ID.eq(term),f'{display}/{term}')
                    for w in ['W0','W8']:
                        add('A',display,omics,term,label,w,float(r[f'{w}_p.adjust']),r[f'{w}_Count'],'both')
                else:
                    # Single-week FDR only allowed with a verified row in input FDR table.
                    for w in ['W0','W8']:
                        x=protein.loc[protein.display_id.eq(display)&protein.network.eq(w)&protein.ID.eq(term)]
                        if len(x)>1:raise ValueError(f'Nonunique {display}/{w}/{term}')
                        if len(x)==1:
                            r=x.iloc[0]
                            if r.module!=pair['module_'+w]:raise ValueError(f'Module mismatch for {w}/{display}/{term}')
                            add('A',display,omics,term,label,w,float(r['p.adjust']),r['Count'],'one-week')
            else:
                both=metab_both.loc[metab_both.display_id.eq(display)&metab_both.main_class.eq(term)]
                if not both.empty:
                    r=one(metab_both,metab_both.display_id.eq(display)&metab_both.main_class.eq(term),
                        f'{display}/{term}')
                    for w in ['W0','W8']:
                        add('A',display,omics,term,label,w,float(r[f'{w}_FDR']),r[f'{w}_count'],'both')
                else:
                    for w in ['W0','W8']:
                        x=metab.loc[metab.display_id.eq(display)&metab.network.eq(w)&metab.main_class.eq(term)]
                        if len(x)>1:raise ValueError(f'Nonunique {display}/{w}/{term}')
                        if len(x)==1:
                            r=x.iloc[0]
                            if r.module!=pair['module_'+w]:raise ValueError(f'Module mismatch for {w}/{display}/{term}')
                            add('A',display,omics,term,label,w,float(r['FDR_within_module']),r.count_in_module,
                                'one-week')
            if not any(r['panel']=='A' and r['group']==display and r['term']==term for r in records):
                raise ValueError(f'No FDR-supported result found for {display}/{term}')

    for (week,display),terms in UNMATCHED.items():
        for term,label in terms:
            r=one(nonprimary,nonprimary.network.eq(week)&nonprimary.display_id.eq(display)&nonprimary.ID.eq(term),
                  f'nonprimary {week}/{display}/{term}')
            add('B',f'{week} {display}','Protein',term,label,week,float(r['p.adjust']),r['Count'],'unmatched')

    out=pd.DataFrame(records)
    if out.duplicated(['panel','group','term','week']).any():
        raise ValueError('Duplicate selected data points')
    out['minus_log10_FDR']=-np.log10(out.q.clip(lower=1e-300))
    return out

# Same dynamic palette as the network/Sankey legend (M1-M9, N1).
MODULE_BG_COLORS = {
    'M1':'#F4C7C3', 'M2':'#BFD7F2', 'M3':'#BFE3C0',
    'M4':'#F3E2A7', 'M5':'#D9C8F0', 'M6':'#BFE3EA',
    'M7':'#F7D6BF', 'M8':'#C9D0D8', 'M9':'#D6E8C5',
    'N1':'#F6D1E6',
}
MODULE_EDGE_COLORS = {
    'M1':'#E78B84', 'M2':'#7FB0E3', 'M3':'#7FC68A',
    'M4':'#D8B94F', 'M5':'#B497DE', 'M6':'#7DC8D6',
    'M7':'#E5AE7B', 'M8':'#93A0AD', 'M9':'#9FC37A',
    'N1':'#E49BC4',
}

# Absolute sizes in millimeters, adapted from the STATIC 143 x 86 mm figure.
PANEL_LEFT_X_MM = 2.8
PANEL_RIGHT_X_MM = 72.8
PANEL_W_MM = 67.6
MODULE_W_MM = 6.7
GAP_AFTER_MODULE_MM = 0.5
TERMS_W_MM = 39.0
GAP_BEFORE_DOTS_MM = 0.6
DOTS_W_MM = PANEL_W_MM - MODULE_W_MM - GAP_AFTER_MODULE_MM - TERMS_W_MM - GAP_BEFORE_DOTS_MM
PLOT_TOP_MM = 14.0
PLOT_H_MM = 59.0
GROUP_GAP_MM = 0.70

def init_style():
    # Do not silently substitute Liberation Sans/DejaVu. This makes the Arial
    # requirement explicit and prevents findfont: Liberation Sans not found.
    font_manager.findfont(font_manager.FontProperties(family='Arial'), fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.sans-serif': ['Arial'],
        'font.size': FONT_SIZE, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.edgecolor': '#747B82',
        'axes.linewidth': .7, 'axes.unicode_minus': False,
        'text.color': '#20252A',
    })

def mm_box(x, top, width, height, fig_w, fig_h):
    return [x/fig_w, (fig_h-top-height)/fig_h, width/fig_w, height/fig_h]

def unique_terms_in_order(data, group):
    sub = data.loc[data.group.eq(group)]
    seen = set()
    result = []
    for term, label in sub[['term','label']].itertuples(index=False, name=None):
        if term not in seen:
            result.append((term,label))
            seen.add(term)
    return result

def draw_panel(fig, x0_mm, title, data, groups, xmax, fig_w, fig_h, label_checks):
    bg = fig.add_axes(mm_box(x0_mm,PLOT_TOP_MM,PANEL_W_MM,PLOT_H_MM,fig_w,fig_h),zorder=0)
    bg.set_xlim(0,PANEL_W_MM); bg.set_ylim(PLOT_H_MM,0); bg.axis('off')

    mod_x = x0_mm
    lab_x = mod_x + MODULE_W_MM + GAP_AFTER_MODULE_MM
    dot_x = lab_x + TERMS_W_MM + GAP_BEFORE_DOTS_MM
    mod = fig.add_axes(mm_box(mod_x,PLOT_TOP_MM,MODULE_W_MM,PLOT_H_MM,fig_w,fig_h),zorder=2)
    lab = fig.add_axes(mm_box(lab_x,PLOT_TOP_MM,TERMS_W_MM,PLOT_H_MM,fig_w,fig_h),zorder=2)
    dots = fig.add_axes(mm_box(dot_x,PLOT_TOP_MM,DOTS_W_MM,PLOT_H_MM,fig_w,fig_h),zorder=3)
    for ax in (mod,lab,dots):
        ax.patch.set_alpha(0)
        ax.set_ylim(PLOT_H_MM,0)
    mod.set_xlim(0,MODULE_W_MM); mod.axis('off')
    lab.set_xlim(0,TERMS_W_MM); lab.axis('off')

    fig.text(x0_mm/fig_w,1-12.2/fig_h,title,fontsize=8.8,fontweight='bold',ha='left',va='bottom')
    group_terms = [unique_terms_in_order(data, group) for group in groups]
    nrows = sum(len(rows) for rows in group_terms)
    if nrows == 0: raise RuntimeError(f'{title}: no selected pathways')
    row_h = (PLOT_H_MM-GROUP_GAP_MM*(len(groups)-1))/nrows
    if row_h < 2.75:
        raise ValueError(f'{title}: {nrows} pathway rows need more vertical space; current row height={row_h:.2f} mm. Increase PLOT_H_MM or reduce GROUP_GAP_MM.')
    cursor=0.0
    for gi,(group,terms) in enumerate(zip(groups,group_terms)):
        height=row_h*len(terms)
        bg.add_patch(Rectangle((0,cursor),PANEL_W_MM,height,facecolor=('#F3F5F6' if gi%2 else '#FFFFFF'),
            edgecolor='none'))
        mid = group.split()[-1]  # e.g. W0 M8 -> M8
        bg.add_patch(Rectangle((0,cursor),MODULE_W_MM,height,facecolor=MODULE_BG_COLORS[mid],edgecolor='none'))
        bg.add_patch(Rectangle((0,cursor),.42,height,facecolor=MODULE_EDGE_COLORS[mid],edgecolor='none'))
        mod.text(MODULE_W_MM/2+.15,cursor+height/2,mid,ha='center',va='center',fontsize=7.0,fontweight='bold')
        for j,(term,label) in enumerate(terms):
            yy=cursor+(j+.5)*row_h
            art=lab.text(.3,yy,label,ha='left',va='center',fontsize=7.0,color='#20252A')
            label_checks.append((art,lab,dots))
            rows=data.loc[data.group.eq(group)&data.term.eq(term)]
            if len(rows)==2:
                v=rows.set_index('week')['minus_log10_FDR']
                if {'W0','W8'}.issubset(v.index):
                    dots.plot([v['W0'],v['W8']],[yy,yy],color='#A9B6BF',lw=.78,zorder=2)
            for r in rows.itertuples(index=False):
                dots.scatter(r.minus_log10_FDR,yy,marker='o' if r.omics=='Protein' else '^',
                             s=21 if r.omics=='Protein' else 28,c=[WEEK_COLORS[r.week]],
                             edgecolors='white',linewidths=.35,zorder=4,clip_on=True)
        cursor+=height+(GROUP_GAP_MM if gi<len(groups)-1 else 0)
    dots.set_xlim(0,xmax)
    dots.axvline(-np.log10(FDR_CUTOFF),lw=.8,color='#A4AFB8',ls=(0,(2.5,2.4)),zorder=1)
    import math
    step=4 if xmax<=20 else int(math.ceil(xmax/5/2)*2)
    dots.set_xticks(np.arange(0,xmax+.01,step))
    dots.set_yticks([])
    dots.tick_params(axis='x',width=.65,length=2.5,labelsize=7,pad=1.6,colors='#333A40')
    for edge in ('left','top','right'): dots.spines[edge].set_visible(False)
    dots.spines['bottom'].set_color('#4C5760'); dots.spines['bottom'].set_linewidth(.8)
    dots.set_xlabel(r'$-\log_{10}(\mathrm{FDR})$',fontsize=7.2,labelpad=2.0)

def draw(records, output_dir: Path, stem=OUTPUT_STEM, size_mm=FIGURE_MM):
    init_style()
    output_dir.mkdir(parents=True,exist_ok=True)
    fw,fh=map(float,size_mm)
    if (fw,fh)!=(143.0,86.0):
        raise ValueError('To match static layout exactly, the page must be 143 x 86 mm.')
    valid=records['minus_log10_FDR'].replace([np.inf,-np.inf],np.nan).dropna()
    if valid.empty: raise ValueError('No significant enrichment points')
    import math
    xmax=max(4,int(math.ceil((float(valid.max())+.7)/4))*4)
    fig=plt.figure(figsize=(fw/25.4,fh/25.4),facecolor='white')
    labels=[]
    draw_panel(fig,PANEL_LEFT_X_MM,'Preserved modules',records.loc[records.panel.eq('A')],list(MATCHED),xmax,fw,fh,
        labels)
    unmatched=['W0 M8','W0 M9','W8 N1']
    draw_panel(fig,PANEL_RIGHT_X_MM,'Unmatched modules',records.loc[records.panel.eq('B')],unmatched,xmax,fw,fh,
        labels)
    handles=[Line2D([],[],ls='',marker='o',ms=5.0,mfc=WEEK_COLORS[w],mec='none',label=w) for w in ('W0','W8')]
    fig.legend(handles=handles,ncol=2,loc='center',bbox_to_anchor=(.89,.948),frameon=False,
               fontsize=7,handletextpad=.35,columnspacing=1.)
    kind_handles=[
       Line2D([],[],ls='',marker='o',ms=4.6,mfc='#777777',mec='none',label='Protein GO'),
       Line2D([],[],ls='',marker='^',ms=4.8,mfc='#777777',mec='none',label='Metabolite class'),
       Line2D([],[],ls=(0,(2.5,2.4)),lw=.8,color='#A4AFB8',label='FDR = 0.05')]
    fig.legend(handles=kind_handles,ncol=3,loc='lower center',bbox_to_anchor=(.5,.012),
               frameon=False,fontsize=7,handlelength=1.8,handletextpad=.43,columnspacing=1.6)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for txt,term_ax,dot_ax in labels:
        t=txt.get_window_extent(renderer)
        d=dot_ax.get_window_extent(renderer)
        if t.x1>=d.x0-1 or t.x1>=term_ax.get_window_extent(renderer).x1-.3:
            plt.close(fig)
            raise RuntimeError('Pathway label exceeds its allotted area: '+txt.get_text())
    pdf=output_dir/(stem+'.pdf')
    png=output_dir/(stem+'.png')
    # Never bbox_inches='tight': keep exported PDF at exactly 143 x 86 mm.
    fig.savefig(pdf,dpi=400,facecolor='white',metadata={'Title':'Dynamic module enrichment; static-matched 143x86mm'})
    fig.savefig(png,dpi=400,facecolor='white')
    plt.close(fig)
    return pdf

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,default=INPUT_DIR)
    p.add_argument('--output-dir',type=Path,default=OUTPUT_DIR)
    if argv is None and ('ipykernel' in sys.modules or Path(sys.argv[0]).name=='ipykernel_launcher.py'):
        argv=[]
    args=p.parse_args(argv)
    records=prepare(args.input_dir)
    pdf=draw(records,args.output_dir)
    records.to_csv(args.output_dir/(OUTPUT_STEM+'_selected_terms.tsv'),sep='\t',index=False)
    print(str(pdf.resolve()))

if __name__=='__main__':
    main()
