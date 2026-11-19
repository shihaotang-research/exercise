"""Week 0/week 8 resting-state Spearman networks and positive-edge Louvain modules.
Significant edges require BH-FDR < 0.05 and |rho| >= 0.50."""
import os
from pathlib import Path
import re
import warnings
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore')
WORKDIR = str(Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))))
OUT_DIR = 'resting_network'
PRO_FILE = 'protein_matrix.tsv'
MET_FILE = 'metabolite_matrix.tsv'
REF_FILE = 'protein_annotations.tsv'
PROTEIN_LIST_FILES = {'Baseline-only': 'protein_baseline_only.tsv', 'Trend-only': 'protein_dynamic_only.tsv',
    'Dual-response': 'protein_dual_response.tsv'}
METABOLITE_LIST_FILES = {'Baseline-only': 'metabolite_baseline_only.tsv',
    'Trend-only': 'metabolite_dynamic_only.tsv', 'Dual-response': 'metabolite_dual_response.tsv'}
STATIC_SPEC = {'W0': 'A', 'W8': 'D'}
ALL_TIMEPOINTS = ['A', 'B1', 'B2', 'B3', 'D', 'F1', 'F2', 'F3']
MIN_SUBJECTS = 10
MIN_FEATURE_SD = 1e-12
FDR_CUTOFF = 0.05
MIN_ABS_RHO = 0.5
COMMUNITY_RESOLUTION = 1.2
MIN_MODULE_SIZE = 10
SEED = 2026
np.random.seed(SEED)

def outpath(name):
    return os.path.join(OUT_DIR, name)

def save_tsv(df, name):
    df.to_csv(outpath(name), sep='\t', index=False)

def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Required file not found: {path}')

def read_table_auto(path, dtype=str):
    require_file(path)
    last_error = None
    for kwargs in [{'sep': '\t'}, {'sep': ','}, {'sep': None, 'engine': 'python'}]:
        try:
            df = pd.read_csv(path, dtype=dtype, **kwargs)
            if df.shape[1] > 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Cannot read {path}: {last_error}')

def resolve_column(columns, candidates):
    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]
    raise ValueError('Cannot resolve column from: ' + ', '.join(candidates))

def natural_key(x):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\\d+)', str(x))]

def read_list_file(path):
    require_file(path)
    x = pd.read_csv(path, sep='\t', header=None, dtype=str, comment='#').iloc[:, 0]
    x = x.dropna().astype(str).str.strip()
    bad = {'protein', 'metabolite', 'molecule', 'node', 'id', 'feature', 'baseline_only', 'baseline-only',
        'baseline only', 'trend_only', 'trend-only', 'trend only', 'both', 'dual-response', 'dual response'}
    return [v for v in x if v and v.lower() not in bad]

def bh_fdr(pvalues):
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q

def load_expression(path, name):
    df = read_table_auto(path, dtype=str)
    id_col = resolve_column(df.columns, ['id', 'ID', 'Individual', 'individual', 'Subject', 'subject'])
    time_col = resolve_column(df.columns, ['time', 'Time', 'TIME', 'timepoint', 'Timepoint'])
    df[id_col] = df[id_col].astype(str).str.strip()
    df[time_col] = df[time_col].astype(str).str.strip()
    print(f'[{name}] rows={len(df)}, columns={df.shape[1]}, ID={id_col}, Time={time_col}')
    return (df, id_col, time_col)

def load_reference():
    require_file(REF_FILE)
    ref = pd.read_csv(REF_FILE, sep='\t', header=None, dtype=str)
    if ref.shape[1] < 2:
        return {}
    ref = ref.iloc[:, :2].copy()
    ref.columns = ['node', 'label']
    ref['node'] = ref['node'].astype(str).str.strip()
    ref['label'] = ref['label'].astype(str).str.strip()
    ref = ref.dropna().drop_duplicates('node')
    return dict(zip(ref['node'], ref['label']))

def build_annotation(pro_df, met_df, id_map):
    rows = []
    for group, path in PROTEIN_LIST_FILES.items():
        for node in read_list_file(path):
            rows.append({'node': node, 'label': id_map.get(node, node), 'moltype': 'Protein', 'group': group,
                'present': node in pro_df.columns})
    for group, path in METABOLITE_LIST_FILES.items():
        for node in read_list_file(path):
            rows.append({'node': node, 'label': id_map.get(node, node), 'moltype': 'Metabolite', 'group': group,
                'present': node in met_df.columns})
    ann = pd.DataFrame(rows)
    if ann.empty:
        raise RuntimeError('No molecules found in class-list files.')
    duplicated = ann[ann.duplicated('node', keep=False)].copy()
    if not duplicated.empty:
        conflict = duplicated.groupby('node').agg(n_type=('moltype', 'nunique'), n_group=('group', 'nunique'))
        conflict = conflict[(conflict['n_type'] > 1) | (conflict['n_group'] > 1)]
        if not conflict.empty:
            save_tsv(duplicated, '00_duplicate_selected_nodes.txt')
            raise RuntimeError('Conflicting duplicated nodes.')
        ann = ann.drop_duplicates('node', keep='first')
    missing = ann.loc[~ann['present']].copy()
    if not missing.empty:
        save_tsv(missing, '00_selected_molecules_missing_from_matrices.txt')
    ann = ann.loc[ann['present']].copy().reset_index(drop=True)
    protein_nodes = set(ann.loc[ann['moltype'] == 'Protein', 'node'])
    metabolite_nodes = set(ann.loc[ann['moltype'] == 'Metabolite', 'node'])
    overlap = protein_nodes & metabolite_nodes
    if overlap:
        raise RuntimeError('Protein/metabolite node-name overlap: ' + ', '.join(sorted(overlap)[:20]))
    ann = ann.sort_values(['moltype', 'group', 'node'], key=lambda s: s.map(str)).reset_index(drop=True)
    save_tsv(ann, '00_selected_molecules_matched.txt')
    return ann

def subjects_with_times(df, id_col, time_col):
    time_sets = df.groupby(id_col)[time_col].apply(lambda x: set(x.astype(str)))
    required = set(ALL_TIMEPOINTS)
    return set(time_sets.index[time_sets.map(lambda x: required.issubset(x))].astype(str))

def common_subjects(pro_df, pro_id, pro_time, met_df, met_id, met_time):
    subjects = sorted(subjects_with_times(pro_df, pro_id, pro_time) & subjects_with_times(met_df, met_id,
        met_time), key=natural_key)
    if len(subjects) < MIN_SUBJECTS:
        raise RuntimeError(f'Only {len(subjects)} complete common subjects remain.')
    save_tsv(pd.DataFrame({'subject': subjects}), '00_common_subjects_used.txt')
    return subjects

def extract_time(df, id_col, time_col, tp, subjects, features):
    x = df.loc[df[time_col].astype(str).eq(tp) & df[id_col].astype(str).isin(subjects),
        [id_col] + list(features)].copy()
    for feature in features:
        x[feature] = pd.to_numeric(x[feature], errors='coerce')
    x[id_col] = x[id_col].astype(str)
    x = x.groupby(id_col)[list(features)].mean().reindex(subjects)
    x.index.name = 'subject'
    return x

def build_static_inputs(pro_df, pro_id, pro_time, met_df, met_id, met_time, ann, subjects):
    protein_nodes = ann.loc[ann['moltype'] == 'Protein', 'node'].tolist()
    metabolite_nodes = ann.loc[ann['moltype'] == 'Metabolite', 'node'].tolist()
    node_order = ann['node'].tolist()

    def combined_time(tp):
        parts = []
        if protein_nodes:
            parts.append(extract_time(pro_df, pro_id, pro_time, tp, subjects, protein_nodes))
        if metabolite_nodes:
            parts.append(extract_time(met_df, met_id, met_time, tp, subjects, metabolite_nodes))
        return pd.concat(parts, axis=1).reindex(index=subjects, columns=node_order)
    matrices_all = {network: combined_time(tp) for network, tp in STATIC_SPEC.items()}
    qc_rows = []
    for node in node_order:
        row = {'node': node}
        usable_both = True
        for network in ['W0', 'W8']:
            values = matrices_all[network][node].to_numpy(dtype=float)
            finite = np.isfinite(values)
            sd = np.std(values[finite], ddof=1) if finite.sum() > 1 else np.nan
            usable = bool(finite.all() and np.isfinite(sd) and (sd > MIN_FEATURE_SD))
            row[f'n_finite_{network}'] = int(finite.sum())
            row[f'sd_{network}'] = sd
            row[f'usable_{network}'] = usable
            usable_both &= usable
        row['usable_in_both'] = usable_both
        qc_rows.append(row)
    qc = pd.DataFrame(qc_rows).merge(ann[['node', 'label', 'moltype', 'group']], on='node', how='left')
    save_tsv(qc, '00_feature_QC_same_set_W0_W8.txt')
    usable_nodes = qc.loc[qc['usable_in_both'], 'node'].tolist()
    if len(usable_nodes) < 2:
        raise RuntimeError('Fewer than two usable molecules.')
    ann_used = ann.set_index('node').loc[usable_nodes].reset_index()
    save_tsv(ann_used, '00_selected_molecules_final_same_set.txt')
    matrices = {}
    for network in ['W0', 'W8']:
        matrix = matrices_all[network].loc[subjects, usable_nodes].copy()
        matrices[network] = matrix
        matrix.reset_index().to_csv(outpath(f'01_{network}_subject_by_molecule_static_matrix.txt'), sep='\t',
            index=False)
    return (matrices, ann_used)

def infer_network(matrix, network, ann):
    if matrix.isna().any().any():
        raise RuntimeError(f'{network}: missing values in network matrix.')
    values = matrix.to_numpy(dtype=float)
    n_subjects, n_features = values.shape
    ranks = pd.DataFrame(values).rank(axis=0, method='average').to_numpy(dtype=float)
    rho_matrix = np.corrcoef(ranks, rowvar=False)
    rho_matrix = np.clip(rho_matrix, -1.0, 1.0)
    np.fill_diagonal(rho_matrix, 1.0)
    nodes = matrix.columns.to_numpy(dtype=str)
    pd.DataFrame(rho_matrix, index=nodes, columns=nodes).to_csv(outpath(f'02_{network}_Spearman_rho_matrix.txt'),
        sep='\t', index_label='node')
    iu, ju = np.triu_indices(n_features, k=1)
    rho = rho_matrix[iu, ju]
    dfree = n_subjects - 2
    denominator = np.maximum(1.0 - rho ** 2, np.finfo(float).tiny)
    t_values = rho * np.sqrt(dfree / denominator)
    pvalue = 2.0 * student_t.sf(np.abs(t_values), df=dfree)
    pvalue[np.isclose(np.abs(rho), 1.0)] = 0.0
    fdr = bh_fdr(pvalue)
    keep = (fdr < FDR_CUTOFF) & (np.abs(rho) >= MIN_ABS_RHO)
    edges = pd.DataFrame({'source': nodes[iu[keep]], 'target': nodes[ju[keep]], 'rho': rho[keep],
        'abs_rho': np.abs(rho[keep]), 'pvalue': pvalue[keep], 'fdr': fdr[keep]})
    edges['sign'] = np.where(edges['rho'] > 0, 'Positive', 'Negative')
    annotation_map = ann.set_index('node')
    for side in ['source', 'target']:
        edges[f'{side}_label'] = edges[side].map(annotation_map['label'])
        edges[f'{side}_moltype'] = edges[side].map(annotation_map['moltype'])
        edges[f'{side}_group'] = edges[side].map(annotation_map['group'])
    edges['edge_type'] = ['-'.join(sorted([a, b])) for a, b in zip(edges['source_moltype'], edges['target_moltype'])]
    edges = edges.sort_values(['fdr', 'abs_rho'], ascending=[True, False]).reset_index(drop=True)
    save_tsv(edges, f'02_{network}_significant_edges_all_signs.txt')
    print(f'\n[{network}] subjects={n_subjects}, molecules={n_features}, all_pairs={len(rho):,}')
    print(f"[{network}] significant={len(edges):,}; positive={int(edges['rho'].gt(0).sum()):,}; negative={int(edges['rho'].lt(0).sum()):,}")
    return edges

def make_graph(ann, edges):
    graph = nx.Graph()
    for _, row in ann.iterrows():
        graph.add_node(row['node'], label=row['label'], moltype=row['moltype'], group=row['group'])
    for _, row in edges.iterrows():
        graph.add_edge(row['source'], row['target'], rho=float(row['rho']), abs_rho=float(row['abs_rho']),
            pvalue=float(row['pvalue']), fdr=float(row['fdr']), sign=row['sign'])
    return graph

def detect_modules(graph, network, ann, edges):
    positive_graph = nx.Graph()
    positive_graph.add_nodes_from(graph.nodes(data=True))
    positive_graph.add_edges_from([(u, v, d) for u, v, d in graph.edges(data=True) if d['rho'] > 0])
    active_positive_nodes = [node for node, degree in positive_graph.degree() if degree > 0]
    active_positive_graph = positive_graph.subgraph(active_positive_nodes).copy()
    communities_all = []
    if active_positive_graph.number_of_edges() > 0:
        communities_all = list(nx.community.louvain_communities(active_positive_graph, weight='abs_rho',
            resolution=COMMUNITY_RESOLUTION, seed=SEED))
    communities_all = sorted([set(c) for c in communities_all], key=lambda c: (-len(c), sorted(c,
        key=natural_key)[0] if c else ''))
    retained = [c for c in communities_all if len(c) >= MIN_MODULE_SIZE]
    node_to_module = {}
    for index, community in enumerate(retained, start=1):
        module = f'M{index}'
        for node in community:
            node_to_module[node] = module
    assignment = ann[['node', 'label', 'moltype', 'group']].copy()
    assignment.insert(0, 'network', network)
    assignment['module'] = assignment['node'].map(node_to_module).fillna('Unassigned')
    module_sizes = {f'M{i}': len(c) for i, c in enumerate(retained, start=1)}
    assignment['module_size'] = assignment['module'].map(module_sizes)
    degree_all = dict(graph.degree())
    degree_positive = dict(positive_graph.degree())
    degree_negative = {node: sum((1 for _, _, data in graph.edges(node,
        data=True) if data['rho'] < 0)) for node in graph.nodes()}
    community_size_before_filter = {}
    for community in communities_all:
        for node in community:
            community_size_before_filter[node] = len(community)
    assignment['degree_all_signs'] = assignment['node'].map(degree_all).fillna(0).astype(int)
    assignment['degree_positive'] = assignment['node'].map(degree_positive).fillna(0).astype(int)
    assignment['degree_negative'] = assignment['node'].map(degree_negative).fillna(0).astype(int)
    assignment['positive_community_size_before_filter'] = assignment['node'].map(community_size_before_filter)

    def assignment_reason(row):
        if row['module'] != 'Unassigned':
            return 'Retained module'
        if row['degree_all_signs'] == 0:
            return 'Isolated'
        if row['degree_positive'] == 0 and row['degree_negative'] > 0:
            return 'Negative-only'
        csize = row['positive_community_size_before_filter']
        if pd.notna(csize) and csize < MIN_MODULE_SIZE:
            return 'Small positive community'
        return 'Other'
    assignment['assignment_reason'] = assignment.apply(assignment_reason, axis=1)
    rows = []
    for index, community in enumerate(retained, start=1):
        module = f'M{index}'
        members = assignment.loc[assignment['module'] == module]
        subgraph = positive_graph.subgraph(community).copy()
        internal_rho = np.array([data['rho'] for _, _, data in subgraph.edges(data=True)], dtype=float)
        rows.append({'network': network, 'module': module, 'module_size': len(community),
            'n_protein': int((members['moltype'] == 'Protein').sum()),
            'n_metabolite': int((members['moltype'] == 'Metabolite').sum()),
            'n_Baseline_only': int((members['group'] == 'Baseline-only').sum()),
            'n_Trend_only': int((members['group'] == 'Trend-only').sum()),
            'n_Dual_response': int((members['group'] == 'Dual-response').sum()),
            'positive_internal_edges': subgraph.number_of_edges(),
            'positive_internal_density': nx.density(subgraph) if len(community) > 1 else 0.0,
            'mean_internal_rho': float(internal_rho.mean()) if len(internal_rho) else np.nan})
    module_summary = pd.DataFrame(rows)
    save_tsv(assignment, f'03_{network}_module_assignment.txt')
    save_tsv(module_summary, f'03_{network}_module_summary.txt')
    save_tsv(assignment, f'04_{network}_Cytoscape_nodes_ALL.txt')
    retained_nodes = set(assignment.loc[assignment['module'] != 'Unassigned', 'node'])
    cyt_nodes_module = assignment.loc[assignment['node'].isin(retained_nodes)].copy()
    save_tsv(cyt_nodes_module, f'04_{network}_Cytoscape_nodes_RETAINED_MODULES.txt')
    save_tsv(edges, f'04_{network}_Cytoscape_edges_ALL_SIG.txt')
    cyt_edges_module = edges.loc[edges['rho'].gt(0) & edges['source'].isin(retained_nodes) & edges['target'].isin(retained_nodes)].copy()
    save_tsv(cyt_edges_module, f'04_{network}_Cytoscape_edges_RETAINED_MODULES_POSITIVE.txt')
    n_candidate = len(assignment)
    n_active = int((assignment['degree_all_signs'] > 0).sum())
    n_isolated = int((assignment['degree_all_signs'] == 0).sum())
    n_negative_only = int(((assignment['degree_positive'] == 0) & (assignment['degree_negative'] > 0)).sum())
    n_small = int((assignment['assignment_reason'] == 'Small positive community').sum())
    n_assigned = int((assignment['module'] != 'Unassigned').sum())
    summary = {'network': network, 'timepoint': STATIC_SPEC[network], 'candidate_nodes': n_candidate,
        'active_nodes': n_active, 'isolated_nodes': n_isolated, 'negative_only_nodes': n_negative_only,
        'small_positive_community_nodes': n_small, 'retained_module_nodes': n_assigned,
        'unassigned_nodes': n_candidate - n_assigned, 'significant_edges_all_signs': len(edges),
        'positive_edges': int(edges['rho'].gt(0).sum()), 'negative_edges': int(edges['rho'].lt(0).sum()),
        'n_modules': len(module_summary), 'cytoscape_module_positive_edges': len(cyt_edges_module)}
    print(f'[{network}] modules={len(module_summary)}, assigned={n_assigned}, unassigned={n_candidate - n_assigned}, active={n_active}, isolated={n_isolated}')
    if not module_summary.empty:
        print(module_summary[['module', 'module_size', 'n_protein', 'n_metabolite']].to_string(index=False))
    return (assignment, module_summary, summary)

def main():
    os.chdir(WORKDIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    required = [PRO_FILE, MET_FILE, REF_FILE, *PROTEIN_LIST_FILES.values(), *METABOLITE_LIST_FILES.values()]
    for path in required:
        require_file(path)
    print('=' * 100)
    print('W0/W8 STATIC MULTI-OMICS NETWORK — FINAL SIMPLE')
    print('W0 = A')
    print('W8 = D')
    print(f'Edge: BH-FDR < {FDR_CUTOFF}, |rho| >= {MIN_ABS_RHO}')
    print(f'Module: positive significant edges only; Louvain resolution={COMMUNITY_RESOLUTION}; min size={MIN_MODULE_SIZE}')
    print('=' * 100)
    pro_df, pro_id, pro_time = load_expression(PRO_FILE, 'Protein')
    met_df, met_id, met_time = load_expression(MET_FILE, 'Metabolite')
    ann = build_annotation(pro_df, met_df, load_reference())
    subjects = common_subjects(pro_df, pro_id, pro_time, met_df, met_id, met_time)
    print(f'\nComplete common subjects = {len(subjects)}')
    print(f'Selected molecules before static QC = {len(ann)}')
    matrices, ann_used = build_static_inputs(pro_df, pro_id, pro_time, met_df, met_id, met_time, ann, subjects)
    print(f'Final common molecules after static QC = {len(ann_used)}')
    summary_rows = []
    module_tables = []
    for network in ['W0', 'W8']:
        edges = infer_network(matrices[network], network, ann_used)
        graph = make_graph(ann_used, edges)
        assignment, module_summary, summary = detect_modules(graph, network, ann_used, edges)
        summary['n_subjects'] = len(subjects)
        summary_rows.append(summary)
        if not module_summary.empty:
            module_tables.append(module_summary)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[['network', 'timepoint', 'n_subjects', 'candidate_nodes', 'active_nodes',
        'isolated_nodes', 'negative_only_nodes', 'small_positive_community_nodes', 'retained_module_nodes',
        'unassigned_nodes', 'significant_edges_all_signs', 'positive_edges', 'negative_edges', 'n_modules',
        'cytoscape_module_positive_edges']]
    save_tsv(summary_df, '05_W0_W8_network_module_summary.txt')
    if module_tables:
        save_tsv(pd.concat(module_tables, ignore_index=True), '05_W0_W8_all_module_summary.txt')
    print('\n' + '=' * 100)
    print('FINAL NODE / EDGE COUNTS')
    print('=' * 100)
    print(summary_df.to_string(index=False))
    print('\nOutput:')
    print(os.path.abspath(OUT_DIR))
if __name__ == '__main__':
    main()
