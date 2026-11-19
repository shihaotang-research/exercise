# -*- coding: utf-8 -*-
"""Week 0/week 8 acute exercise-response Spearman networks and Louvain modules.
Significant edges require BH-FDR < 0.05 and |rho| >= 0.50."""
import os
from pathlib import Path
import re
import math
import warnings
import numpy as np
import pandas as pd
import networkx as nx
warnings.filterwarnings('ignore')
WORKDIR = str(Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))))
OUT_DIR = 'acute_response_network'
PRO_FILE = 'protein_matrix.tsv'
MET_FILE = 'metabolite_matrix.tsv'
REF_FILE = 'protein_annotations.tsv'
PROTEIN_LIST_FILES = {'Baseline-only': 'protein_baseline_only.tsv', 'Trend-only': 'protein_dynamic_only.tsv',
    'Dual-response': 'protein_dual_response.tsv'}
METABOLITE_LIST_FILES = {'Baseline-only': 'metabolite_baseline_only.tsv',
    'Trend-only': 'metabolite_dynamic_only.tsv', 'Dual-response': 'metabolite_dual_response.tsv'}
NETWORK_SPEC = {'W0': {'base': 'A', 'responses': [('Delta2', 'B1'), ('Delta30', 'B2'), ('Delta60', 'B3')]},
    'W8': {'base': 'D', 'responses': [('Delta2', 'F1'), ('Delta30', 'F2'), ('Delta60', 'F3')]}}
ALL_TIMEPOINTS = ['A', 'B1', 'B2', 'B3', 'D', 'F1', 'F2', 'F3']
MIN_SUBJECTS = 10
MIN_FEATURE_SD = 1e-12
FDR_CUTOFF = 0.05
MIN_ABS_RHO = 0.5
N_BLOCK_PERMUTATIONS = 1000
PERMUTATION_EDGE_BATCH_SIZE = 250
PERMUTATION_CHUNK_SIZE = 20
PERMUTATION_PROGRESS_EVERY_BATCH = 10
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
    for x in candidates:
        if str(x).strip().lower() in lookup:
            return lookup[str(x).strip().lower()]
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
    ref = ref.iloc[:, :2]
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
                'source_list': path, 'present': node in pro_df.columns})
    for group, path in METABOLITE_LIST_FILES.items():
        for node in read_list_file(path):
            rows.append({'node': node, 'label': id_map.get(node, node), 'moltype': 'Metabolite', 'group': group,
                'source_list': path, 'present': node in met_df.columns})
    ann = pd.DataFrame(rows)
    if ann.empty:
        raise RuntimeError('No molecules found in class-list files.')
    dup = ann[ann.duplicated('node', keep=False)].copy()
    if not dup.empty:
        conflict = dup.groupby('node').agg(n_type=('moltype', 'nunique'), n_group=('group', 'nunique'))
        conflict = conflict[(conflict['n_type'] > 1) | (conflict['n_group'] > 1)]
        if not conflict.empty:
            save_tsv(dup, '00_duplicate_selected_nodes.txt')
            raise RuntimeError('Conflicting duplicated nodes; see 00_duplicate_selected_nodes.txt')
        ann = ann.drop_duplicates('node', keep='first')
    missing = ann.loc[~ann['present']].copy()
    if not missing.empty:
        save_tsv(missing, '00_selected_molecules_missing_from_matrices.txt')
    ann = ann.loc[ann['present']].copy().reset_index(drop=True)
    pnodes = set(ann.loc[ann['moltype'] == 'Protein', 'node'])
    mnodes = set(ann.loc[ann['moltype'] == 'Metabolite', 'node'])
    overlap = pnodes & mnodes
    if overlap:
        raise RuntimeError('Protein/metabolite node-name overlap: ' + ', '.join(sorted(overlap)[:20]))
    ann = ann.sort_values(['moltype', 'group', 'node'], key=lambda s: s.map(str)).reset_index(drop=True)
    save_tsv(ann, '00_selected_molecules_matched.txt')
    return ann

def subjects_with_times(df, id_col, time_col):
    time_sets = df.groupby(id_col)[time_col].apply(lambda x: set(x.astype(str)))
    need = set(ALL_TIMEPOINTS)
    return set(time_sets.index[time_sets.map(lambda x: need.issubset(x))].astype(str))

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
    for f in features:
        x[f] = pd.to_numeric(x[f], errors='coerce')
    x[id_col] = x[id_col].astype(str)
    x = x.groupby(id_col)[list(features)].mean().reindex(subjects)
    x.index.name = 'subject'
    missing_rows = x.index[x.isna().all(axis=1)].tolist()
    if missing_rows:
        raise RuntimeError(f'{tp}: missing rows for {missing_rows[:20]}')
    return x

def build_delta_inputs(pro_df, pro_id, pro_time, met_df, met_id, met_time, ann, subjects):
    pro_nodes = ann.loc[ann['moltype'] == 'Protein', 'node'].tolist()
    met_nodes = ann.loc[ann['moltype'] == 'Metabolite', 'node'].tolist()
    node_order = ann['node'].tolist()

    def combined_time(tp):
        parts = []
        if pro_nodes:
            parts.append(extract_time(pro_df, pro_id, pro_time, tp, subjects, pro_nodes))
        if met_nodes:
            parts.append(extract_time(met_df, met_id, met_time, tp, subjects, met_nodes))
        return pd.concat(parts, axis=1).reindex(index=subjects, columns=node_order)
    raw = {tp: combined_time(tp) for tp in ALL_TIMEPOINTS}
    cubes_all = {}
    labels = {}
    for network, spec in NETWORK_SPEC.items():
        base = raw[spec['base']]
        blocks = []
        labels[network] = []
        for delta_name, tp in spec['responses']:
            delta = raw[tp] - base
            blocks.append(delta.loc[subjects, node_order].to_numpy(float))
            labels[network].append(delta_name)
        cubes_all[network] = np.stack(blocks, axis=1)
    qc = []
    for j, node in enumerate(node_order):
        row = {'node': node}
        ok_both = True
        for network in ['W0', 'W8']:
            v = cubes_all[network][:, :, j].reshape(-1)
            finite = np.isfinite(v)
            sd = np.std(v[finite], ddof=1) if finite.sum() > 1 else np.nan
            ok = bool(finite.all() and np.isfinite(sd) and (sd > MIN_FEATURE_SD))
            row[f'n_finite_{network}'] = int(finite.sum())
            row[f'sd_{network}_Delta3TP'] = sd
            row[f'usable_{network}'] = ok
            ok_both &= ok
        row['usable_in_both'] = ok_both
        qc.append(row)
    qc = pd.DataFrame(qc).merge(ann[['node', 'label', 'moltype', 'group']], on='node', how='left')
    save_tsv(qc, '00_feature_QC_Delta3TP_same_set_W0_W8.txt')
    usable = qc.loc[qc['usable_in_both'], 'node'].tolist()
    if len(usable) < 2:
        raise RuntimeError('Fewer than two usable molecules.')
    ann_used = ann.set_index('node').loc[usable].reset_index()
    save_tsv(ann_used, '00_selected_molecules_final_same_set.txt')
    pos = {node: i for i, node in enumerate(node_order)}
    idx = [pos[node] for node in usable]
    matrices, cubes = ({}, {})
    for network in ['W0', 'W8']:
        cube = cubes_all[network][:, :, idx]
        cubes[network] = cube
        flat = cube.reshape(len(subjects) * 3, len(usable))
        mi = pd.MultiIndex.from_tuples([(str(s), d) for s in subjects for d in labels[network]], names=['subject',
            'response_time'])
        matrix = pd.DataFrame(flat, index=mi, columns=usable)
        matrices[network] = matrix
        matrix.reset_index().to_csv(outpath(f'01_{network}_Delta2_Delta30_Delta60_network_input.txt'), sep='\t',
            index=False)
    return (matrices, cubes, ann_used)

def bh_fdr_total(candidate_p, total_tests):
    """BH correction where untested pairs are effectively p=1."""
    p = np.asarray(candidate_p, float)
    if len(p) == 0:
        return np.array([], float)
    order = np.argsort(p, kind='mergesort')
    ranked = p[order]
    rank = np.arange(1, len(p) + 1, dtype=float)
    q = ranked * float(total_tests) / rank
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def block_permutation(rank_z_cube, src, dst, observed_rho, network):
    """


    Permute subjects, not individual Delta timepoints.


    The 3 response points from one subject always move together.


    """
    n_subjects, n_times, _ = rank_z_cube.shape
    if n_times != 3:
        raise ValueError('Delta3TP network must have exactly 3 response timepoints.')
    n_edges = len(src)
    if n_edges == 0:
        return (np.array([], float), np.array([], int))
    rng = np.random.default_rng(SEED + (0 if network == 'W0' else 100003))
    perms = np.vstack([rng.permutation(n_subjects) for _ in range(N_BLOCK_PERMUTATIONS)]).astype(np.int32)
    exceed = np.zeros(n_edges, dtype=np.int64)
    abs_obs = np.abs(observed_rho)
    subject_rows = np.arange(n_subjects, dtype=np.int32)
    n_batches = math.ceil(n_edges / PERMUTATION_EDGE_BATCH_SIZE)
    for batch_no, start in enumerate(range(0, n_edges, PERMUTATION_EDGE_BATCH_SIZE), 1):
        stop = min(start + PERMUTATION_EDGE_BATCH_SIZE, n_edges)
        s = src[start:stop]
        d = dst[start:stop]
        x = rank_z_cube[:, :, s].transpose(2, 0, 1)
        y = rank_z_cube[:, :, d].transpose(2, 0, 1)
        cross = np.einsum('est,eut->esu', x, y, optimize=True)
        local = np.zeros(stop - start, dtype=np.int64)
        obs = abs_obs[start:stop]
        for p0 in range(0, N_BLOCK_PERMUTATIONS, PERMUTATION_CHUNK_SIZE):
            p1 = min(p0 + PERMUTATION_CHUNK_SIZE, N_BLOCK_PERMUTATIONS)
            pc = perms[p0:p1]
            selected = cross[:, subject_rows[None, :], pc]
            perm_rho = selected.sum(axis=2)
            local += np.sum(np.abs(perm_rho) >= obs[:, None] - 1e-15, axis=1)
        exceed[start:stop] = local
        if batch_no == 1 or batch_no == n_batches or batch_no % PERMUTATION_PROGRESS_EVERY_BATCH == 0:
            print(f'[{network}] permutation batch {batch_no}/{n_batches}')
    p = (exceed + 1.0) / (N_BLOCK_PERMUTATIONS + 1.0)
    return (p, exceed)

def infer_network(matrix, cube, network, ann):
    n_subjects, n_times, n_features = cube.shape
    if n_times != 3 or matrix.shape != (n_subjects * 3, n_features):
        raise RuntimeError(f'{network}: invalid Delta3TP matrix dimensions.')
    if matrix.isna().any().any():
        raise RuntimeError(f'{network}: missing values in network input.')
    values = matrix.to_numpy(float)
    ranks = pd.DataFrame(values).rank(axis=0, method='average').to_numpy(float)
    ranks -= ranks.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(ranks ** 2, axis=0, keepdims=True))
    if np.any(norms <= MIN_FEATURE_SD):
        raise RuntimeError(f'{network}: zero-variance rank vector.')
    rank_z = ranks / norms
    rho_mat = np.clip(rank_z.T @ rank_z, -1, 1)
    np.fill_diagonal(rho_mat, 1.0)
    nodes = matrix.columns.to_numpy(str)
    pd.DataFrame(rho_mat, index=nodes,
        columns=nodes).to_csv(outpath(f'02_{network}_Delta3TP_Spearman_rho_matrix.txt'), sep='\t', index_label='node')
    iu, ju = np.triu_indices(n_features, k=1)
    rho = rho_mat[iu, ju]
    total_pairs = len(rho)
    mask = np.abs(rho) >= MIN_ABS_RHO
    src_idx = iu[mask]
    dst_idx = ju[mask]
    candidate_rho = rho[mask]
    print(f'\n[{network}] subjects={n_subjects}, rows={n_subjects * 3}, molecules={n_features}, all_pairs={total_pairs:,}, |rho|>={MIN_ABS_RHO} candidates={len(candidate_rho):,}')
    rank_z_cube = rank_z.reshape(n_subjects, 3, n_features)
    p, exceed = block_permutation(rank_z_cube, src_idx, dst_idx, candidate_rho, network)
    fdr = bh_fdr_total(p, total_pairs)
    edges = pd.DataFrame({'source': nodes[src_idx], 'target': nodes[dst_idx], 'rho': candidate_rho,
        'abs_rho': np.abs(candidate_rho), 'pvalue': p, 'fdr': fdr, 'permutation_exceedances': exceed,
        'n_permutations': N_BLOCK_PERMUTATIONS})
    edges['sign'] = np.where(edges['rho'] > 0, 'Positive', 'Negative')
    amap = ann.set_index('node')
    for side in ['source', 'target']:
        edges[f'{side}_label'] = edges[side].map(amap['label'])
        edges[f'{side}_moltype'] = edges[side].map(amap['moltype'])
        edges[f'{side}_group'] = edges[side].map(amap['group'])
    edges['edge_type'] = ['-'.join(sorted([a, b])) for a, b in zip(edges['source_moltype'], edges['target_moltype'])]
    edges = edges.sort_values(['fdr', 'abs_rho'], ascending=[True, False]).reset_index(drop=True)
    save_tsv(edges, f'02_{network}_permutation_tested_candidate_edges.txt')
    sig = edges.loc[edges['fdr'].lt(FDR_CUTOFF) & edges['abs_rho'].ge(MIN_ABS_RHO)].copy().reset_index(drop=True)
    save_tsv(sig, f'02_{network}_significant_edges_all_signs.txt')
    summary = pd.DataFrame([{'network': network, 'n_subjects': n_subjects, 'n_response_timepoints': 3,
        'n_network_rows': n_subjects * 3, 'n_molecules': n_features, 'total_pairwise_tests_for_BH': total_pairs,
        'rho_cutoff': MIN_ABS_RHO, 'candidate_pairs_permuted': len(edges), 'n_block_permutations': N_BLOCK_PERMUTATIONS,
        'minimum_empirical_p': 1 / (N_BLOCK_PERMUTATIONS + 1), 'significant_edges': len(sig),
        'positive_edges': int(sig['rho'].gt(0).sum()), 'negative_edges': int(sig['rho'].lt(0).sum())}])
    save_tsv(summary, f'02_{network}_network_summary.txt')
    print(f"[{network}] significant={len(sig):,}; positive={int(sig['rho'].gt(0).sum()):,}; negative={int(sig['rho'].lt(0).sum()):,}")
    return sig

def make_graph(ann, sig_edges):
    g = nx.Graph()
    for _, r in ann.iterrows():
        g.add_node(r['node'], label=r['label'], moltype=r['moltype'], group=r['group'])
    for _, r in sig_edges.iterrows():
        g.add_edge(r['source'], r['target'], rho=float(r['rho']), abs_rho=float(r['abs_rho']),
            pvalue=float(r['pvalue']), fdr=float(r['fdr']), sign=r['sign'])
    return g

def louvain_partition_at_resolution(active_graph, resolution, network):
    """Original weighted positive-edge Louvain, seed fixed for reproducibility."""
    if active_graph.number_of_edges() == 0:
        return []
    try:
        partition = list(nx.community.louvain_communities(active_graph, weight='abs_rho',
            resolution=float(resolution), seed=SEED))
    except AttributeError as exc:
        raise RuntimeError('NetworkX with louvain_communities is required; no silent fallback to a different community algorithm.') from exc
    return sorted([set(nodes) for nodes in partition], key=lambda c: (-len(c), sorted(c,
        key=natural_key)[0] if c else ''))

def detect_modules(g, network, ann):
    pg = nx.Graph()
    pg.add_nodes_from(g.nodes(data=True))
    pg.add_edges_from([(u, v, d) for u, v, d in g.edges(data=True) if d['rho'] > 0])
    active = [n for n, degree in pg.degree() if degree > 0]
    ag = pg.subgraph(active).copy()
    chosen_resolution = float(COMMUNITY_RESOLUTION)
    communities = louvain_partition_at_resolution(ag, chosen_resolution, network)
    retained = [c for c in communities if len(c) >= MIN_MODULE_SIZE]
    if network == 'W0':
        save_tsv(pd.DataFrame([{'resolution': chosen_resolution, 'n_retained_modules': len(retained),
            'n_all_communities': len(communities), 'n_assigned_nodes': sum((len(c) for c in retained)),
            'n_small_communities': len(communities) - len(retained),
            'community_sizes': ';'.join((str(len(c)) for c in communities)), 'exact_target': pd.NA,
            'mode': 'fixed_resolution'}]), '03_W0_resolution_scan.tsv')
    node_to_module = {}
    for i, community in enumerate(retained, 1):
        for node in community:
            node_to_module[node] = f'M{i}'
    assignment = ann[['node', 'label', 'moltype', 'group']].copy()
    assignment.insert(0, 'network', network)
    assignment['module'] = assignment['node'].map(node_to_module).fillna('Unassigned')
    sizes = {f'M{i}': len(c) for i, c in enumerate(retained, 1)}
    assignment['module_size'] = assignment['module'].map(sizes)
    rows = []
    for i, community in enumerate(retained, 1):
        module = f'M{i}'
        members = assignment.loc[assignment['module'] == module]
        sub = pg.subgraph(community).copy()
        rhos = np.array([d['rho'] for _, _, d in sub.edges(data=True)], float)
        rows.append({'network': network, 'louvain_resolution': chosen_resolution, 'module': module,
            'module_size': len(community), 'n_protein': int((members['moltype'] == 'Protein').sum()),
            'n_metabolite': int((members['moltype'] == 'Metabolite').sum()),
            'n_Baseline_only': int((members['group'] == 'Baseline-only').sum()),
            'n_Trend_only': int((members['group'] == 'Trend-only').sum()),
            'n_Dual_response': int((members['group'] == 'Dual-response').sum()),
            'positive_internal_edges': sub.number_of_edges(),
            'positive_internal_density': nx.density(sub) if len(community) > 1 else 0.0,
            'mean_internal_rho': float(rhos.mean()) if len(rhos) else np.nan})
    module_summary = pd.DataFrame(rows)
    save_tsv(assignment, f'03_{network}_module_assignment.txt')
    save_tsv(module_summary, f'03_{network}_module_summary.txt')
    cyt_nodes = assignment.copy()
    cyt_nodes['degree_all_signs'] = cyt_nodes['node'].map(dict(g.degree())).fillna(0).astype(int)
    cyt_nodes['degree_positive'] = cyt_nodes['node'].map(dict(pg.degree())).fillna(0).astype(int)
    save_tsv(cyt_nodes, f'04_{network}_Cytoscape_nodes.txt')
    print(f"[{network}] resolution={chosen_resolution:.3f}, modules={len(retained)}, assigned={int((assignment['module'] != 'Unassigned').sum())}, unassigned={int((assignment['module'] == 'Unassigned').sum())}")
    if not module_summary.empty:
        print(module_summary[['module', 'module_size', 'n_protein', 'n_metabolite']].to_string(index=False))
    return (assignment, module_summary)

def main():
    os.chdir(WORKDIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    required = [PRO_FILE, MET_FILE, REF_FILE, *PROTEIN_LIST_FILES.values(), *METABOLITE_LIST_FILES.values()]
    for f in required:
        require_file(f)
    print('=' * 90)
    print('W0/W8 Delta3TP response networks')
    print('W0: B1-A, B2-A, B3-A')
    print('W8: F1-D, F2-D, F3-D')
    print(f'Edge: block-permutation BH-FDR<{FDR_CUTOFF}, |rho|>={MIN_ABS_RHO}')
    print(f'Modules: positive edges, weighted Louvain, gamma={COMMUNITY_RESOLUTION}, min size={MIN_MODULE_SIZE}; number unrestricted')
    print('=' * 90)
    pro_df, pro_id, pro_time = load_expression(PRO_FILE, 'Protein')
    met_df, met_id, met_time = load_expression(MET_FILE, 'Metabolite')
    ann = build_annotation(pro_df, met_df, load_reference())
    subjects = common_subjects(pro_df, pro_id, pro_time, met_df, met_id, met_time)
    print(f'\nComplete common subjects = {len(subjects)}')
    print(f'Selected molecules before Delta QC = {len(ann)}')
    matrices, cubes, ann_used = build_delta_inputs(pro_df, pro_id, pro_time, met_df, met_id, met_time, ann, subjects)
    print(f'Final common molecules after Delta QC = {len(ann_used)}')
    network_rows = []
    module_tables = []
    for network in ['W0', 'W8']:
        sig = infer_network(matrices[network], cubes[network], network, ann_used)
        g = make_graph(ann_used, sig)
        assignment, module_summary = detect_modules(g, network, ann_used)
        save_tsv(sig, f'04_{network}_Cytoscape_edges.txt')
        network_rows.append({'network': network, 'n_subjects': len(subjects), 'n_molecules': len(ann_used),
            'n_significant_edges': len(sig), 'n_positive_edges': int(sig['rho'].gt(0).sum()),
            'n_negative_edges': int(sig['rho'].lt(0).sum()), 'n_modules': len(module_summary),
            'n_assigned_nodes': int((assignment['module'] != 'Unassigned').sum()),
            'n_unassigned_nodes': int((assignment['module'] == 'Unassigned').sum())})
        if not module_summary.empty:
            module_tables.append(module_summary)
    save_tsv(pd.DataFrame(network_rows), '05_W0_W8_network_module_summary.txt')
    combined_modules = pd.concat(module_tables, ignore_index=True) if module_tables else pd.DataFrame()
    save_tsv(combined_modules, '05_W0_W8_all_module_summary.txt')
    print('\n' + '=' * 90)
    print('FINISHED')
    print(pd.DataFrame(network_rows).to_string(index=False))
    print('Output:', os.path.abspath(OUT_DIR))
if __name__ == '__main__':
    main()
