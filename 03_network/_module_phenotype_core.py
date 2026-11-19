"""Module–phenotype association statistics based on fixed network assignments."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata, spearmanr
from statsmodels.stats.multitest import multipletests

PHENOTYPES = ('BMI', 'WC', 'FatMass', 'BFP', 'VO2', 'O2pluse', 'RHR', 'VCmax')
TIMES = np.array([0.0, 2.0, 30.0, 60.0])
RESPONSES = ('Delta2', 'Delta30', 'Delta60')
N_PERMUTATIONS = 10000
PERM_SEED = 20261008
MIN_SIZE = 10
JACCARD = 0.40

def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep='\t', low_memory=False)

def write(frame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, na_rep='NA')

def assignments(network_dir):
    result = {}
    for week in ('W0', 'W8'):
        x = read(Path(network_dir) / f'03_{week}_module_assignment.txt')
        if not {'node', 'module'}.issubset(x):
            raise ValueError(f'{week}: expected node and module columns')
        if x.node.astype(str).duplicated().any():
            raise ValueError(f'{week}: duplicate nodes')
        x = x.loc[x.module.notna() & x.module.ne('Unassigned')].copy()
        x['node'] = x.node.astype(str)
        groups = {str(mod): set(g.node) for mod, g in x.groupby('module', sort=False)}
        if not groups or any(len(members) < MIN_SIZE for members in groups.values()):
            raise ValueError(f'{week}: retained modules missing or size < {MIN_SIZE}')
        if 'louvain_resolution' in x.columns:
            if not np.allclose(pd.to_numeric(x.louvain_resolution), 1.2):
                raise ValueError(f'{week}: expected Louvain gamma = 1.2')
        result[week] = dict(sorted(groups.items(), key=lambda k: int(k[0][1:]) if k[0][1:].isdigit() else k[0]))
    return result['W0'], result['W8']

def match_modules(w0, w8, method):
    pairs = pd.DataFrame([
        dict(W0_module=a, W8_module=b, overlap_n=len(v & w),
             W0_size=len(v), W8_size=len(w),
             jaccard=len(v & w) / len(v | w))
        for a, v in w0.items() for b, w in w8.items()
    ])
    ids0, ids8 = list(w0), list(w8)
    matched = {}
    if method == 'hungarian':
        matrix = np.array([[len(w0[a] & w8[b]) / len(w0[a] | w8[b]) for b in ids8] for a in ids0])
        valid = matrix > JACCARD
        utility = np.column_stack([np.where(valid, matrix, 0), np.zeros((len(ids0), len(ids0)))])
        rows, cols = linear_sum_assignment(-utility)
        matched = {ids0[i]: ids8[j] for i, j in zip(rows, cols) if j < len(ids8) and valid[i, j]}
    elif method == 'greedy':
        used8 = set()
        for r in pairs.sort_values(['jaccard', 'overlap_n', 'W0_module', 'W8_module'],
                                  ascending=[False, False, True, True]).itertuples():
            if r.jaccard <= JACCARD:
                break
            if r.W0_module not in matched and r.W8_module not in used8:
                matched[r.W0_module] = r.W8_module
                used8.add(r.W8_module)
    else:
        raise ValueError(method)
    used8 = set(matched.values())
    ordered0 = ([a for a in ids0 if a in matched] + [a for a in ids0 if a not in matched]
                if method == 'greedy' else ids0)
    rows = []
    for i, a in enumerate(ordered0, 1):
        b = matched.get(a)
        nodes = w0[a] & w8[b] if b else w0[a]
        rows.append(dict(module=f'M{i}', kind='Matched' if b else 'W0_only',
                         W0_module=a, W8_module=b, W0_size=len(w0[a]),
                         W8_size=len(w8[b]) if b else np.nan,
                         jaccard=len(nodes)/len(w0[a] | w8[b]) if b else np.nan,
                         overlap_n=len(nodes) if b else np.nan,
                         n_scoring_nodes=len(nodes),
                         scoring_rule='intersection' if b else 'full_W0',
                         scoring_nodes=';'.join(sorted(nodes)), small_scoring_set=len(nodes)<3))
    for i, b in enumerate((b for b in ids8 if b not in used8), 1):
        nodes = w8[b]
        rows.append(dict(module=f'N{i}', kind='W8_new', W0_module=None, W8_module=b,
                         W0_size=np.nan, W8_size=len(nodes), jaccard=np.nan,
                         overlap_n=np.nan, n_scoring_nodes=len(nodes),
                         scoring_rule='full_W8', scoring_nodes=';'.join(sorted(nodes)),
                         small_scoring_set=len(nodes)<3))
    mapping = pd.DataFrame(rows)
    if method == 'hungarian':
        ordering = {'Matched': 0, 'W0_only': 1, 'W8_new': 2}
        mapping = mapping.assign(_order=mapping.kind.map(ordering)).sort_values('_order',
            kind='stable').drop(columns='_order')
    return pairs, mapping.reset_index(drop=True)

def _subject_frame(frame, labels=('subject',)):
    x = frame.copy()
    if 'subject' not in x:
        raise ValueError('Missing subject column')
    x['subject'] = x.subject.astype(str).str.strip().str.lower()
    if x.duplicated(list(labels)).any():
        raise ValueError('Duplicated subject/time rows')
    return x

def static_delta(network_dir):
    frames = []
    for week in ('W0', 'W8'):
        x = _subject_frame(read(Path(network_dir) / f'01_{week}_subject_by_molecule_static_matrix.txt'))
        x = x.set_index('subject').apply(pd.to_numeric, errors='raise')
        frames.append(x)
    a, d = frames
    if set(a.index) != set(d.index) or set(a.columns) != set(d.columns):
        raise ValueError('Resting W0/W8 subjects or molecules are inconsistent')
    d = d.reindex(index=a.index, columns=a.columns)
    delta = d - a
    if not np.isfinite(delta.to_numpy(float)).all():
        raise ValueError('Non-finite resting molecular changes')
    delta.index.name = 'subject'
    return delta

def _acute_cube(network_dir, week):
    x = _subject_frame(read(Path(network_dir) / f'01_{week}_Delta2_Delta30_Delta60_network_input.txt'),
                       labels=('subject', 'response_time'))
    if 'response_time' not in x:
        raise ValueError('Missing response_time')
    x['response_time'] = x.response_time.astype(str).str.strip()
    subjects = sorted(x.subject.unique())
    nodes = [c for c in x if c not in ('subject','response_time')]
    expected = pd.MultiIndex.from_product([subjects, RESPONSES], names=['subject','response_time'])
    actual = pd.MultiIndex.from_frame(x[['subject', 'response_time']])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f'{week}: incomplete 3-time-point trajectories')
    values = x.set_index(['subject','response_time']).reindex(expected)[nodes].apply(pd.to_numeric,
        errors='raise').to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f'{week}: invalid acute response values')
    return subjects, nodes, values.reshape(len(subjects), 3, len(nodes))

def exact_absauc(values):
    z = np.concatenate([np.zeros((len(values), 1, values.shape[-1])), values], axis=1)
    result = np.zeros((len(values), values.shape[-1]))
    for i in range(3):
        a, b = z[:, i, :], z[:, i+1, :]
        delta_t = TIMES[i+1] - TIMES[i]
        aa, bb = np.abs(a), np.abs(b)
        segment = (aa + bb) * (delta_t / 2)
        cross = a*b < 0
        if cross.any():
            segment[cross] = delta_t * (aa[cross]**2 + bb[cross]**2) / (2*(aa[cross] + bb[cross]))
        result += segment
    return result

def acute_delta(network_dir):
    s0, n0, x0 = _acute_cube(network_dir, 'W0')
    s8, n8, x8 = _acute_cube(network_dir, 'W8')
    if set(s0) != set(s8) or set(n0) != set(n8):
        raise ValueError('Acute W0/W8 subjects or molecules are inconsistent')
    x8 = x8[[s8.index(s) for s in s0]][:, :, [n8.index(n) for n in n0]]
    sd = np.std(np.concatenate([x0, x8], axis=1), axis=(0,1), ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 1e-12):
        raise ValueError('Zero/invalid pooled response SD')
    changes = exact_absauc(x8 / sd[None, None, :]) - exact_absauc(x0 / sd[None, None, :])
    return pd.DataFrame(changes, index=pd.Index(s0, name='subject'), columns=n0)

def phenotype_changes(path, fat_mass_excel, subjects):
    x = _subject_frame(read(path)).set_index('subject')
    basic = [p for p in PHENOTYPES if p != 'FatMass']
    required = [f'Delta_{p}' for p in basic]
    if not set(required).issubset(x):
        raise ValueError(f'Phenotype table missing: {sorted(set(required) - set(x))}')
    x = x[required].apply(pd.to_numeric, errors='coerce')
    s = pd.Index([str(v).strip().lower() for v in subjects], name='subject')
    if not s.isin(x.index).all():
        raise ValueError('Phenotype input lacks some network subjects')
    raw = pd.read_excel(fat_mass_excel, usecols=['ID','group','TOTAL_FAT_MASS'])
    raw['ID'] = raw.ID.astype(str).str.strip().str.lower()
    raw['group'] = raw.group.astype(str).str.strip().str.lower()
    raw = raw[raw.group.isin(['week0','week8'])].copy()
    if raw.duplicated(['ID','group']).any():
        raise ValueError('Repeated fat mass subject/week')
    raw['TOTAL_FAT_MASS'] = pd.to_numeric(raw.TOTAL_FAT_MASS, errors='raise')
    wide = raw.pivot(index='ID', columns='group', values='TOTAL_FAT_MASS')
    if not s.isin(wide.index).all() or not {'week0','week8'}.issubset(wide):
        raise ValueError('Fat mass input missing subjects or weeks')
    x = x.reindex(s)
    x['Delta_FatMass'] = (wide.loc[s, 'week8'] - wide.loc[s, 'week0']).to_numpy()
    return x[[f'Delta_{p}' for p in PHENOTYPES]]

def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method='fdr_bh')[1]
    return q

def correlate(scores, phenotypes, n_perms=N_PERMUTATIONS, seed=PERM_SEED):
    if not scores.index.equals(phenotypes.index):
        raise ValueError('Subject order mismatch')
    modules = scores.columns.tolist()
    x = scores.to_numpy(float)
    rng = np.random.default_rng(seed)
    rows = []
    for p in PHENOTYPES:
        y0 = phenotypes[f'Delta_{p}'].to_numpy(float)
        mask = np.isfinite(y0)
        n = int(mask.sum())
        if n < 10:
            rows.extend(dict(module=m, phenotype=p, n=n, rho=np.nan, pvalue=np.nan,
                             empirical_p=np.nan, perm_maxT_p=np.nan) for m in modules)
            continue
        xx, yy = x[mask], y0[mask]
        yr = rankdata(yy); yr -= yr.mean()
        xr = np.column_stack([rankdata(xx[:, i]) for i in range(len(modules))]).astype(float)
        xr -= xr.mean(axis=0)
        yn, xn = np.linalg.norm(yr), np.linalg.norm(xr, axis=0)
        ok = (xn > 1e-12) & (yn > 1e-12)
        rho = np.full(len(modules), np.nan); raw = rho.copy(); emp = rho.copy(); maxp = rho.copy()
        if ok.any():
            rho[ok] = np.clip((yr @ xr[:, ok]) / (yn * xn[ok]), -1, 1)
            for j in np.flatnonzero(ok):
                raw[j] = spearmanr(xx[:, j], yy).pvalue
            hits = np.zeros(ok.sum(), dtype=np.int64)
            max_hits = np.zeros(ok.sum(), dtype=np.int64)
            for start in range(0, n_perms, 1000):
                permutations = np.array([rng.permutation(n) for _ in range(min(1000, n_perms-start))])
                rp = np.abs((yr[permutations] @ xr[:, ok]) / (yn * xn[ok]))
                observed = np.abs(rho[ok])
                hits += (rp >= observed[None, :] - 1e-12).sum(axis=0)
                max_hits += (rp.max(axis=1)[:, None] >= observed[None, :] - 1e-12).sum(axis=0)
            if n_perms:
                emp[ok] = (hits + 1) / (n_perms + 1)
                maxp[ok] = (max_hits + 1) / (n_perms + 1)
        rows.extend(dict(module=m, phenotype=p, n=n, rho=rho[j], pvalue=raw[j],
                         empirical_p=emp[j], perm_maxT_p=maxp[j]) for j, m in enumerate(modules))
    table = pd.DataFrame(rows)
    table['raw_fdr_global'] = bh(table.pvalue)
    table['empirical_fdr_global'] = bh(table.empirical_p)
    for raw_col, q_col in (('pvalue','raw_fdr_within_phenotype'), ('empirical_p','empirical_fdr_within_phenotype')):
        table[q_col] = np.nan
        for _, ind in table.groupby('phenotype').groups.items():
            table.loc[ind, q_col] = bh(table.loc[ind, raw_col])
    return table

def run(kind, root):
    root = Path(root)
    source = root / ('resting_network' if kind == 'static'
                     else 'acute_response_network')
    output = root / ('resting_module_phenotype' if kind == 'static'
                     else 'acute_module_phenotype')
    w0, w8 = assignments(source)
    pairs, mapping = match_modules(w0, w8, 'hungarian' if kind == 'static' else 'greedy')
    delta = static_delta(source) if kind == 'static' else acute_delta(source)
    if not set(mapping.scoring_nodes.str.split(';').explode()).issubset(delta.columns):
        raise ValueError('Module nodes missing from frozen molecular matrices')
    scores = pd.DataFrame({r.module:delta[r.scoring_nodes.split(';')].mean(axis=1)
                           for r in mapping.itertuples(index=False)}, index=delta.index)
    if not np.isfinite(scores.to_numpy(float)).all():
        raise ValueError('Non-finite module scores')
    phenos = phenotype_changes(root/'Delta3TP_network_module_8phenotypes_CORRECTED_results'/'00_8phenotype_values_for_network_subjects.txt',

                               root/'Total_before_after.xlsx', scores.index)
    table = correlate(scores, phenos)
    table = table.merge(mapping.drop(columns='scoring_nodes'), on='module', validate='many_to_one')
    table.insert(0, 'resolution', 1.2)
    table.insert(1, 'min_size', MIN_SIZE)
    table.insert(2, 'weight_mode', 'weighted')
    write(pairs, output/'W0_W8_Jaccard_all_pairs.tsv')
    write(mapping, output/'module_scoring_groups.tsv')
    write(scores.reset_index(), output/'subject_module_scores.tsv')
    # Preserve the exact phenotype change values used for correlations (Figure 4F/5F).
    write(phenos.reset_index(), output/'00_8phenotype_values_used_FatMass.tsv')
    write(table, output/'module_phenotype_all_tests.tsv')
    print(f'{kind}: {len(scores)} subjects, {len(mapping)} module groups, {len(table)} tests -> {output}')
