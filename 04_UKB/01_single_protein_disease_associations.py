"""Single-protein Cox associations across UK Biobank incident disease endpoints."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
from pathlib import Path
from contextlib import nullcontext
import multiprocessing as mp
import re
import time
import traceback
import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

PROJECT_DIR = Path(os.environ.get('PROJECT_DIR', str(Path(os.environ.get("MULTIOMICS_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data"))))))
UKB_DIR = Path(os.environ.get('UKB_DIR', str(Path(os.environ.get("MULTIOMICS_UKB_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "ukb"))))))
INPUT_CASE_DIR = Path(os.environ.get('UKB_CASE_DIR', str(UKB_DIR / 'incident_disease_records')))
SUMMARY_PATH = Path(os.environ.get('UKB_CASE_SUMMARY', str(INPUT_CASE_DIR / 'disease_records_summary.csv')))
OUT_DIR = PROJECT_DIR / 'ukb_incident_cox'
INCIDENT_OUT_DIR = OUT_DIR / 'per_disease_incident_HR'
ADMIN_CENSOR_DATE = pd.Timestamp('2023-11-30')
DEATH_DATE_FILE = None
DEATH_EID_COL, DEATH_DATE_COL = 'eid', 'death_date'
N_CORES = min(8, os.cpu_count() or 1)
MIN_N, MIN_EVENTS, MIN_CONTROLS = 100, 20, 20
RAW_P_THRESHOLD, FDR_THRESHOLD, BONF_ALPHA = .05, .05, .05
ANALYSIS_VERSION = 'incident_Cox_input630_v1'
SOURCE_CASE_CONTROL_VERSION = 'USER_SUPPLIED_630'
EXPECTED_INCIDENT_TOTAL = 630
PROJECT_REF, UKB_REF = PROJECT_DIR / 'protein_annotations.tsv', UKB_DIR / 'protein_annotations.tsv'
OLINK_FILE = UKB_DIR / 'ukb_protein_matrix.csv'
AGE_FILE, SEX_FILE = UKB_DIR / 'ukb_age.csv', UKB_DIR / 'ukb_sex.csv'
PROCESSED_53_FILE = UKB_DIR / 'ukb_baseline_dates.csv'
PROTEIN_LIST_FILES = {
    'Baseline-only': PROJECT_DIR / 'protein_baseline_only.tsv',
    'Trend-only': PROJECT_DIR / 'protein_dynamic_only.tsv',
    'Dual-response': PROJECT_DIR / 'protein_dual_response.tsv',
}
G_OLINK = G_PROTEINS = G_ANNOTATION = None
G_INCIDENT_DIR = INPUT_CASE_DIR
G_INCIDENT_META = None
G_BASELINE = G_DEATH = None
BASE_REQUIRED = {'analysis_type', 'analysis_version', 'source_case_control_version',
                 'disease','coding','gene','MoleculeClass','N','beta','SE','p_value',
                 'CI95_low','CI95_high'}

def clean_text(x):
    return '' if pd.isna(x) else str(x).strip()

def normalize_gene(x):
    x = clean_text(x)
    return np.nan if not x else x.upper()

def normalize_protein_id(x):
    x = clean_text(x)
    if not x:
        return np.nan
    if x.startswith('X') and len(x) > 1:
        x = x[1:]
    return x

def safe_filename(name):
    return re.sub('[\\\\/:*?"<>|]+', '_', str(name).strip())

def clean_eid_series(series):
    return series.astype(str).str.strip().str.replace('\\.0$', '', regex=True)

def ensure_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{label} not found: {path}')

def ensure_dir(path, label):
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f'{label} not found: {path}')

def atomic_csv(df, path):
    """
    Atomic checkpoint:
      write -> temporary file -> os.replace()
    so an interrupted write cannot masquerade as a completed disease result.
    """
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def as_bool(series):
    """
    Robust conversion of CSV boolean-like values.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({'true', '1', 'yes', 'y', 't'})

def metadata_for_result(disease, analysis_type):
    meta = G_PREVALENT_META if analysis_type == 'Prevalent' else G_INCIDENT_META
    r = meta.loc[disease]

    def val(col):
        x = r[col]
        return '' if pd.isna(x) else str(x)
    return {'icd_mapping': val('icd_mapping_input'), 'icd_mapping_expanded': val('icd_mapping_expanded'),
        'PRE_CONDITIONS': val('PRE_CONDITIONS'), 'CONDITIONS': val('CONDITIONS'),
        'CONTROL_PRECONDITIONS': val('CONTROL_PRECONDITIONS'), 'CONTROL_CONDITIONS': val('CONTROL_CONDITIONS'),
        'CONTROL_EXCLUDE': val('CONTROL_EXCLUDE')}

def read_project_ref(path):
    df = pd.read_csv(path, sep='\t', header=None, dtype=str, comment='#')
    if df.shape[1] < 2:
        raise ValueError(f'{path} has fewer than 2 columns')
    df = df.iloc[:, :2].copy()
    df.columns = ['protein_id', 'gene']
    if len(df):
        a = str(df.iloc[0, 0]).strip().lower()
        b = str(df.iloc[0, 1]).strip().lower()
        if a in {'protein', 'protein_id', 'id', 'coding'} or b in {'gene', 'symbol', 'gene_symbol', 'genesymbol'}:
            df = df.iloc[1:].copy()
    df['protein_id'] = df['protein_id'].map(normalize_protein_id)
    df['gene'] = df['gene'].map(normalize_gene)
    return df.dropna(subset=['protein_id', 'gene']).drop_duplicates()

def read_ukb_ref(path):
    df = pd.read_csv(path, sep='\t', dtype=str)
    df.columns = [str(x).strip() for x in df.columns]
    if 'coding' not in df.columns or 'gene' not in df.columns:
        if df.shape[1] < 2:
            raise ValueError('Cannot resolve coding/gene from UKB protein_annotations.tsv')
        df = df.iloc[:, :2].copy()
        df.columns = ['coding', 'gene']
    df = df[['coding', 'gene']].copy()
    df['coding'] = df['coding'].map(clean_text)
    df['gene'] = df['gene'].map(normalize_gene)
    return df.dropna(subset=['coding', 'gene']).drop_duplicates()

def read_protein_list(path):
    values = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = normalize_protein_id(re.split('[\\t,\\s]+', line)[0])
            if pd.isna(value) or str(value).lower() in {'protein', 'protein_id', 'id'}:
                continue
            values.append(value)
    return list(dict.fromkeys(values))

def build_three_class_intersection():
    print('\n' + '=' * 100)
    print('STEP 1: THREE-CLASS UKB INTERSECTION')
    print('=' * 100)
    ensure_file(PROJECT_REF, 'project ref')
    ensure_file(UKB_REF, 'UKB ref')
    ensure_file(OLINK_FILE, 'Olink')
    project_ref = read_project_ref(PROJECT_REF)
    ukb_ref = read_ukb_ref(UKB_REF)
    rows = []
    for cls, path in PROTEIN_LIST_FILES.items():
        ensure_file(path, f'{cls} protein list')
        ids = read_protein_list(path)
        tmp = pd.DataFrame({'protein_id': ids, 'MoleculeClass': cls})
        tmp = tmp.merge(project_ref, on='protein_id', how='left')
        rows.append(tmp)
        print(f'{cls:16s}: {len(ids):4d}')
    three = pd.concat(rows, ignore_index=True)
    intersection = three.merge(ukb_ref, on='gene', how='inner')
    intersection = intersection[['protein_id', 'gene', 'MoleculeClass', 'coding']].drop_duplicates()
    olink_cols = set(map(str, pd.read_csv(OLINK_FILE, nrows=0).columns))
    intersection = intersection[intersection['coding'].astype(str).isin(olink_cols)].copy()
    intersection = intersection.sort_values(['MoleculeClass', 'gene', 'coding']).drop_duplicates('coding',
        keep='first').reset_index(drop=True)
    if intersection.empty:
        raise RuntimeError('No three-class proteins matched to UKB Olink')
    summary = intersection.groupby('MoleculeClass').agg(n_project_protein=('protein_id', 'nunique'),
        n_gene=('gene', 'nunique'), n_UKB_coding=('coding', 'nunique')).reset_index()
    print('\n' + summary.to_string(index=False))
    print(f"\nTotal UKB assays: {intersection['coding'].nunique():,}")
    intersection.to_csv(OUT_DIR / 'ukb_protein_class_mapping.tsv', sep='\t', index=False)
    return intersection

def build_annotation_map(intersection):
    return {str(r.coding): {'protein_id': r.protein_id, 'gene': r.gene,
        'MoleculeClass': r.MoleculeClass} for r in intersection.itertuples(index=False)}

def normalize_sex(series):
    numeric = pd.to_numeric(series, errors='coerce')
    vals = set(numeric.dropna().unique().tolist())
    if vals and vals.issubset({0, 1, 0.0, 1.0}):
        return numeric.astype(float)
    text = series.astype(str).str.strip().str.lower()
    mapping = {'female': 0.0, 'f': 0.0, 'woman': 0.0, 'women': 0.0, 'male': 1.0, 'm': 1.0, 'man': 1.0, 'men': 1.0}
    mapped = text.map(mapping)
    if mapped.notna().sum() == series.notna().sum():
        return mapped
    raise ValueError('Cannot recognize sex coding')

def load_covariates():
    ensure_file(AGE_FILE, 'ukb_age.csv')
    ensure_file(SEX_FILE, 'ukb_sex.csv')
    age = pd.read_csv(AGE_FILE, usecols=['eid', 'age'], dtype={'eid': str}, low_memory=False)
    sex = pd.read_csv(SEX_FILE, usecols=['eid', 'sex'], dtype={'eid': str}, low_memory=False)
    age['eid'] = clean_eid_series(age['eid'])
    sex['eid'] = clean_eid_series(sex['eid'])
    age['age'] = pd.to_numeric(age['age'], errors='coerce')
    sex['sex'] = normalize_sex(sex['sex'])
    age = age.drop_duplicates('eid')
    sex = sex.drop_duplicates('eid')
    cov = age.merge(sex, on='eid', how='inner').dropna(subset=['age', 'sex'])
    print(f'Complete Age + Sex: {len(cov):,}')
    return cov

def load_olink(intersection, cov):
    proteins = intersection['coding'].astype(str).drop_duplicates().tolist()
    olink = pd.read_csv(OLINK_FILE, usecols=['eid'] + proteins, dtype={'eid': str}, low_memory=False)
    olink['eid'] = clean_eid_series(olink['eid'])
    olink[proteins] = olink[proteins].apply(pd.to_numeric, errors='coerce')
    olink = olink.merge(cov, on='eid', how='inner', sort=False)
    print(f'Subjects: {len(olink):,}')
    print(f'Proteins: {len(proteins):,}')
    return (olink, proteins)

def load_baseline_dates():
    ensure_file(PROCESSED_53_FILE, 'ukb_baseline_dates.csv')
    df = pd.read_csv(PROCESSED_53_FILE, usecols=['eid', 'date'], dtype={'eid': str}, low_memory=False)
    df['eid'] = clean_eid_series(df['eid'])
    df['baseline_date'] = pd.to_datetime(df['date'], errors='coerce')
    return df[['eid', 'baseline_date']].dropna(subset=['baseline_date']).drop_duplicates('eid')

def load_death_dates():
    if DEATH_DATE_FILE is None:
        return None
    path = Path(DEATH_DATE_FILE)
    ensure_file(path, 'death date file')
    df = pd.read_csv(path, usecols=[DEATH_EID_COL, DEATH_DATE_COL], dtype={DEATH_EID_COL: str}, low_memory=False)
    df.columns = ['eid', 'death_date']
    df['eid'] = clean_eid_series(df['eid'])
    df['death_date'] = pd.to_datetime(df['death_date'], errors='coerce')
    return df.drop_duplicates('eid')

def case_control_paths(disease, disease_dir):
    base = safe_filename(disease)
    return (disease_dir / f'{base}.csv', disease_dir / f'{base}_control.csv')

def read_eid_file(path):
    df = pd.read_csv(path, usecols=['eid'], dtype={'eid': str}, low_memory=False)
    return set(clean_eid_series(df['eid']).dropna().tolist())

def read_incident_case_file(path):
    """
    INPUT630 incident case file MUST contain:
        eid,event_date
    """
    header = pd.read_csv(path, nrows=0)
    required = {'eid', 'event_date'}
    if not required.issubset(header.columns):
        raise ValueError(f'{path} is not a INPUT630 incident case file; required columns are {sorted(required)}')
    df = pd.read_csv(path, usecols=['eid', 'event_date'], dtype={'eid': str}, low_memory=False)
    df['eid'] = clean_eid_series(df['eid'])
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    df = df.dropna(subset=['eid'])
    if df['eid'].duplicated().any():
        dup_n = int(df['eid'].duplicated().sum())
        raise ValueError(f'{path}: {dup_n} duplicated incident case eids')
    if df['event_date'].isna().any():
        n = int(df['event_date'].isna().sum())
        raise ValueError(f'{path}: {n} incident cases have missing event_date')
    return df

def load_incident_case_control(disease):
    case_file, ctrl_file = case_control_paths(disease, G_INCIDENT_DIR)
    if not case_file.exists() or not ctrl_file.exists():
        return None
    cases = read_incident_case_file(case_file)
    controls = read_eid_file(ctrl_file)
    case_eids = set(cases['eid'])
    overlap = case_eids & controls
    if overlap:
        raise ValueError(f'{disease}: {len(overlap)} subjects appear in both V4 incident case and control files')
    return (cases, controls)

def add_disease_fdr(df):
    df = df.copy()
    df['FDR_disease'] = np.nan
    valid = df['p_value'].notna()
    if valid.any():
        df.loc[valid, 'FDR_disease'] = multipletests(df.loc[valid, 'p_value'].values, method='fdr_bh')[1]
    return df

def add_global_statistics(df):
    if df.empty:
        return df
    df = df.copy()
    valid = df['p_value'].notna()
    df['FDR_global'] = np.nan
    df['Bonferroni_global'] = np.nan
    if valid.any():
        p = df.loc[valid, 'p_value'].values
        df.loc[valid, 'FDR_global'] = multipletests(p, method='fdr_bh')[1]
        df.loc[valid, 'Bonferroni_global'] = multipletests(p, method='bonferroni')[1]
    df['sig_raw'] = df['p_value'] < RAW_P_THRESHOLD
    df['sig_disease_FDR'] = df['FDR_disease'] < FDR_THRESHOLD
    df['sig_global_FDR'] = df['FDR_global'] < FDR_THRESHOLD
    df['sig_Bonferroni'] = df['Bonferroni_global'] < BONF_ALPHA
    df['direction'] = np.select([df['effect'] < 1, df['effect'] > 1], ['Protective', 'Risk'], default='Neutral')
    return df

def result_path(disease, analysis_type):
    if analysis_type == 'Prevalent':
        return PREVALENT_OUT_DIR / f'{safe_filename(disease)}_OR.csv'
    return INCIDENT_OUT_DIR / f'{safe_filename(disease)}_HR.csv'

def load_valid_existing(disease, analysis_type):
    """
    A disease is considered completed only when:
      - its file exists and is readable,
      - required result columns exist,
      - analysis/source version tags match V4 exactly,
      - OR or HR has at least one valid value,
      - incident censor date matches current configuration.

    Therefore old disease-definition results cannot be reused accidentally.
    """
    path = result_path(disease, analysis_type)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    if not BASE_REQUIRED.issubset(df.columns):
        return None
    versions = df['analysis_version'].dropna().astype(str).unique().tolist()
    if not versions or any((x != ANALYSIS_VERSION for x in versions)):
        return None
    sources = df['source_case_control_version'].dropna().astype(str).unique().tolist()
    if not sources or any((x != SOURCE_CASE_CONTROL_VERSION for x in sources)):
        return None
    if analysis_type == 'Prevalent':
        if 'OR' not in df.columns or df['OR'].notna().sum() == 0:
            return None
    else:
        if 'HR' not in df.columns or df['HR'].notna().sum() == 0:
            return None
        if 'admin_censor_date' not in df.columns:
            return None
        dates = df['admin_censor_date'].dropna().astype(str).unique().tolist()
        expected = str(ADMIN_CENSOR_DATE.date())
        if not dates or any((x != expected for x in dates)):
            return None
        if 'event_date_source' not in df.columns:
            return None
        srcs = df['event_date_source'].dropna().astype(str).unique().tolist()
        if not srcs or any((x != 'Input 630 incident case CSV' for x in srcs)):
            return None
    return df

def scan_results(diseases, analysis_type):
    existing = []
    pending = []
    for disease in diseases:
        if load_valid_existing(disease, analysis_type) is not None:
            existing.append(disease)
        else:
            pending.append(disease)
    return (existing, pending)

def collect_results(diseases, analysis_type):
    frames = []
    missing = []
    for disease in diseases:
        df = load_valid_existing(disease, analysis_type)
        if df is None:
            missing.append(disease)
        else:
            frames.append(df)
    if not frames:
        return (pd.DataFrame(), missing)
    return (pd.concat(frames, ignore_index=True, sort=False), missing)

def build_survival_table(disease):
    """
    IMPORTANT:
    No ICD-date reconstruction happens here.

    Cases:
        read exact event_date from V4 incident case CSV.

    Controls:
        read exact V4 control IDs.

    This preserves V4 PRE_CONDITIONS / CONDITIONS / control QC.
    """
    cc = load_incident_case_control(disease)
    if cc is None:
        return (None, 'missing INPUT630 case/control', {})
    case_df, controls = cc
    if len(case_df) < MIN_EVENTS or len(controls) < MIN_CONTROLS:
        return (None, f'insufficient INPUT630 case/control ({len(case_df)}/{len(controls)})', {})
    cases = case_df[['eid', 'event_date']].copy()
    cases['event'] = 1
    ctrls = pd.DataFrame({'eid': list(controls), 'event': 0})
    ctrls['event_date'] = pd.NaT
    pheno = pd.concat([cases, ctrls], ignore_index=True, sort=False)
    pheno = pheno.merge(G_BASELINE, on='eid', how='inner', sort=False)
    n_source_cases = len(cases)
    n_source_controls = len(ctrls)
    invalid_case_prebaseline = (pheno['event'] == 1) & pheno['event_date'].notna() & pheno['baseline_date'].notna() & (pheno['event_date'] <= pheno['baseline_date'])
    n_invalid_prebaseline = int(invalid_case_prebaseline.sum())
    if n_invalid_prebaseline > 0:
        return (None,
            f'INPUT630 inconsistency: {n_invalid_prebaseline} incident cases have event_date <= baseline',
            {'n_invalid_prebaseline': n_invalid_prebaseline})
    pheno['censor_date'] = ADMIN_CENSOR_DATE
    if G_DEATH is not None:
        pheno = pheno.merge(G_DEATH, on='eid', how='left', sort=False)
        m = pheno['death_date'].notna() & (pheno['death_date'] < pheno['censor_date'])
        pheno.loc[m, 'censor_date'] = pheno.loc[m, 'death_date']
        impossible = (pheno['event'] == 1) & pheno['death_date'].notna() & (pheno['event_date'] > pheno['death_date'])
        if impossible.any():
            n = int(impossible.sum())
            return (None, f'{n} V4 incident cases have event_date after death_date', {'n_event_after_death': n})
    after_censor = (pheno['event'] == 1) & (pheno['event_date'] > pheno['censor_date'])
    n_case_after_censor = int(after_censor.sum())
    if n_case_after_censor > 0:
        pheno = pheno.loc[~after_censor].copy()
    pheno['end_date'] = pheno['censor_date']
    em = pheno['event'] == 1
    pheno.loc[em, 'end_date'] = pheno.loc[em, 'event_date']
    pheno['time_days'] = (pheno['end_date'] - pheno['baseline_date']).dt.days
    invalid_time = pheno['time_days'].isna() | (pheno['time_days'] <= 0)
    n_invalid_time = int(invalid_time.sum())
    if n_invalid_time > 0:
        pheno = pheno.loc[~invalid_time].copy()
    events = int(pheno['event'].sum())
    non_events = len(pheno) - events
    if events < MIN_EVENTS:
        return (None, f'events after follow-up filtering={events}', {'n_source_cases': n_source_cases,
            'n_source_controls': n_source_controls, 'n_case_after_censor': n_case_after_censor,
            'n_invalid_time': n_invalid_time})
    if non_events < MIN_CONTROLS:
        return (None, f'non-events after follow-up filtering={non_events}', {'n_source_cases': n_source_cases,
            'n_source_controls': n_source_controls, 'n_case_after_censor': n_case_after_censor,
            'n_invalid_time': n_invalid_time})
    info = {'n_source_cases': n_source_cases, 'n_source_controls': n_source_controls,
        'n_case_after_censor': n_case_after_censor, 'n_invalid_time': n_invalid_time, 'n_survival_rows': len(pheno),
        'n_survival_events': events, 'n_survival_nonevents': non_events}
    return (pheno, 'OK', info)

def fit_cox_one(merged, coding, disease):
    tmp = merged[[coding, 'time_days', 'event', 'age', 'sex']].dropna()
    n = len(tmp)
    if n < MIN_N:
        return None
    events = int(tmp['event'].sum())
    non_events = n - events
    if events < MIN_EVENTS or non_events < MIN_CONTROLS:
        return None
    protein = tmp[coding].to_numpy(dtype=np.float64, copy=False)
    sd = np.std(protein, ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return None
    z = (protein - np.mean(protein)) / sd
    age = tmp['age'].to_numpy(dtype=np.float64, copy=False)
    sex = tmp['sex'].to_numpy(dtype=np.float64, copy=False)
    arrays = [z]
    covs = []
    if np.nanstd(age) > 0:
        arrays.append(age)
        covs.append('Age')
    if np.nanstd(sex) > 0:
        arrays.append(sex)
        covs.append('Sex')
    X = np.column_stack(arrays)
    time_days = tmp['time_days'].to_numpy(dtype=np.float64, copy=False)
    event = tmp['event'].to_numpy(dtype=np.int8, copy=False)
    try:
        ctx = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
        with ctx:
            fit = PHReg(endog=time_days, exog=X, status=event, ties='efron').fit(disp=0)
        beta = float(fit.params[0])
        se = float(fit.bse[0])
        p = float(fit.pvalues[0])
    except Exception:
        return None
    if not np.isfinite([beta, se, p]).all():
        return None
    HR = float(np.exp(beta))
    ann = G_ANNOTATION[str(coding)]
    meta = metadata_for_result(disease, 'Incident')
    return {'analysis_type': 'Incident', 'effect_type': 'HR', 'analysis_version': ANALYSIS_VERSION,
        'source_case_control_version': SOURCE_CASE_CONTROL_VERSION, 'event_date_source': 'Input 630 incident case CSV',
        'disease': disease, **meta, 'coding': str(coding), 'protein_id': ann['protein_id'], 'gene': ann['gene'],
        'MoleculeClass': ann['MoleculeClass'], 'N': n, 'cases': events, 'controls': non_events, 'events': events,
        'beta': beta, 'SE': se, 'OR': np.nan, 'HR': HR, 'effect': HR, 'CI95_low': float(np.exp(beta - 1.96 * se)),
        'CI95_high': float(np.exp(beta + 1.96 * se)), 'p_value': p,
        'median_followup_years': float(np.median(time_days) / 365.25), 'admin_censor_date': str(ADMIN_CENSOR_DATE.date()),
        'protein_unit': 'per 1 SD', 'model': 'Z(Protein)' + (' + ' + ' + '.join(covs) if covs else '')}

def incident_worker(task):
    idx, total, disease = task
    t0 = time.time()
    try:
        pheno, reason, info = build_survival_table(disease)
        if pheno is None:
            return {'index': idx, 'total': total, 'disease': disease, 'status': 'SKIP', 'reason': reason,
                'tests': 0, 'seconds': time.time() - t0, **info}
        merged = G_OLINK.merge(pheno[['eid', 'time_days', 'event']], on='eid', how='inner', sort=False)
        rows = []
        for coding in G_PROTEINS:
            r = fit_cox_one(merged, coding, disease)
            if r is not None:
                rows.append(r)
        if not rows:
            return {'index': idx, 'total': total, 'disease': disease, 'status': 'FAIL',
                'reason': 'no successful Cox models', 'tests': 0, 'seconds': time.time() - t0, **info}
        df = add_disease_fdr(pd.DataFrame(rows)).sort_values('p_value').reset_index(drop=True)
        for k, v in info.items():
            df[k] = v
        atomic_csv(df, result_path(disease, 'Incident'))
        return {'index': idx, 'total': total, 'disease': disease, 'status': 'OK', 'reason': '', 'tests': len(df),
            'events': int(df['events'].max()), 'N': int(df['N'].max()), 'seconds': time.time() - t0, **info}
    except Exception as e:
        return {'index': idx, 'total': total, 'disease': disease, 'status': 'ERROR', 'reason': str(e), 'tests': 0,
            'seconds': time.time() - t0, 'traceback': traceback.format_exc()}

def ensure_common_columns(df, analysis_type):
    df = df.copy()
    if 'effect' not in df.columns:
        df['effect'] = df['OR'] if analysis_type == 'Prevalent' else df['HR']
    if 'FDR_disease' not in df.columns:
        parts = []
        for _, g in df.groupby('disease', sort=False):
            parts.append(add_disease_fdr(g))
        df = pd.concat(parts, ignore_index=True) if parts else df
    return df

def load_input_panel():
    path = SUMMARY_PATH
    if not path.exists():
        matches = list(INPUT_CASE_DIR.glob('*summary*.csv'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected 630-endpoint summary at {path}; found {len(matches)} alternative summaries')
        path = matches[0]
    panel = pd.read_csv(path, low_memory=False)
    needed = {'disease', 'output_saved', 'icd_mapping_input', 'icd_mapping_expanded',
              'PRE_CONDITIONS','CONDITIONS','CONTROL_PRECONDITIONS',
              'CONTROL_CONDITIONS','CONTROL_EXCLUDE'}
    missing = needed - set(panel)
    if missing:
        raise ValueError(f'Summary missing columns: {sorted(missing)}')
    panel['disease'] = panel.disease.astype(str).str.strip()
    if len(panel) != EXPECTED_INCIDENT_TOTAL or panel.disease.duplicated().any():
        raise ValueError('Input summary must contain 630 unique incident endpoints; no filtering is performed')
    if not as_bool(panel.output_saved).all():
        raise ValueError('Every supplied endpoint must have saved case/control files')
    return panel.set_index('disease', drop=False)

def main():
    global G_INCIDENT_META, G_OLINK, G_PROTEINS, G_ANNOTATION, G_BASELINE, G_DEATH
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    INCIDENT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    G_INCIDENT_META = load_input_panel()
    diseases = G_INCIDENT_META.index.tolist()
    intersection = build_three_class_intersection()
    G_ANNOTATION = build_annotation_map(intersection)
    G_OLINK, G_PROTEINS = load_olink(intersection, load_covariates())
    G_BASELINE = load_baseline_dates()
    G_DEATH = load_death_dates()
    completed, pending = scan_results(diseases, 'Incident')
    print(f'630 input diseases; {len(completed)} cached; {len(pending)} pending')
    tasks = [(i, len(pending), d) for i, d in enumerate(pending, 1)]
    log = []
    if tasks:
        if N_CORES > 1:
            with mp.get_context('fork').Pool(processes=N_CORES, maxtasksperchild=20) as pool:
                for result in pool.imap_unordered(incident_worker, tasks, chunksize=1):
                    log.append(result)
                    if result['status'] != 'OK':
                        print(f"[{result['status']}] {result['disease']}: {result['reason']}")
        else:
            log = [incident_worker(t) for t in tasks]
        pd.DataFrame(log).to_csv(OUT_DIR / '00_endpoint_runlog.csv', index=False)
    result, missing = collect_results(diseases, 'Incident')
    if result.empty:
        raise RuntimeError('No successful incident Cox models')
    result = add_global_statistics(ensure_common_columns(result, 'Incident'))
    result = result.sort_values(['disease', 'p_value']).reset_index(drop=True)
    result.to_csv(OUT_DIR / 'incident_protein_cox_results.tsv', sep='\t', index=False)
    result[result.sig_global_FDR].to_csv(OUT_DIR / '09_incident_globalFDR_lt_0.05_630.tsv', sep='\t', index=False)
    pd.DataFrame({'disease':missing}).to_csv(OUT_DIR / '10_unfitted_endpoints.csv', index=False)
    print(f'Completed: {result.disease.nunique()}/{len(diseases)} endpoints; {len(result)} tests; {len(missing)} unfitted')

if __name__ == '__main__':
    main()
