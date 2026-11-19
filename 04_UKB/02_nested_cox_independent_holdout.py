"""UK Biobank: 630 incident endpoints, fixed 80/20 split, nested 5-by-5 CV.
LASSO-Cox models; independent within-cohort holdout 10-year IPCW AUC.
The frozen split was used previously; this is not external validation."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from threadpoolctl import threadpool_limits
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
PROJECT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data")))
UKB = Path(os.environ.get("MULTIOMICS_UKB_DIR", str(Path(__file__).resolve().parents[1] / "data" / "ukb")))
ROOT = PROJECT / 'ukb_prediction_analysis'
MAPPING = PROJECT / 'ukb_incident_cox' / 'ukb_protein_class_mapping.tsv'
AGE_FILE = UKB / 'ukb_age.csv'
SEX_FILE = UKB / 'ukb_sex.csv'
OLINK = UKB / 'ukb_protein_matrix.csv'
BASELINE_DATES = UKB / 'ukb_baseline_dates.csv'
CASE_DIR = UKB / 'incident_disease_records'
ANNOTATION_FILE = PROJECT / 'disease_definitions.tsv'
POLICY_CANDIDATES = [ROOT / 'sex_specific_endpoint_policy.tsv']
ADMIN_CENSOR = pd.Timestamp('2023-11-30')
DEATH_DATES_FILE = None
RUN_MODE = 'ALL_630'
N_JOBS = 8
TEST_FRACTION = 0.2
SPLIT_SEED = 20261110
MODEL_SEED = 20261105
AGE_BAND_YEARS = 10
OUTER_FOLDS = 5
INNER_FOLDS = 5
N_ALPHAS = 40
ALPHA_MIN_RATIO = 0.02
LAMBDA_RULES = ('min', '1se')
MIN_COVERAGE = 0.8
MIN_DEVELOPMENT_EVENTS = 60
MIN_DEVELOPMENT_NON_EVENTS = 100
MIN_TEST_EVENTS_10Y = 30
MIN_TEST_CONTROLS_AFTER_10Y = 30
MIN_EVENTS_PER_INNER_VALID = 5
MAX_ITER = 30000
TOL = 1e-06
SELECTED_THRESHOLD = 1e-07
CHUNK_SIZE = 20000
SAVE_DEV_OOF = False
SAVE_TEST_PREDICTIONS = True
PLOT_PDF = False
CACHE = True
MATCH_SCOPE = 'OFF'
MATCH_REPEATS = 20
MATCH_TARGETS = (74,)
BOOTSTRAP_SCOPE = 'OFF'
BOOTSTRAP_B = 300
CLASSES = ('Baseline-only', 'Dual-response', 'Dynamic-only')
MODELS = ('Age+Sex',) + CLASSES
COLORS = {'Age+Sex': '#8B9399', 'Baseline-only': '#4C95BC', 'Dual-response': '#55A96E', 'Dynamic-only': '#E57C5E'}
HORIZON_DAYS = 10.0 * 365.25
MAX_FOLLOWUP_DAYS = int(np.ceil(HORIZON_DAYS)) + 2
VERSION = '46_80_20_frozenV45_split_nested5x5_lasso_cox_20261111_v1'
SPLIT_VERSION = '80_20_agesex_stratified_dev5fold_lasso_cox_no_focus_20261110_v1'
RUN_ROOT = ROOT / 'independent_holdout_models'
OUT = RUN_ROOT / RUN_MODE.lower()
SPLIT_SOURCE_DIR = ROOT / 'frozen_model_split' / 'shared_split'
SPLIT_FILE = SPLIT_SOURCE_DIR / 'frozen_participant_split.tsv'
SPLIT_METADATA = SPLIT_SOURCE_DIR / 'frozen_split_metadata.json'
SPLIT_DIR = RUN_ROOT / 'split_audit'

def require(p):
    p = Path(p)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p

def eid_clean(s):
    return s.astype('string').str.strip().str.replace('\\.0$', '', regex=True)

def safe_name(x):
    return re.sub('[\\\\/:*?"<>|]+', '_', str(x).strip())

def atomic_tsv(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    tmp = path.with_suffix(path.suffix + '.tmp')
    data.to_csv(tmp, sep='\t', index=False)
    tmp.replace(path)

def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()

def manifest_signature(paths, settings):
    items = []
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        items.append([str(p), p.stat().st_size, p.stat().st_mtime_ns] if p.is_file() else [str(p), 'MISSING'])
    return sha256_bytes(json.dumps({'paths': items, 'settings': settings}, sort_keys=True, default=str).encode())[:24]

def normalize_sex(s):
    nums = pd.to_numeric(s, errors='coerce')
    words = s.astype('string').str.strip().str.lower()
    val = words.map({'female': 0.0, 'f': 0.0, 'woman': 0.0, 'women': 0.0, 'male': 1.0, 'm': 1.0, 'man': 1.0,
        'men': 1.0})
    result = nums.where(nums.isin([0.0, 1.0]), val)
    bad = s.notna() & words.ne('') & result.isna()
    if bad.any():
        raise ValueError('Unrecognized sex coding: ' + str(words[bad].unique().tolist()[:8]))
    return pd.to_numeric(result, errors='coerce')

def read_panel():
    p = pd.read_csv(require(MAPPING), sep='\t', dtype=str)
    expected = {'gene', 'coding', 'MoleculeClass'}
    if not expected.issubset(p):
        raise ValueError(f'Protein mapping missing columns: {expected - set(p)}')
    p = p[['gene', 'coding', 'MoleculeClass']].dropna().copy()
    for c in p:
        p[c] = p[c].str.strip()
    p['gene'] = p.gene.str.upper()
    p['MoleculeClass'] = p.MoleculeClass.replace({'Trend-only': 'Dynamic-only'})
    p = p[p.MoleculeClass.isin(CLASSES)].drop_duplicates().copy()
    if p.gene.duplicated().any() or p.coding.duplicated().any():
        raise ValueError('Protein mapping duplicates across classes')
    counts = p.groupby('MoleculeClass').gene.nunique().to_dict()
    if counts != {'Baseline-only': 74, 'Dual-response': 75, 'Dynamic-only': 145}:
        raise ValueError(f'Expected 74/75/145 protein assays, got {counts}')
    print('[Proteins]', counts, flush=True)
    return p.sort_values(['MoleculeClass', 'gene']).reset_index(drop=True)

def read_policy_and_registry():
    path = next((p for p in POLICY_CANDIDATES if p.is_file()), None)
    if path is None:
        raise FileNotFoundError('Sex-specific policy file not found')
    p = pd.read_csv(path, sep='\t', dtype={'disease': str})
    required = {'disease', 'risk_sex'}
    if not required.issubset(p):
        raise ValueError('Policy requires disease,risk_sex')
    p['disease'] = p.disease.str.strip()
    p['risk_sex'] = p.risk_sex.str.strip().str.lower()
    if len(p) != 630 or p.disease.duplicated().any() or (not p.risk_sex.isin(['all', 'female', 'male']).all()):
        raise ValueError('Sex policy is not exactly 630 unique valid endpoints')
    a = pd.read_csv(require(ANNOTATION_FILE), sep='\t', dtype=str)
    if not {'Disease code', 'Root'}.issubset(a):
        raise ValueError('disease_definitions.tsv missing Disease code or Root')
    a = a.rename(columns={'Disease code': 'disease'})
    if a.disease.duplicated().any():
        raise ValueError('Annotation disease duplicates')
    if 'Disease' not in a.columns:
        a['Disease'] = a.disease
    reg = p[['disease', 'risk_sex']].merge(a[['disease', 'Disease', 'Root']], on='disease', how='left',
        validate='one_to_one')
    if reg.Root.isna().any() or reg.Root.str.match('^Chapter I(?:\\s|$)').any():
        raise ValueError('Infectious/unannotated endpoint in 630 endpoint panel')
    if int(reg.Root.str.match('^Chapter IX(?:\\s|$)').sum()) != 65:
        raise ValueError('Expected 65 Chapter IX endpoints')
    print('[Endpoints]', len(reg), 'sex policy', p.risk_sex.value_counts().to_dict(), flush=True)
    return (reg.sort_values('disease').reset_index(drop=True), path)

def read_demographics():
    a = pd.read_csv(require(AGE_FILE), usecols=['eid', 'age'], dtype={'eid': str})
    b = pd.read_csv(require(SEX_FILE), usecols=['eid', 'sex'], dtype={'eid': str})
    for v in (a, b):
        v['eid'] = eid_clean(v.eid)
        if v.eid.isna().any() or v.eid.duplicated().any():
            raise ValueError('Missing or duplicated participant IDs in covariates')
    a.age = pd.to_numeric(a.age, errors='coerce')
    b.sex = normalize_sex(b.sex)
    d = a.merge(b, on='eid', validate='one_to_one').dropna(subset=['age', 'sex']).copy()
    if not np.isfinite(d.age).all() or not d.sex.isin([0, 1]).all():
        raise ValueError('Invalid age or sex values')
    return d[['eid', 'age', 'sex']]

def read_people(panel):
    base = read_demographics()
    assays = panel.coding.tolist()
    available = set(pd.read_csv(require(OLINK), nrows=0).columns)
    if not set(['eid'] + assays).issubset(available):
        raise ValueError('Missing assay columns: ' + str(list(set(['eid'] + assays) - available)[:15]))
    ids = set(base.eid)
    parts = []
    for block in pd.read_csv(OLINK, usecols=['eid'] + assays, dtype={'eid': str}, chunksize=CHUNK_SIZE,
        low_memory=False):
        block.eid = eid_clean(block.eid)
        block = block.loc[block.eid.isin(ids)].copy()
        if len(block):
            block[assays] = block[assays].apply(pd.to_numeric, errors='coerce')
            parts.append(block)
    if not parts:
        raise ValueError('No Olink/covariate overlap')
    m = pd.concat(parts, ignore_index=True)
    if m.eid.duplicated().any():
        raise ValueError('Olink duplicates by eid')
    d = base.merge(m, on='eid', validate='one_to_one')
    mask = np.ones(len(d), dtype=bool)
    for cls in CLASSES:
        code = panel.loc[panel.MoleculeClass.eq(cls), 'coding'].tolist()
        n_req = math.ceil(len(code) * MIN_COVERAGE)
        mask &= np.isfinite(d[code].to_numpy(float)).sum(axis=1) >= n_req
        print(f'[Coverage] {cls}: >={n_req}/{len(code)}', flush=True)
    d = d.loc[mask].sort_values('eid').reset_index(drop=True)
    print('[Common sample] N=', len(d), flush=True)
    return d

def read_baselines():
    d = pd.read_csv(require(BASELINE_DATES), usecols=['eid', 'date'], dtype={'eid': str})
    d.eid = eid_clean(d.eid)
    d['baseline_date'] = pd.to_datetime(d.date, errors='coerce')
    d = d[['eid', 'baseline_date']].dropna()
    if d.eid.duplicated().any():
        raise ValueError('Duplicate baseline records')
    return d

def read_deaths():
    if DEATH_DATES_FILE is None:
        return None
    d = pd.read_csv(require(DEATH_DATES_FILE), usecols=['eid', 'death_date'], dtype={'eid': str})
    d.eid = eid_clean(d.eid)
    d['death_date'] = pd.to_datetime(d.death_date, errors='coerce')
    if d.eid.duplicated().any():
        raise ValueError('Duplicate death dates')
    return d[['eid', 'death_date']]

def target_endpoints(reg):
    if RUN_MODE == 'PILOT':
        pilot = 'E4_DM2NOCOMP'
        if pilot not in set(reg.disease):
            raise ValueError(f'Pilot disease not found: {pilot}')
        return [pilot]
    if RUN_MODE == 'ALL_CVD':
        return reg.loc[reg.Root.str.match('^Chapter IX(?:\\s|$)'), 'disease'].tolist()
    if RUN_MODE == 'ALL_630':
        return reg.disease.tolist()
    raise ValueError('RUN_MODE should be PILOT/ALL_CVD/ALL_630')

def scope_on(scope):
    if scope == 'OFF':
        return False
    if scope == 'ALL':
        return True
    if scope == 'ALL_CVD_ONLY':
        return RUN_MODE == 'ALL_CVD'
    raise ValueError('Scope must be OFF/ALL_CVD_ONLY/ALL')

def age_sex_strata(people):
    age = pd.to_numeric(people.age).to_numpy(float)
    agebin = np.floor(age / AGE_BAND_YEARS).astype(int)
    sex = people.sex.astype(int).to_numpy()
    strata = sex.astype(str) + '_age' + agebin.astype(str)
    counts = pd.Series(strata).value_counts()
    strata = np.asarray([st if counts[st] >= 5 else f'{sex[i]}_other' for i, st in enumerate(strata)], dtype=object)
    if pd.Series(strata).value_counts().min() < 2:
        strata = sex.astype(str)
    return strata

def load_or_create_frozen_split(people, reg):
    """Keep old function interface; READ the exact V45 split, never create it.

    This prevents accidentally reporting a newly generated holdout while
    claiming to have reused V45. Signature is calculated according to the
    ORIGINAL Version-45 settings (not the new nested model settings).
    """
    if not SPLIT_FILE.is_file() or not SPLIT_METADATA.is_file():
        raise FileNotFoundError(f'Version 45 fixed split/metadata not found. Expected: {SPLIT_FILE} and {SPLIT_METADATA}. Do not regenerate a different random split in Version 46.')
    registry_hash = sha256_bytes(reg[['disease', 'risk_sex', 'Root']].to_csv(index=False).encode())
    expected_signature = manifest_signature([MAPPING, AGE_FILE, SEX_FILE, OLINK, BASELINE_DATES, ANNOTATION_FILE,
        DEATH_DATES_FILE], dict(version=SPLIT_VERSION, split_seed=SPLIT_SEED, test_fraction=TEST_FRACTION,
        age_band=AGE_BAND_YEARS, registry_sha256=registry_hash, min_coverage=MIN_COVERAGE, admin_censor=str(ADMIN_CENSOR),
        death_file=str(DEATH_DATES_FILE)))
    population_digest = sha256_bytes('\n'.join(people.eid.astype(str)).encode())
    metadata = json.loads(SPLIT_METADATA.read_text(encoding='utf-8'))
    if metadata.get('signature') != expected_signature or metadata.get('population_sha256') != population_digest:
        raise RuntimeError('Frozen Version-45 split metadata does not match the current inputs, population or original splitting settings. Check inputs; do not silently resplit or mix datasets.')
    split = pd.read_csv(SPLIT_FILE, sep='\t', dtype={'eid': str})
    if len(split) != len(people) or split.eid.duplicated().any() or set(split.eid) != set(people.eid.astype(str)) or (not split.dataset.isin(['Development',
        'Test']).all()):
        raise RuntimeError('The read-only Version-45 split fails integrity checks')
    counts = split.dataset.value_counts()
    if int(counts.get('Development', 0)) != int(metadata.get('n_development', -1)) or int(counts.get('Test',
        0)) != int(metadata.get('n_test', -1)):
        raise RuntimeError('Version-45 split participant counts mismatch')
    if 'age_sex_stratum' in split:
        original = split.sort_values('eid')['age_sex_stratum'].astype(str).to_numpy()
        rebuilt = pd.DataFrame({'eid': people.eid.astype(str),
            'stratum': age_sex_strata(people)}).sort_values('eid')['stratum'].to_numpy()
        if not np.array_equal(original, rebuilt):
            raise RuntimeError('Frozen split age-sex-stratification differs')
    split_sha = sha256_bytes(SPLIT_FILE.read_bytes())
    print('[FROZEN SPLIT] V45 READ-ONLY:', SPLIT_FILE, flush=True)
    print('[FROZEN SPLIT] sha256:', split_sha, flush=True)
    print('[FROZEN SPLIT] participants:', counts.to_dict(), flush=True)
    frozen_audit = dict(source=str(SPLIT_FILE), file_sha256=split_sha, original_metadata=metadata,
        source_version=SPLIT_VERSION)
    return (split, frozen_audit)

def make_endpoint_data(disease, people, risk_policy):
    case_file = CASE_DIR / f'{safe_name(disease)}.csv'
    ctrl_file = CASE_DIR / f'{safe_name(disease)}_control.csv'
    c = pd.read_csv(require(case_file), usecols=['eid', 'event_date'], dtype={'eid': str})
    z = pd.read_csv(require(ctrl_file), usecols=['eid'], dtype={'eid': str})
    c.eid = eid_clean(c.eid)
    z.eid = eid_clean(z.eid)
    c['event_date'] = pd.to_datetime(c.event_date, errors='coerce')
    if c.eid.isna().any() or z.eid.isna().any() or c.eid.duplicated().any() or z.eid.duplicated().any():
        raise ValueError('Case/control has null or repeated participant IDs')
    if len(set(c.eid) & set(z.eid)):
        raise ValueError('Case and control IDs overlap')
    c['event'] = True
    z['event'] = False
    z['event_date'] = pd.NaT
    pop_cols = people.columns.to_list()
    cohort = pd.concat([c, z], ignore_index=True).merge(people, on='eid', how='inner', validate='one_to_one')
    if risk_policy == 'female':
        cohort = cohort.loc[cohort.sex.eq(0)].copy()
    if risk_policy == 'male':
        cohort = cohort.loc[cohort.sex.eq(1)].copy()
    if cohort.empty:
        raise ValueError('No eligible participants in endpoint risk set')
    if risk_policy == 'all' and cohort.sex.nunique() < 2:
        raise ValueError('Both-sex outcome with one sex only')
    if cohort.loc[cohort.event, 'event_date'].isna().any():
        raise ValueError('Missing event dates')
    if (cohort.loc[cohort.event, 'event_date'] <= cohort.loc[cohort.event, 'baseline_date']).any():
        raise ValueError('Case event date <= baseline date')
    if (cohort.loc[cohort.event, 'event_date'] > ADMIN_CENSOR).any():
        raise ValueError('Case event after administrative censor')
    censor = pd.Series(ADMIN_CENSOR, index=cohort.index)
    if 'death_date' in cohort.columns:
        death = cohort.death_date
        after_death = cohort.event & death.notna() & cohort.event_date.gt(death)
        if after_death.any():
            raise ValueError('Case event date after recorded death')
        censor = pd.concat([censor, death], axis=1).min(axis=1)
    cohort['censor_date'] = pd.to_datetime(censor)
    cohort['end_date'] = cohort.censor_date
    cohort.loc[cohort.event, 'end_date'] = cohort.loc[cohort.event, 'event_date']
    cohort['time_days_orig'] = (cohort.end_date - cohort.baseline_date).dt.days.astype(float)
    cohort = cohort.loc[cohort.time_days_orig.gt(0)].copy()
    cohort['event_10y'] = cohort.event & cohort.time_days_orig.le(HORIZON_DAYS)
    cohort['time_days'] = cohort.time_days_orig.clip(upper=MAX_FOLLOWUP_DAYS)
    if not cohort.dataset.isin(['Development', 'Test']).all():
        raise RuntimeError('Unknown cohort split label')
    if cohort.eid.duplicated().any():
        raise RuntimeError('Duplicate endpoint participant')
    return cohort.reset_index(drop=True)

def audit_endpoints(people, reg):
    rows = []
    for i, row in enumerate(reg.itertuples(index=False), 1):
        try:
            d = make_endpoint_data(row.disease, people, row.risk_sex)
            dev = d.dataset.eq('Development')
            te = ~dev
            ndev = int(dev.sum())
            nte = int(te.sum())
            evdev = int((dev & d.event_10y).sum())
            evte = int((te & d.event_10y).sum())
            ctrte = int((te & d.time_days_orig.gt(HORIZON_DAYS)).sum())
            ctrdev = int((dev & d.time_days_orig.gt(HORIZON_DAYS)).sum())
            reasons = []
            if evdev < MIN_DEVELOPMENT_EVENTS:
                reasons.append('LOW_DEVELOPMENT_EVENTS')
            if ndev - evdev < MIN_DEVELOPMENT_NON_EVENTS:
                reasons.append('LOW_DEVELOPMENT_NON_EVENTS')
            if evte < MIN_TEST_EVENTS_10Y:
                reasons.append('LOW_TEST_EVENTS')
            if ctrte < MIN_TEST_CONTROLS_AFTER_10Y:
                reasons.append('LOW_TEST_10Y_CONTROLS')
            if d.loc[dev, 'time_days_orig'].max() <= HORIZON_DAYS:
                reasons.append('DEVELOPMENT_FOLLOWUP_LT_10Y')
            if d.loc[te, 'time_days_orig'].max() <= HORIZON_DAYS:
                reasons.append('TEST_FOLLOWUP_LT_10Y')
            rows.append(dict(disease=row.disease, DiseaseLabel=row.Disease, DiseaseSystem=row.Root,
                risk_sex=row.risk_sex, status='PASS' if not reasons else 'SKIP', reason=';'.join(reasons), N_total=len(d),
                N_development=ndev, N_test=nte, events10_total=int(d.event_10y.sum()), events10_development=evdev,
                events10_test=evte, controls_after10_development=ctrdev, controls_after10_test=ctrte,
                test_fraction_of_events=evte / (evdev + evte) if evdev + evte else np.nan,
                event_rate_development=evdev / ndev if ndev else np.nan, event_rate_test=evte / nte if nte else np.nan,
                early_censor_test=int((te & ~d.event_10y & d.time_days_orig.le(HORIZON_DAYS)).sum())))
        except Exception as ex:
            rows.append(dict(disease=row.disease, DiseaseLabel=row.Disease, DiseaseSystem=row.Root,
                risk_sex=row.risk_sex, status='ERROR', reason=f'{type(ex).__name__}: {ex}'))
        if i % 100 == 0 or i == len(reg):
            print(f'  [Risk-set audit] {i}/{len(reg)}', flush=True)
    return pd.DataFrame(rows)

def get_feature_layout(d, panel, risk_sex):
    cov = ['age'] + (['sex'] if risk_sex == 'all' else [])
    codes = panel.coding.tolist()
    raw = d[cov + codes].to_numpy(float)
    index = {}
    genes = {}
    for cls in CLASSES:
        pp = panel.loc[panel.MoleculeClass.eq(cls), ['gene', 'coding']].reset_index(drop=True)
        index[cls] = np.r_[np.arange(len(cov)), np.array([len(cov) + codes.index(s) for s in pp.coding], int)]
        genes[cls] = pp
    return (raw, len(cov), index, genes, cov, codes)

def train_only_transform(a, b, n_cov):
    """Mean impute / standardize proteins and age using training partition ONLY."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not (np.isfinite(a[:, :n_cov]).all() and np.isfinite(b[:, :n_cov]).all()):
        raise ValueError('Missing age/sex in model input')
    aa = np.where(np.isfinite(a), a, np.nan)
    bb = np.where(np.isfinite(b), b, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mu = np.nanmean(aa, axis=0)
        sd = np.nanstd(aa, axis=0, ddof=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    if n_cov == 2:
        mu[1] = 0.0
        sd[1] = 1.0
    xt = (np.where(np.isfinite(aa), aa, mu) - mu) / sd
    xv = (np.where(np.isfinite(bb), bb, mu) - mu) / sd
    if not (np.isfinite(xt).all() and np.isfinite(xv).all()):
        raise ValueError('Nonfinite transformed protein / covariate')
    return (xt, xv, mu, sd)

def surv_fit(d):
    return Surv.from_arrays(event=d.event_10y.to_numpy(bool), time=d.time_days.to_numpy(float))

def surv_reference(d):
    """Full follow-up for censoring estimation. Do not use artificial cap here."""
    return Surv.from_arrays(event=d.event.to_numpy(bool), time=d.time_days_orig.to_numpy(float))

def fit_lasso_path(x, y, n_cov, alphas=None):
    options = dict(l1_ratio=1.0, normalize=False, fit_baseline_model=False, penalty_factor=np.r_[np.zeros(n_cov),
        np.ones(x.shape[1] - n_cov)], max_iter=MAX_ITER, tol=TOL)
    if alphas is None:
        options.update(n_alphas=N_ALPHAS, alpha_min_ratio=ALPHA_MIN_RATIO)
    else:
        options['alphas'] = np.asarray(alphas, float)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = CoxnetSurvivalAnalysis(**options).fit(x, y)
    if len(model.alphas_) < 2 or not np.isfinite(model.coef_).all():
        raise ValueError('Coxnet alpha path invalid')
    return model

class HeldoutBreslowDeviance:
    """Negative twice *held-out* Cox partial log likelihood, per event."""

    def __init__(self, y):
        tt = np.asarray(y['time'], float)
        ev = np.asarray(y['event'], bool)
        if not np.isfinite(tt).all() or (tt <= 0).any() or ev.sum() < MIN_EVENTS_PER_INNER_VALID:
            raise ValueError('Too few inner validation events')
        self.order = np.argsort(-tt, kind='stable')
        times = tt[self.order]
        self.ev = ev[self.order]
        self.n = int(ev.sum())
        ends = np.flatnonzero(np.r_[np.diff(times) != 0, True])
        starts = np.r_[0, ends[:-1] + 1]
        counts = np.add.reduceat(self.ev.astype(float), starts)
        keep = counts > 0
        self.ends = ends[keep]
        self.counts = counts[keep]

    def score(self, risk):
        z = np.asarray(risk, float)[self.order]
        log_riskset = np.logaddexp.accumulate(z)
        return float(-2 * (z[self.ev].sum() - np.dot(self.counts, log_riskset[self.ends])) / self.n)

def choose_lambdas(raw, y, inds, n_cov, seed, experiment):
    """5-fold tuning within a specified development-training subset.
    As in the source code, alpha grid is computed on this subset; for strict
    nested CV the outer holdout is never accessed.
    """
    full, _, _, _ = train_only_transform(raw, raw, n_cov)
    all_path = fit_lasso_path(full[:, inds], y, n_cov)
    grid = np.asarray(all_path.alphas_, float)
    dev = np.full((INNER_FOLDS, len(grid)), np.nan)
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    for fold, (itr, ival) in enumerate(cv.split(raw, y['event'].astype(int)), 1):
        xt, xv, _, _ = train_only_transform(raw[itr], raw[ival], n_cov)
        m = fit_lasso_path(xt[:, inds], y[itr], n_cov, alphas=grid)
        eta = xv[:, inds] @ m.coef_
        scorer = HeldoutBreslowDeviance(y[ival])
        for j, alpha in enumerate(m.alphas_):
            jj = int(np.argmin(abs(grid - alpha)))
            if np.isclose(grid[jj], alpha, rtol=1e-07, atol=1e-12):
                dev[fold - 1, jj] = scorer.score(eta[:, j])
    good = np.isfinite(dev).all(axis=0)
    if not good.any():
        raise ValueError(f'{experiment}: no alpha evaluated in all inner folds')
    mu = dev[:, good].mean(axis=0)
    se = dev[:, good].std(axis=0, ddof=1) / np.sqrt(INNER_FOLDS)
    va = grid[good]
    imin = int(np.flatnonzero(mu <= mu.min() + 1e-12)[0])
    i1se = int(np.flatnonzero(mu <= mu[imin] + se[imin] + 1e-12)[0])
    chosen = {'min': float(va[imin]), '1se': float(va[i1se])}
    audit = []
    for j, a in enumerate(grid):
        audit.append(dict(experiment=experiment, alpha=float(a), inner_complete_folds=int(np.isfinite(dev[:,
            j]).sum()), mean_heldout_deviance=float(np.mean(dev[:, j])) if good[j] else np.nan,
            SE_heldout_deviance=float(np.std(dev[:, j], ddof=1) / np.sqrt(INNER_FOLDS)) if good[j] else np.nan,
            chosen_min=bool(np.isclose(a, chosen['min'])), chosen_1se=bool(np.isclose(a, chosen['1se']))))
    return (chosen, grid, pd.DataFrame(audit))

def selected_beta(path, alpha):
    j = int(np.argmin(np.abs(path.alphas_ - alpha)))
    if not np.isclose(path.alphas_[j], alpha, rtol=1e-06, atol=1e-12):
        raise ValueError('Alpha absent in final path')
    return np.asarray(path.coef_[:, j], float)

def fit_reference_model(x, y, n_cov):
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = CoxPHSurvivalAnalysis(alpha=0.0, ties='breslow', n_iter=150, tol=1e-07)
        model.fit(x[:, :n_cov], y)
    return np.asarray(model.coef_, float)

def evaluate_auc(yref, ytest, risk, minimum_cases, minimum_controls):
    result = dict(AUC_10y=np.nan, Cindex_10y_Uno=np.nan, reason='',
        test_events_10y=int(np.asarray(ytest['event']).sum()),
        test_controls_after_10y=int(np.sum(ytest['time'] > HORIZON_DAYS)))
    if result['test_events_10y'] < minimum_cases or result['test_controls_after_10y'] < minimum_controls:
        result['reason'] = 'INSUFFICIENT_TEST_EVENTS_OR_CONTROLS'
        return result
    if HORIZON_DAYS >= np.max(yref['time']) or HORIZON_DAYS >= np.max(ytest['time']):
        result['reason'] = 'FOLLOWUP_DOES_NOT_SUPPORT_10Y'
        return result
    risk = np.asarray(risk, float)
    if not np.isfinite(risk).all() or len(risk) != len(ytest):
        result['reason'] = 'INVALID_RISK_PREDICTION'
        return result
    try:
        value, _ = cumulative_dynamic_auc(yref, ytest, risk, np.array([HORIZON_DAYS]))
        result['AUC_10y'] = float(value[0])
    except Exception as ex:
        result['reason'] = f'AUC_FAILURE:{type(ex).__name__}: {ex}'
    try:
        result['Cindex_10y_Uno'] = float(concordance_index_ipcw(yref, ytest, risk, tau=HORIZON_DAYS)[0])
    except Exception:
        pass
    return result

def development_nested_cv(d, panel, risk_sex):
    """All model construction limited to the 80% Development population."""
    d = d.loc[d.dataset.eq('Development')].reset_index(drop=True)
    if d.eid.duplicated().any() or not d.dataset.eq('Development').all():
        raise RuntimeError('Nested CV may access Development participants only')
    raw, ncov, indices, genes, cov, codes = get_feature_layout(d, panel, risk_sex)
    y = surv_fit(d)
    splitter = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=MODEL_SEED)
    rows = []
    oof = []
    for f, (tr, te) in enumerate(splitter.split(raw, y['event'].astype(int)), 1):
        if int(y[tr]['event'].sum()) < INNER_FOLDS * MIN_EVENTS_PER_INNER_VALID:
            raise ValueError('Not enough development events for inner CV')
        xt, xv, _, _ = train_only_transform(raw[tr], raw[te], ncov)
        ytr = y[tr]
        yte = y[te]
        yr = surv_reference(d.iloc[tr])
        basecoef = fit_reference_model(xt, ytr, ncov)
        eta = xv[:, :ncov] @ basecoef
        e = evaluate_auc(yr, yte, eta, 10, 10)
        for rule in LAMBDA_RULES:
            rows.append(dict(outer_fold=f, model='Age+Sex', lambda_rule=rule, alpha=np.nan, n_selected=0,
                status='OK' if np.isfinite(e['AUC_10y']) else 'AUC_NA', **e))
        if SAVE_DEV_OOF:
            for ii, pred in zip(te, eta):
                oof.append(dict(eid=str(d.at[ii, 'eid']), outer_fold=f, model='Age+Sex', lambda_rule='shared',
                    log_risk=float(pred)))
        for c in CLASSES:
            inds = indices[c]
            try:
                chosen, grid, _ = choose_lambdas(raw[tr], ytr, inds, ncov, MODEL_SEED + f * 1000 + 731,
                    f'{c}_fold{f}')
                path = fit_lasso_path(xt[:, inds], ytr, ncov, alphas=grid)
                for rule in LAMBDA_RULES:
                    beta = selected_beta(path, chosen[rule])
                    pred = xv[:, inds] @ beta
                    out = evaluate_auc(yr, yte, pred, 10, 10)
                    nsel = int(np.sum(np.abs(beta[ncov:]) > SELECTED_THRESHOLD))
                    rows.append(dict(outer_fold=f, model=c, lambda_rule=rule, alpha=chosen[rule], n_selected=nsel,
                        status='OK' if np.isfinite(out['AUC_10y']) else 'AUC_NA', **out))
                    if SAVE_DEV_OOF:
                        for ii, lp in zip(te, pred):
                            oof.append(dict(eid=str(d.at[ii, 'eid']), outer_fold=f, model=c, lambda_rule=rule,
                                log_risk=float(lp)))
            except Exception as ex:
                for rule in LAMBDA_RULES:
                    rows.append(dict(outer_fold=f, model=c, lambda_rule=rule, alpha=np.nan, n_selected=np.nan,
                        status='FAILED', reason=f'{type(ex).__name__}: {ex}', AUC_10y=np.nan, Cindex_10y_Uno=np.nan))
    return (pd.DataFrame(rows), pd.DataFrame(oof))

def final_full_development_fit(d, panel, risk_sex, disease, output_folder):
    """After nested CV, re-tune lambda on full 80% Development and refit.
    Returns fixed test predictions, model metrics, weights, tuning audits.
    """
    development = d.loc[d.dataset.eq('Development')].reset_index(drop=True)
    test = d.loc[d.dataset.eq('Test')].reset_index(drop=True)
    if set(development.eid.astype(str)) & set(test.eid.astype(str)):
        raise RuntimeError('Data leakage: overlap between Development and Test')
    if development.empty or test.empty:
        raise RuntimeError('Empty Development or Test risk set')
    raw_dev, ncov, indices, genes, cov, codes = get_feature_layout(development, panel, risk_sex)
    raw_test = test[cov + codes].to_numpy(float)
    yt = surv_fit(test)
    yd = surv_fit(development)
    yr = surv_reference(development)
    xt, xv, mu, sd = train_only_transform(raw_dev, raw_test, ncov)
    records = []
    weights = []
    audits = []
    predictions = []
    beta0 = fit_reference_model(xt, yd, ncov)
    eta0 = xv[:, :ncov] @ beta0
    base_eval = evaluate_auc(yr, yt, eta0, MIN_TEST_EVENTS_10Y, MIN_TEST_CONTROLS_AFTER_10Y)
    for rule in LAMBDA_RULES:
        records.append(dict(disease=disease, model='Age+Sex', lambda_rule=rule, alpha=np.nan, n_selected=0,
            status='OK' if np.isfinite(base_eval['AUC_10y']) else 'AUC_NA', **base_eval))
    for i, lp in enumerate(eta0):
        predictions.append(dict(eid=str(test.at[i, 'eid']), disease=disease, dataset='Test', model='Age+Sex',
            lambda_rule='shared', log_risk=float(lp), event_10y=bool(test.at[i, 'event_10y']), time_days=float(test.at[i,
            'time_days']), event_full=bool(test.at[i, 'event']), time_days_orig=float(test.at[i, 'time_days_orig'])))
    output_folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_folder / 'final_AgeSex_model.npz', coef=beta0, mu=mu[:ncov], sd=sd[:ncov],
        columns=np.asarray(cov), model='Age+Sex', disease=disease)
    for c in CLASSES:
        inds = indices[c]
        try:
            chosen, grid, audit = choose_lambdas(raw_dev, yd, inds, ncov, MODEL_SEED + 702911,
                f'{c}_FINAL_DEVELOPMENT')
            audit['disease'] = disease
            audit['model'] = c
            audit['phase'] = 'final_80pct_development'
            audits.append(audit)
            fit = fit_lasso_path(xt[:, inds], yd, ncov, alphas=grid)
            for rule in LAMBDA_RULES:
                beta = selected_beta(fit, chosen[rule])
                eta = xv[:, inds] @ beta
                nsel = int(np.sum(np.abs(beta[ncov:]) > SELECTED_THRESHOLD))
                ev = evaluate_auc(yr, yt, eta, MIN_TEST_EVENTS_10Y, MIN_TEST_CONTROLS_AFTER_10Y)
                records.append(dict(disease=disease, model=c, lambda_rule=rule, alpha=chosen[rule],
                    n_selected=nsel, status='OK' if np.isfinite(ev['AUC_10y']) else 'AUC_NA', **ev))
                for j, g in genes[c].iterrows():
                    b = float(beta[ncov + j])
                    if abs(b) <= SELECTED_THRESHOLD:
                        continue
                    col = inds[ncov + j]
                    weights.append(dict(disease=disease, model=c, lambda_rule=rule, gene=g.gene, coding=g.coding,
                        alpha=float(chosen[rule]), beta_per_development_protein_SD=b, beta_per_raw_protein_unit=b / sd[col],
                        development_protein_mean=float(mu[col]), development_protein_SD=float(sd[col])))
                for i, lp in enumerate(eta):
                    predictions.append(dict(eid=str(test.at[i, 'eid']), disease=disease, dataset='Test', model=c,
                        lambda_rule=rule, log_risk=float(lp), event_10y=bool(test.at[i, 'event_10y']), time_days=float(test.at[i,
                        'time_days']), event_full=bool(test.at[i, 'event']), time_days_orig=float(test.at[i, 'time_days_orig'])))
                np.savez_compressed(output_folder / f'final_{c}_{rule}_model.npz', coef=beta, mu=mu[inds],
                    sd=sd[inds], columns=np.asarray(cov + genes[c].coding.tolist()), alpha=float(chosen[rule]), disease=disease,
                    model=c, lambda_rule=rule)
        except Exception as ex:
            for rule in LAMBDA_RULES:
                records.append(dict(disease=disease, model=c, lambda_rule=rule, alpha=np.nan, n_selected=np.nan,
                    status='FAILED', reason=f'{type(ex).__name__}: {ex}', AUC_10y=np.nan, Cindex_10y_Uno=np.nan))
    for r in records:
        r['N_development'] = int(len(development))
        r['N_test'] = int(len(test))
        r['events10_development'] = int(development.event_10y.sum())
    return (pd.DataFrame(records), pd.DataFrame(weights), pd.concat(audits,
        ignore_index=True) if audits else pd.DataFrame(), pd.DataFrame(predictions))

def delta_vs_age_sex(metrics):
    x = metrics.copy()
    base = x.loc[x.model.eq('Age+Sex') & x.lambda_rule.eq('min'), ['disease', 'AUC_10y',
        'Cindex_10y_Uno']].rename(columns={'AUC_10y': 'AUC_age_sex', 'Cindex_10y_Uno': 'Cindex_age_sex'})
    if base.disease.duplicated().any():
        raise ValueError('Duplicate baseline disease')
    x = x.merge(base, on='disease', how='left', validate='many_to_one')
    x['Delta_AUC_10y'] = x.AUC_10y - x.AUC_age_sex
    x['Delta_Cindex_10y_Uno'] = x.Cindex_10y_Uno - x.Cindex_age_sex
    x.loc[x.model.eq('Age+Sex'), 'Delta_AUC_10y'] = 0.0
    x.loc[x.model.eq('Age+Sex'), 'Delta_Cindex_10y_Uno'] = 0.0
    x.loc[x.AUC_10y.isna(), 'Delta_AUC_10y'] = np.nan
    return x

def development_cv_summary(frame, disease):
    result = []
    for (model, rule), g in frame.groupby(['model', 'lambda_rule']):
        v = pd.to_numeric(g.AUC_10y, errors='coerce').dropna()
        result.append(dict(disease=disease, model=model, lambda_rule=rule, n_outer_valid=int(v.size),
            development_outer_mean_AUC_10y=float(v.mean()) if v.size else np.nan,
            development_outer_SD_AUC_10y=float(v.std(ddof=1)) if v.size > 1 else np.nan,
            mean_proteins_selected=float(pd.to_numeric(g.n_selected, errors='coerce').mean())))
    return pd.DataFrame(result)

def stable_seed(*parts):
    return int.from_bytes(hashlib.blake2b('|'.join(map(str, (SPLIT_SEED,) + parts)).encode(),
        digest_size=8).digest(), 'little')

def dynamic_matched_holdout(d, panel, risk_sex, disease, reference_auc):
    dev = d.loc[d.dataset.eq('Development')].reset_index(drop=True)
    test = d.loc[d.dataset.eq('Test')].reset_index(drop=True)
    raw, ncov, _, _, cov, codes = get_feature_layout(dev, panel, risk_sex)
    raw_test = test[cov + codes].to_numpy(float)
    xt, xv, _, _ = train_only_transform(raw, raw_test, ncov)
    yd = surv_fit(dev)
    yt = surv_fit(test)
    yr = surv_reference(dev)
    target_codes = panel.loc[panel.MoleculeClass.eq('Dynamic-only'), 'coding'].tolist()
    target_index = {code: ncov + codes.index(code) for code in target_codes}
    result = []
    for size in MATCH_TARGETS:
        for repeat in range(MATCH_REPEATS):
            rng = np.random.default_rng(stable_seed(disease, 'dynamic_matched', size, repeat))
            selected = sorted(rng.choice(target_codes, size=size, replace=False).tolist())
            inds = np.r_[np.arange(ncov), np.asarray([target_index[c] for c in selected], int)]
            try:
                chosen, grid, _ = choose_lambdas(raw, yd, inds, ncov, MODEL_SEED + 702911,
                    f'matched_{size}_r{repeat}')
                path = fit_lasso_path(xt[:, inds], yd, ncov, alphas=grid)
                for rule in LAMBDA_RULES:
                    coef = selected_beta(path, chosen[rule])
                    eta = xv[:, inds] @ coef
                    e = evaluate_auc(yr, yt, eta, MIN_TEST_EVENTS_10Y, MIN_TEST_CONTROLS_AFTER_10Y)
                    result.append(dict(disease=disease, model=f'Dynamic_matched_{size}', candidate_n=size,
                        repeat=repeat, lambda_rule=rule, AUC_age_sex=reference_auc,
                        Delta_AUC_10y=e['AUC_10y'] - reference_auc if np.isfinite(e['AUC_10y']) and np.isfinite(reference_auc) else np.nan,
                        status='OK' if np.isfinite(e['AUC_10y']) else 'AUC_NA', **e))
            except Exception as ex:
                for rule in LAMBDA_RULES:
                    result.append(dict(disease=disease, model=f'Dynamic_matched_{size}', candidate_n=size,
                        repeat=repeat, lambda_rule=rule, AUC_10y=np.nan, Delta_AUC_10y=np.nan, status='FAILED',
                        reason=f'{type(ex).__name__}: {ex}'))
    return pd.DataFrame(result)

def bootstrap_fixed_holdout(d, pred, metric, B):
    """Paired subject-level bootstrap conditional on fixed trained models.
    IPCW censoring reference estimated on the original Development cohort.
    No rerandomization or refitting is performed in bootstrap.
    """
    dev = d.loc[d.dataset.eq('Development')]
    yref = surv_reference(dev)
    long = pred.copy()
    baseline = long.loc[long.model.eq('Age+Sex'), ['eid', 'log_risk']].rename(columns={'log_risk': 'Age+Sex'})
    if baseline.eid.duplicated().any():
        return pd.DataFrame()
    wide = long.loc[long.model.eq('Age+Sex'), ['eid', 'event_10y', 'time_days', 'time_days_orig']].copy()
    wide = wide.merge(baseline, on='eid', validate='one_to_one')
    for cls in CLASSES:
        sub = long.loc[long.model.eq(cls) & long.lambda_rule.eq('min'), ['eid',
            'log_risk']].rename(columns={'log_risk': cls})
        wide = wide.merge(sub, on='eid', how='left', validate='one_to_one')
    if wide[list(MODELS)].isna().any().any():
        return pd.DataFrame()
    y = Surv.from_arrays(event=wide.event_10y.to_numpy(bool), time=wide.time_days.to_numpy(float))
    n = len(y)
    rng = np.random.default_rng(stable_seed(str(d.disease.iloc[0]) if 'disease' in d else 'boot', 'holdout'))
    rows = []
    for rep in range(B):
        ii = rng.integers(0, n, n)
        ys = y[ii]
        if int(ys['event'].sum()) < MIN_TEST_EVENTS_10Y or np.sum(ys['time'] > HORIZON_DAYS) < MIN_TEST_CONTROLS_AFTER_10Y:
            continue
        vals = {}
        for model in MODELS:
            v = evaluate_auc(yref, ys, wide[model].to_numpy(float)[ii], MIN_TEST_EVENTS_10Y,
                MIN_TEST_CONTROLS_AFTER_10Y)['AUC_10y']
            vals[model] = v
        for model in CLASSES:
            if np.isfinite(vals[model]) and np.isfinite(vals['Age+Sex']):
                rows.append(dict(bootstrap_rep=rep, model=model, Delta_AUC_10y=vals[model] - vals['Age+Sex']))
        for a, b in [('Dynamic-only', 'Baseline-only'), ('Dynamic-only', 'Dual-response')]:
            if np.isfinite(vals[a]) and np.isfinite(vals[b]):
                rows.append(dict(bootstrap_rep=rep, model=f'{a} minus {b}', Delta_AUC_10y=vals[a] - vals[b]))
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    res = []
    for model, g in data.groupby('model'):
        z = g.Delta_AUC_10y.dropna().to_numpy(float)
        res.append(dict(model=model, n_boot_valid=len(z), CI95_low=float(np.percentile(z,
            2.5)) if len(z) >= 50 else np.nan, CI95_high=float(np.percentile(z, 97.5)) if len(z) >= 50 else np.nan,
            median=float(np.median(z)) if len(z) else np.nan,
            note='Conditional fixed-model test-set bootstrap; not full-pipeline CI'))
    return pd.DataFrame(res)
_SHARED = {}

def disease_worker(disease, fingerprint, do_match, do_boot):
    folder = OUT / 'per_endpoint' / safe_name(disease)
    folder.mkdir(parents=True, exist_ok=True)
    status_file = folder / 'job_signature.json'
    log_file = folder / 'worker.log'
    required_files = [folder / 'dev_outer_metrics.tsv', folder / 'test_metrics.tsv', folder / 'final_weights.tsv',
        folder / 'final_lambda_cv.tsv']
    if CACHE and status_file.is_file() and all((p.is_file() for p in required_files)):
        try:
            q = json.loads(status_file.read_text())
            if q.get('signature') == fingerprint and q.get('status') == 'OK':
                return dict(disease=disease, status='CACHED', reason='')
        except Exception:
            pass
    try:
        with log_file.open('w', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
                start = time.perf_counter()
                reg = _SHARED['registry'].set_index('disease')
                risk = str(reg.at[disease, 'risk_sex'])
                d = make_endpoint_data(disease, _SHARED['people'], risk)
                dev = d.dataset.eq('Development')
                te = ~dev
                ev_dev = int((dev & d.event_10y).sum())
                ev_test = int((te & d.event_10y).sum())
                ctr_test = int((te & d.time_days_orig.gt(HORIZON_DAYS)).sum())
                if ev_dev < MIN_DEVELOPMENT_EVENTS or len(d.loc[dev]) - ev_dev < MIN_DEVELOPMENT_NON_EVENTS:
                    raise ValueError(f'Predefined SKIP: Development events={ev_dev}')
                if ev_test < MIN_TEST_EVENTS_10Y or ctr_test < MIN_TEST_CONTROLS_AFTER_10Y:
                    raise ValueError(f'Predefined SKIP: Test cases={ev_test}, controls after 10y={ctr_test}')
                print(f'[Endpoint] {disease} Development={int(dev.sum()):,} events={ev_dev}; Test={int(te.sum()):,} events={ev_test}',
                    flush=True)
                inner, oof = development_nested_cv(d, _SHARED['panel'], risk)
                hold, weights, audit, pred = final_full_development_fit(d, _SHARED['panel'], risk, disease, folder)
                atomic_tsv(inner, folder / 'dev_outer_metrics.tsv')
                atomic_tsv(development_cv_summary(inner, disease), folder / 'dev_cv_summary.tsv')
                atomic_tsv(hold, folder / 'test_metrics.tsv')
                atomic_tsv(weights, folder / 'final_weights.tsv')
                atomic_tsv(audit, folder / 'final_lambda_cv.tsv')
                if SAVE_DEV_OOF and (not oof.empty):
                    oof.to_csv(folder / 'development_oof_predictions.tsv.gz', sep='\t', index=False,
                        compression='gzip')
                if SAVE_TEST_PREDICTIONS:
                    pred.to_csv(folder / 'test_predictions.tsv.gz', sep='\t', index=False, compression='gzip')
                if do_match:
                    mm = dynamic_matched_holdout(d, _SHARED['panel'], risk, disease,
                        float(hold.loc[hold.model.eq('Age+Sex') & hold.lambda_rule.eq('min'), 'AUC_10y'].iloc[0]))
                    atomic_tsv(mm, folder / 'dynamic_matched_test.tsv')
                if do_boot:
                    ci = bootstrap_fixed_holdout(d, pred, hold, BOOTSTRAP_B)
                    atomic_tsv(ci, folder / 'conditional_holdout_bootstrap.tsv')
                status_file.write_text(json.dumps({'signature': fingerprint, 'status': 'OK', 'version': VERSION,
                    'elapsed_s': time.perf_counter() - start}), encoding='utf-8')
                print(f'[DONE] {disease} in {(time.perf_counter() - start) / 60:.2f} min', flush=True)
        return dict(disease=disease, status='OK', reason='')
    except Exception as ex:
        reason = f'{type(ex).__name__}: {ex}'
        with log_file.open('a', encoding='utf-8') as log:
            log.write('\n[FAILED] ' + reason + '\n')
            traceback.print_exc(file=log)
        return dict(disease=disease, status='FAILED', reason=reason)

def run_jobs(targets, signatures, do_match, do_boot):
    args = [(d, signatures[d], do_match, do_boot) for d in targets]
    if N_JOBS == 1:
        for arg in args:
            yield disease_worker(*arg)
        return
    if sys.platform != 'linux':
        raise RuntimeError('Linux fork workers required for N_JOBS>1; set N_JOBS=1 otherwise')
    if 'ipykernel' in sys.modules:
        raise RuntimeError('Launch as a standalone .py in a Linux terminal, not Jupyter')
    with ProcessPoolExecutor(max_workers=N_JOBS, mp_context=mp.get_context('fork')) as ex:
        futures = {ex.submit(disease_worker, *arg): arg[0] for arg in args}
        for f in as_completed(futures):
            try:
                yield f.result()
            except Exception as excep:
                yield dict(disease=futures[f], status='FAILED', reason=str(excep))

def collect_results(good, do_match, do_boot):
    names = {'dev_outer_metrics.tsv': '01_development_outer_fold_metrics.tsv',
        'dev_cv_summary.tsv': '02_development_cv_summary.tsv',
        'test_metrics.tsv': '03_independent_holdout_four_models.tsv',
        'final_weights.tsv': '05_final_model_protein_weights.tsv', 'final_lambda_cv.tsv': '06_final_model_lambda_audit.tsv'}
    outputs = {}
    for filename, destination in names.items():
        parts = []
        for disease in good:
            f = OUT / 'per_endpoint' / safe_name(disease) / filename
            if f.is_file() and f.stat().st_size > 0:
                try:
                    x = pd.read_csv(f, sep='\t', low_memory=False)
                    if len(x):
                        parts.append(x)
                except pd.errors.EmptyDataError:
                    pass
        if parts:
            outputs[filename] = pd.concat(parts, ignore_index=True)
            atomic_tsv(outputs[filename], OUT / destination)
        else:
            atomic_tsv(pd.DataFrame(), OUT / destination)
    if 'test_metrics.tsv' in outputs:
        hold = delta_vs_age_sex(outputs['test_metrics.tsv'])
        reglabels = _SHARED['registry'][['disease', 'Disease', 'Root']].rename(columns={'Disease': 'DiseaseLabel',
            'Root': 'DiseaseSystem'})
        hold = hold.merge(reglabels, on='disease', how='left', validate='many_to_one')
        atomic_tsv(hold, OUT / '03_independent_holdout_four_models.tsv')
        save_system_summaries(hold)
        focus = hold.loc[hold.lambda_rule.eq('min') & hold.model.isin(CLASSES), ['disease', 'model', 'AUC_10y',
            'Delta_AUC_10y']].copy()
        z = focus.pivot(index='disease', columns='model', values='AUC_10y')
        pairs = []
        for a, b in [('Dynamic-only', 'Baseline-only'), ('Dynamic-only', 'Dual-response'), ('Dual-response',
            'Baseline-only')]:
            if {a, b}.issubset(z):
                for d, v in (z[a] - z[b]).items():
                    pairs.append(dict(disease=d, model_A=a, model_B=b,
                        delta_AUC10_A_minus_B=float(v) if np.isfinite(v) else np.nan))
        atomic_tsv(pd.DataFrame(pairs), OUT / '04_independent_holdout_paired_deltas.tsv')
    if do_match:
        parts = []
        for disease in good:
            file = OUT / 'per_endpoint' / safe_name(disease) / 'dynamic_matched_test.tsv'
            if file.is_file():
                x = pd.read_csv(file, sep='\t')
                if len(x):
                    parts.append(x)
        if parts:
            z = pd.concat(parts, ignore_index=True)
            atomic_tsv(z, OUT / '08_dynamic_matched_holdout_repeats.tsv')
            s = []
            for (d, model, rule), g in z.groupby(['disease', 'model', 'lambda_rule']):
                v = g.Delta_AUC_10y.dropna().to_numpy(float)
                s.append(dict(disease=d, model=model, lambda_rule=rule, n_valid_repeats=len(v),
                    mean_Delta_AUC_10y=float(np.mean(v)) if len(v) else np.nan,
                    median_Delta_AUC_10y=float(np.median(v)) if len(v) else np.nan, q05_Delta_AUC_10y=float(np.percentile(v,
                    5)) if len(v) else np.nan, q95_Delta_AUC_10y=float(np.percentile(v, 95)) if len(v) else np.nan,
                    note='Panel-sampling variability; not a 95% confidence interval'))
            atomic_tsv(pd.DataFrame(s), OUT / '09_dynamic_matched_holdout_summary.tsv')
    if do_boot:
        parts = []
        for disease in good:
            file = OUT / 'per_endpoint' / safe_name(disease) / 'conditional_holdout_bootstrap.tsv'
            if file.is_file():
                try:
                    x = pd.read_csv(file, sep='\t')
                    if len(x):
                        x['disease'] = disease
                        parts.append(x)
                except pd.errors.EmptyDataError:
                    pass
        if parts:
            atomic_tsv(pd.concat(parts, ignore_index=True), OUT / '10_conditional_holdout_bootstrap_CI.tsv')
    return outputs

def save_nested_paired_comparisons():
    """Paired AUC contrasts on the SAME development outer test folds.

    Fold scores are correlated; report descriptive contrasts without treating
    the 5 folds as five independent biological cohorts.
    """
    path = OUT / '01_development_outer_fold_metrics.tsv'
    if not path.is_file() or path.stat().st_size == 0:
        return
    try:
        d = pd.read_csv(path, sep='\t', low_memory=False)
    except pd.errors.EmptyDataError:
        return
    keys = ['disease', 'outer_fold', 'lambda_rule']
    if not set(keys + ['model', 'AUC_10y']).issubset(d):
        return
    if d.duplicated(keys + ['model']).any():
        raise RuntimeError('Repeated model/fold rows in nested CV outputs')
    table = d.pivot(index=keys, columns='model', values='AUC_10y')
    if not set(MODELS).issubset(table.columns):
        return
    table = table.dropna(subset=list(MODELS)).reset_index()
    if table.empty:
        return
    for model in CLASSES:
        table['Delta_AUC_10y_' + model] = table[model] - table['Age+Sex']
    table['Dynamic_minus_Baseline'] = table['Dynamic-only'] - table['Baseline-only']
    table['Dynamic_minus_Dual'] = table['Dynamic-only'] - table['Dual-response']
    atomic_tsv(table, OUT / '16_nested_cv_paired_outer_fold_AUC.tsv')
    rows = []
    for (disease, rule), g in table.groupby(['disease', 'lambda_rule'], sort=True):
        for col in list(MODELS) + [f'Delta_AUC_10y_{m}' for m in CLASSES] + ['Dynamic_minus_Baseline',
            'Dynamic_minus_Dual']:
            vals = g[col].to_numpy(float)
            rows.append(dict(disease=disease, lambda_rule=rule, metric=col, n_complete_outer_folds=len(vals),
                mean=float(np.mean(vals)), SD=float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                median=float(np.median(vals)), caution='Outer-fold descriptive SD; correlated folds, not an external validation CI'))
    atomic_tsv(pd.DataFrame(rows), OUT / '17_nested_cv_endpoint_paired_summary.tsv')

def bh_fdr(pvalues):
    """Benjamini-Hochberg across specified nonmissing tests only."""
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if len(good):
        order = np.argsort(p[good])
        rank = np.arange(1, len(good) + 1, dtype=float)
        sorted_p = p[good][order]
        vals = np.minimum.accumulate((sorted_p * len(good) / rank)[::-1])[::-1]
        q[good[order]] = np.clip(vals, 0, 1)
    return q

def save_system_summaries(hold):
    """Figure 6C/D-ready summaries: require all FOUR model test AUCs.
    Wilcoxon across diseases is exploratory: endpoints can be dependent.
    """
    primary = hold.loc[hold.lambda_rule.eq('min') & hold.model.isin(MODELS)].copy()
    matrix = primary.pivot(index='disease', columns='model', values='AUC_10y')
    if not set(MODELS).issubset(matrix.columns):
        return
    complete_codes = matrix.dropna(subset=list(MODELS)).index
    primary = primary.loc[primary.disease.isin(complete_codes)]
    atomic_tsv(primary, OUT / '13_test_complete_four_model_endpoints.tsv')
    systems = []
    tests = []
    for system, group in primary.groupby('DiseaseSystem', sort=True):
        piv = group.pivot(index='disease', columns='model', values='Delta_AUC_10y')
        if not set(CLASSES).issubset(piv):
            continue
        for model in CLASSES:
            v = piv[model].to_numpy(float)
            systems.append(dict(DiseaseSystem=system, model=model, n_diseases=int(len(v)),
                median_Delta_AUC_10y=float(np.median(v)), mean_Delta_AUC_10y=float(np.mean(v)),
                q25_Delta_AUC_10y=float(np.percentile(v, 25)), q75_Delta_AUC_10y=float(np.percentile(v, 75))))
        for a, b in [('Baseline-only', 'Dual-response'), ('Baseline-only', 'Dynamic-only'), ('Dual-response',
            'Dynamic-only')]:
            diff = (piv[a] - piv[b]).to_numpy(float)
            pval = np.nan
            if len(diff) >= 3 and np.any(diff != 0):
                try:
                    pval = float(wilcoxon(diff, zero_method='wilcox').pvalue)
                except ValueError:
                    pass
            tests.append(dict(DiseaseSystem=system, model_A=a, model_B=b, n_diseases=len(diff),
                median_A_minus_B=float(np.median(diff)), p_value=pval,
                caveat='Exploratory disease-level paired test; correlated endpoints'))
    if systems:
        atomic_tsv(pd.DataFrame(systems), OUT / '14_test_12system_median_DeltaAUC.tsv')
    if tests:
        df = pd.DataFrame(tests)
        df['FDR_BH_across_system_pairs'] = bh_fdr(df.p_value.to_numpy(float))
        atomic_tsv(df, OUT / '15_test_system_pairwise_Wilcoxon_BHFDR.tsv')

def plot_holdout_pdf(metrics, registry):
    if metrics.empty:
        return
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'font.size': 8, 'axes.labelcolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black'})
    x = delta_vs_age_sex(metrics)
    x = x.loc[x.lambda_rule.eq('min')].copy()
    label = dict(zip(registry.disease, registry.Disease))
    order = [d for d in registry.disease if d in set(x.disease)]
    for quantity, filename in [('AUC_10y', '11_independent_test_AUC10y_A4.pdf'), ('Delta_AUC_10y',
        '12_independent_test_DeltaAUC10y_A4.pdf')]:
        with PdfPages(OUT / filename) as pdf:
            for st in range(0, len(order), 12):
                codes = order[st:st + 12]
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                fig.subplots_adjust(left=0.48, right=0.95, top=0.88, bottom=0.1)
                yy = np.arange(len(codes))
                names = list(MODELS) if quantity == 'AUC_10y' else list(CLASSES)
                barwidth = 0.76 / len(names)
                values = []
                for j, name in enumerate(names):
                    one = x.loc[x.model.eq(name)].set_index('disease').reindex(codes)
                    vv = one[quantity].to_numpy(float)
                    values.extend(vv[np.isfinite(vv)].tolist())
                    ax.barh(yy + (j - (len(names) - 1) / 2) * barwidth, vv, height=barwidth * 0.87,
                        color=COLORS[name], label=name)
                ylabels = []
                for code in codes:
                    desc = str(label.get(code, code))
                    if len(desc) > 40:
                        desc = desc[:37] + '...'
                    ylabels.append(f'{code} | {desc}')
                ax.set_yticks(yy, ylabels, fontsize=7)
                ax.invert_yaxis()
                ax.spines[['top', 'right']].set_visible(False)
                ax.grid(axis='x', alpha=0.15)
                ax.set_axisbelow(True)
                if quantity.startswith('Delta'):
                    ax.axvline(0, color='gray', lw=0.8)
                    bound = max(0.04, max((abs(v) for v in values), default=0) + 0.025)
                    ax.set_xlim(-bound, bound)
                elif values:
                    ax.set_xlim(max(0.0, min(values) - 0.08), min(1.0, max(values) + 0.06))
                ax.set_xlabel('Independent hold-out 10-year ' + ('IPCW AUC' if quantity == 'AUC_10y' else 'IPCW AUC gain vs Age + Sex'))
                ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.015), ncol=4, fontsize=8, frameon=False)
                fig.suptitle(f'UKB 80% development / 20% fixed hold-out | {st + 1}-{st + len(codes)}', y=0.965,
                    fontsize=10)
                fig.text(0.06, 0.035,
                    'Age + Sex (Age only for sex-specific endpoints). Nested 5x5 CV in development only; final model refit on full 80%. AUC from fixed 20% internal holdout.',
                    fontsize=7)
                pdf.savefig(fig)
                plt.close(fig)

def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if not 0 < TEST_FRACTION < 0.5:
        raise ValueError('TEST_FRACTION must be between 0 and .5')
    if N_JOBS > 1 and 'ipykernel' in sys.modules:
        raise RuntimeError('Save this code as .py and run in Linux terminal; not Jupyter')
    if OUTER_FOLDS != 5 or INNER_FOLDS != 5:
        raise ValueError('Version 46 requires precisely 5 outer and 5 inner folds')
    print('[Project]', VERSION, flush=True)
    panel = read_panel()
    registry, policy_path = read_policy_and_registry()
    people = read_people(panel)
    dates = read_baselines()
    people = people.merge(dates, on='eid', how='inner', validate='one_to_one')
    deaths = read_deaths()
    if deaths is not None:
        people = people.merge(deaths, on='eid', how='left', validate='one_to_one')
    people = people.loc[people.baseline_date.lt(ADMIN_CENSOR)].sort_values('eid').reset_index(drop=True)
    print('[Split-eligible population] N=', len(people), flush=True)
    if len(people) < 1000:
        raise ValueError('Implausibly small UKB common cohort')
    split, metadata = load_or_create_frozen_split(people, registry)
    people = people.merge(split[['eid', 'dataset']], on='eid', validate='one_to_one')
    print('[Fixed split]', people.dataset.value_counts().to_dict(), flush=True)
    audit = audit_endpoints(people, registry)
    atomic_tsv(audit, SPLIT_DIR / '01_all630_endpoint_split_audit.tsv')
    atomic_tsv(audit, OUT / '00_all630_endpoint_split_audit.tsv')
    print('[Audit]', audit.status.value_counts(dropna=False).to_dict(), flush=True)
    targets = target_endpoints(registry)
    target_audit = audit.set_index('disease').reindex(targets)
    ready = [d for d in targets if target_audit.at[d, 'status'] == 'PASS']
    skipped = [dict(disease=d, status=str(target_audit.at[d, 'status']), reason=str(target_audit.at[d,
        'reason'])) for d in targets if d not in set(ready)]
    print(f'[Endpoints] selected={len(targets)}; eligible={len(ready)}; predefined skipped={len(skipped)}',
        flush=True)
    if not ready:
        atomic_tsv(pd.DataFrame(skipped), OUT / '07_endpoint_runlog.tsv')
        raise RuntimeError('No endpoints pass pre-declared feasibility criteria. Inspect audit.')
    do_match = scope_on(MATCH_SCOPE)
    do_boot = scope_on(BOOTSTRAP_SCOPE)
    print(f'[Settings] RUN_MODE={RUN_MODE}, workers={N_JOBS}, matched={do_match}, bootstrap={do_boot}, output={OUT}',
        flush=True)
    fingerprints = {}
    common_files = [MAPPING, AGE_FILE, SEX_FILE, OLINK, BASELINE_DATES, policy_path, ANNOTATION_FILE,
        DEATH_DATES_FILE, SPLIT_FILE, SPLIT_METADATA]
    settings = dict(version=VERSION, mode=RUN_MODE, outer=OUTER_FOLDS, inner=INNER_FOLDS, alpha=N_ALPHAS,
        alpha_ratio=ALPHA_MIN_RATIO, lambda_rules=LAMBDA_RULES, holdout_seed=SPLIT_SEED, model_seed=MODEL_SEED,
        admin=str(ADMIN_CENSOR), dev_min=MIN_DEVELOPMENT_EVENTS, test_min=MIN_TEST_EVENTS_10Y, match=do_match,
        match_repeats=MATCH_REPEATS if do_match else 0, bootstrap=do_boot, bootstrap_B=BOOTSTRAP_B if do_boot else 0,
        save_test=SAVE_TEST_PREDICTIONS, save_oof=SAVE_DEV_OOF)
    for d in ready:
        filenames = common_files + [CASE_DIR / f'{safe_name(d)}.csv', CASE_DIR / f'{safe_name(d)}_control.csv']
        fingerprints[d] = manifest_signature(filenames, settings)
    _SHARED.clear()
    _SHARED.update(people=people, panel=panel, registry=registry)
    logs = list(skipped)
    for k, reply in enumerate(run_jobs(ready, fingerprints, do_match, do_boot), 1):
        logs.append(reply)
        if reply['status'] == 'FAILED':
            print('  [FAILED]', reply['disease'], reply['reason'], flush=True)
        if k % 10 == 0 or k == len(ready):
            print(f"  [Progress] {k}/{len(ready)}, failures={sum((x['status'] == 'FAILED' for x in logs))}",
                flush=True)
    atomic_tsv(pd.DataFrame(logs), OUT / '07_endpoint_runlog.tsv')
    good = [x['disease'] for x in logs if x['status'] in ('OK', 'CACHED')]
    if not good:
        raise RuntimeError('No endpoint successfully modeled; inspect worker logs')
    outputs = collect_results(good, do_match, do_boot)
    save_nested_paired_comparisons()
    if PLOT_PDF and 'test_metrics.tsv' in outputs:
        plot_holdout_pdf(outputs['test_metrics.tsv'], registry)
    readme = (
        f"Version: {VERSION}\n"
        f"Study: 630 noninfectious UKB incident outcomes, 294 proteins in three predefined sets.\n"
        f"Validation: INTERNAL only (both nested CV and frozen holdout drawn from UKB).\n"
        f"NOTE: holdout already examined in previous Version-45 reporting; not a new pristine test set.\n"
        f"The ORIGINAL Version-45 fixed global 80/20 participant split is READ-ONLY.\n"
        f"Original split used age/sex strata and NO disease-outcome optimization.\n"
        f"All four models and all endpoints reuse the identical frozen Version-45 split.\n"
        f"No rerandomization or class-, endpoint-, or AUC-based split selection.\n"
        f"Participant file: {SPLIT_FILE}\n"
        f"All 630 disease audit: {SPLIT_DIR / '01_all630_endpoint_split_audit.tsv'}\n"
        f"RUN_MODE={RUN_MODE}. Endpoint models completed: {len(good)}/{len(targets)}.\n"
        f"Protein sets: Baseline-only=74; Dual-response=75; Dynamic-only=145.\n"
        f"Covariates: Age + Sex (sex-specific endpoints: Age only). No BMI or old Score.\n"
        f"Coxnet: l1_ratio=1; age/sex penalty_factor=0; proteins penalty_factor=1.\n"
        f"Development 80%: outer 5-fold CV each with its own inner 5-fold CV for lambda selection;\n"
        f"all preprocessing fitted on training only in each fold.\n"
        f"Final models: lambda re-tuned with 5-fold CV on full Development, refitted on full Development.\n"
        f"Holdout 20%: all models evaluated using exactly same participants per endpoint.\n"
        f"Time-dependent outcome: 10-year cumulative/dynamic IPCW AUC.\n"
        f"Model fitting: 10y event status and followup capped at 10y+2d.\n"
        f"Censoring estimation for IPCW uses full original Development followup.\n"
        f"Administrative censoring: {ADMIN_CENSOR.date()}; death date file: {DEATH_DATES_FILE}.\n"
        f"Test AUC minimum 10y events: {MIN_TEST_EVENTS_10Y}; >10y controls: {MIN_TEST_CONTROLS_AFTER_10Y}.\n"
        f"Failed / excluded endpoints remain explicitly in 00_all630_endpoint_split_audit.tsv.\n"
        f"Nested CV results: 01_development_outer_fold_metrics.tsv.\n"
        f"Nested CV AUC summary: 02_development_cv_summary.tsv.\n"
        f"Matched-fold model contrasts: 16_nested_cv_paired_outer_fold_AUC.tsv.\n"
        f"Matched-fold endpoint summary: 17_nested_cv_endpoint_paired_summary.tsv.\n"
        f"Primary test results: 03_independent_holdout_four_models.tsv.\n"
        f"Complete 4-model endpoints: 13_test_complete_four_model_endpoints.tsv.\n"
        f"System summary: 14_test_12system_median_DeltaAUC.tsv.\n"
        f"Exploratory paired system tests with BH-FDR: 15_test_system_pairwise_Wilcoxon_BHFDR.tsv.\n"
        f"Comparison between classes: 04_independent_holdout_paired_deltas.tsv.\n"
        f"Protein coefficients: 05_final_model_protein_weights.tsv; lambda audit in 06.\n"
        f"Participant-level holdout predictions: per_endpoint/<disease>/test_predictions.tsv.gz.\n"
        f"Final saved model coefficients/preprocessing: per_endpoint/<disease>/*.npz.\n"
        f"Optional number-matched Dynamic sensitivity: {do_match}, repeats={MATCH_REPEATS}.\n"
        f"Optional holdout conditional bootstrap: {do_boot}, B={BOOTSTRAP_B}.\n"
        f"This is internal held-out testing, NOT external cohort validation.\n"
        f"Since existing UKB results were previously examined, cannot claim this\n"
        f"holdout sample was untouched from the beginning of the research program.\n"
        f"IMPORTANT: inspect death-censoring and baseline/end-date semantics before publication.\n"
    )
    (OUT / 'METHODS_README.txt').write_text(readme, encoding='utf-8')
    print('\n[HOLDOUT RESULTS; lambda.min]')
    if 'test_metrics.tsv' in outputs:
        m = delta_vs_age_sex(outputs['test_metrics.tsv'])
        cols = ['disease', 'model', 'AUC_10y', 'Delta_AUC_10y', 'test_events_10y']
        print(m.loc[m.lambda_rule.eq('min'), cols].head(32).to_string(index=False))
    print(f'\n[DONE] {len(good)}/{len(targets)} complete; elapsed_min={(time.perf_counter() - start) / 60:.1f}; output={OUT}',
        flush=True)
if __name__ == '__main__':
    main()
