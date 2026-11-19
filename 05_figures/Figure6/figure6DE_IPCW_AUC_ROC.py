"""Figure 6D/E: independent within-cohort holdout IPCW AUC and ROC plots."""
from __future__ import annotations
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
PROJECT = Path(os.environ.get("MULTIOMICS_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
UKB = Path(os.environ.get("MULTIOMICS_UKB_DIR", str(Path(__file__).resolve().parents[2] / "data" / "ukb")))
ROOT = PROJECT / 'ukb_prediction_analysis'
RUN_DIR = ROOT / 'independent_holdout_models' / 'all_630'
SPLIT_FILE = ROOT / 'frozen_model_split' / 'shared_split' / 'frozen_participant_split.tsv'
BASELINE_FILE = UKB / 'ukb_baseline_dates.csv'
CASE_DIR = Path(os.environ.get('UKB_CASE_DIR', str(UKB / 'incident_disease_records')))
OUTPUT_PDF = RUN_DIR / 'Figure6DE_Holdout_PAH_COPD_NAFLD_PAD_205x100mm.pdf'
ADMIN_CENSOR = pd.Timestamp('2023-11-30')
DEATH_DATES_FILE = None
HORIZON_DAYS = 10 * 365.25
LAMBDA_RULE = 'min'
ROC_AUC_VALIDATION_TOL = 0.005
ROC_DISEASES = [
    ('I9_HYPTENSPUL', 'Pulmonary arterial hypertension'),
    ('COPD_HOSPITAL_MAIN', 'COPD (hospital diagnosis)'),
    ('NAFLD', 'NAFLD'),
    ('I9_PAD', 'Peripheral artery disease'),
]
CLASSES = ('Baseline-only', 'Dual-response', 'Dynamic-only')
MODELS = ('Age+Sex',) + CLASSES
COLOR = {'Age+Sex': '#8B9399', 'Baseline-only': '#4C95BC', 'Dual-response': '#55A96E', 'Dynamic-only': '#E57C5E'}
NAVY = '#172B46'
MUTED = '#5D6C7A'
GRID = '#E8EDF3'
FIGURE_MM = (205.0, 100.0)
FIGURE_INCHES = (FIGURE_MM[0] / 25.4, FIGURE_MM[1] / 25.4)

def read_tsv(path: Path, required=()):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input file: {path}')
    df = pd.read_csv(path, sep='\t', low_memory=False, dtype={'disease': 'string', 'eid': 'string'})
    df.columns = df.columns.astype(str).str.strip()
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return df

def configure_plot():
    from matplotlib.font_manager import findfont
    try:
        findfont('Arial', fallback_to_default=False)
        font = 'Arial'
    except ValueError:
        font = 'DejaVu Sans'
    plt.rcParams.update({'font.family': font, 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8.0,
        'axes.labelsize': 8.0, 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white'})

def short_system(raw):
    """Merge Chapter VII/VIII into one Eye / ear system: exactly 12 systems."""
    s = str(raw).strip()
    t = s.lower()
    m = re.search('chapter\\s+([ivx]+)\\b', t)
    code = m.group(1).upper() if m else ''
    chapters = {'II': 'Neoplasms', 'III': 'Blood / immune', 'IV': 'Endocrine / metabolic',
        'V': 'Mental / behavioral', 'VI': 'Nervous', 'VII': 'Eye / ear', 'VIII': 'Eye / ear', 'IX': 'Cardiovascular',
        'X': 'Respiratory', 'XI': 'Digestive', 'XII': 'Skin / subcutaneous', 'XIII': 'Musculoskeletal',
        'XIV': 'Genitourinary'}
    if code in chapters:
        return chapters[code]
    for key, val in [('endocrin', 'Endocrine / metabolic'), ('metabolic', 'Endocrine / metabolic'), ('circulat',
        'Cardiovascular'), ('cardiovasc', 'Cardiovascular'), ('respirat', 'Respiratory'), ('digest', 'Digestive'),
        ('genitourin', 'Genitourinary'), ('musculoskel', 'Musculoskeletal'), ('neoplasm', 'Neoplasms'), ('mental',
        'Mental / behavioral'), ('nervous', 'Nervous'), ('blood', 'Blood / immune'), ('immune', 'Blood / immune'),
        ('skin', 'Skin / subcutaneous'), ('eye', 'Eye / ear'), ('ear', 'Eye / ear')]:
        if key in t:
            return val
    return s

def load_four_model_complete():
    path = RUN_DIR / '03_independent_holdout_four_models.tsv'
    d = read_tsv(path, ['disease', 'model', 'lambda_rule', 'AUC_10y'])
    if 'status' not in d.columns:
        raise ValueError(f'{path}: Version-46 status column is missing; verify output version.')
    d = d.loc[d['status'].astype(str).str.strip().eq('OK')].copy()
    d['disease'] = d['disease'].astype('string').str.strip()
    d['model'] = d['model'].astype(str).str.strip()
    d['lambda_rule'] = d['lambda_rule'].astype(str).str.strip()
    d = d.loc[d['model'].isin(MODELS)].copy()
    d['AUC_10y'] = pd.to_numeric(d['AUC_10y'], errors='coerce')
    d.loc[~d['AUC_10y'].between(0, 1), 'AUC_10y'] = np.nan
    protein = d.loc[d['model'].isin(CLASSES) & d['lambda_rule'].eq(LAMBDA_RULE)].copy()
    baseline = d.loc[d['model'].eq('Age+Sex')].copy()
    if baseline.empty:
        raise ValueError('Age+Sex holdout AUC rows are missing.')
    base_preferred = baseline.loc[baseline['lambda_rule'].eq(LAMBDA_RULE)]
    if not base_preferred.empty:
        baseline = base_preferred
    else:
        for disease, g in baseline.groupby('disease'):
            x = g['AUC_10y'].dropna().to_numpy(float)
            if len(x) > 1 and np.ptp(x) > 1e-08:
                raise ValueError(f'Inconsistent Age+Sex AUC across lambda rules: {disease}')
        baseline = baseline.drop_duplicates(['disease', 'model'])
    used = pd.concat([protein, baseline], ignore_index=True)
    if used.duplicated(['disease', 'model']).any():
        x = used.loc[used.duplicated(['disease', 'model'], keep=False), ['disease', 'model', 'lambda_rule']].head(12)
        raise ValueError('Duplicate disease-model rows:\n' + x.to_string(index=False))
    wide = used.pivot(index='disease', columns='model', values='AUC_10y')
    for model in MODELS:
        if model not in wide.columns:
            wide[model] = np.nan
    wide = wide.dropna(subset=list(MODELS)).copy()
    if wide.empty:
        raise ValueError('There are no diseases with four valid holdout AUCs.')
    audit_path = RUN_DIR / '00_all630_endpoint_split_audit.tsv'
    audit = read_tsv(audit_path, ['disease']) if audit_path.is_file() else pd.DataFrame()
    meta = used.drop_duplicates('disease').set_index('disease')
    if not audit.empty:
        audit = audit.drop_duplicates('disease').set_index('disease')
    for col in ['DiseaseSystem', 'DiseaseLabel']:
        if col not in meta:
            meta[col] = pd.NA
        if not audit.empty and col in audit:
            meta[col] = meta[col].fillna(audit[col])
    wide = wide.join(meta[['DiseaseSystem', 'DiseaseLabel']], how='left')
    if wide['DiseaseSystem'].isna().any():
        missing = list(wide.index[wide['DiseaseSystem'].isna()])[:10]
        raise ValueError(f'DiseaseSystem missing for: {missing}')
    wide['system'] = wide['DiseaseSystem'].map(short_system)
    for cls in CLASSES:
        wide['delta_' + cls] = wide[cls] - wide['Age+Sex']
    wide.index = wide.index.astype(str)
    wide.index.name = 'disease'
    return wide.reset_index()

def summarize_systems(wide):
    records = []
    for system, g in wide.groupby('system'):
        r = {'system': system, 'n_diseases': len(g)}
        for c in CLASSES:
            r[c] = float(g['delta_' + c].mean())
        vals = np.array([r[c] for c in CLASSES], dtype=float)
        vmax = np.max(vals)
        winners = [c for c, v in zip(CLASSES, vals) if np.isclose(v, vmax, rtol=0, atol=1e-12)]
        r['top_class'] = winners[0] if len(winners) == 1 else 'Tie'
        r['top_delta_auc'] = float(vmax)
        records.append(r)
    stats = pd.DataFrame(records).sort_values('Dynamic-only', ascending=False)
    return stats.reset_index(drop=True)

def clean_eid(series):
    return series.astype('string').str.strip().str.replace('\\.0$', '', regex=True)

def parse_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    s = series.astype('string').str.strip().str.lower()
    mapping = {'true': True, 'false': False, '1': True, '0': False}
    if not s.isin(mapping).all():
        raise ValueError(f'Invalid boolean values: {s.unique()[:8]}')
    return s.map(mapping).to_numpy(dtype=bool)

def safe_name(value):
    return re.sub('[\\/:*?"<>|]+', '_', str(value).strip())

def resolve_roc_diseases(wide):
    available = set(wide['disease'].astype(str).str.strip())
    resolved = []
    for code, title in ROC_DISEASES:
        if code not in available:
            raise ValueError(f"No four-model-complete holdout AUC for {title.replace(chr(10),
                ' ')} ({code}). Check that exact endpoint in 03_independent_holdout_four_models.tsv. Choose a replacement explicitly in ROC_DISEASES if needed.")
        resolved.append((code, title))
    return resolved

def make_training_reference(disease, split, baseline, risk_policy, death_file=None):
    """Reconstruct frozen Version-46 Development risk set for IPCW.
    SPLIT_FILE enumerates ONLY the globally eligible age/sex/Olink-coverage
    cohort, so joining case/control records to it enforces the original
    protein eligibility definition without rereading the 294-assay matrix.
    Follow-up dates and sex eligibility must match make_endpoint_data().
    """
    casefile = CASE_DIR / f'{safe_name(disease)}.csv'
    ctrlfile = CASE_DIR / f'{safe_name(disease)}_control.csv'
    case = pd.read_csv(casefile, usecols=['eid', 'event_date'], dtype={'eid': str})
    ctrl = pd.read_csv(ctrlfile, usecols=['eid'], dtype={'eid': str})
    case['eid'], ctrl['eid'] = (clean_eid(case['eid']), clean_eid(ctrl['eid']))
    case['event_date'] = pd.to_datetime(case['event_date'], errors='coerce')
    if case.eid.isna().any() or ctrl.eid.isna().any() or case.eid.duplicated().any() or ctrl.eid.duplicated().any() or len(set(case.eid) & set(ctrl.eid)):
        raise ValueError(f'{disease}: invalid case/control identifiers')
    case['event'] = True
    ctrl['event'] = False
    ctrl['event_date'] = pd.NaT
    cohort = pd.concat([case, ctrl], ignore_index=True)
    cohort = cohort.merge(split[['eid', 'dataset', 'sex']], on='eid', how='inner', validate='one_to_one')
    if risk_policy == 'female':
        cohort = cohort.loc[cohort.sex.eq(0)].copy()
    elif risk_policy == 'male':
        cohort = cohort.loc[cohort.sex.eq(1)].copy()
    elif risk_policy != 'all':
        raise ValueError(f'{disease}: unsupported risk sex policy {risk_policy!r}')
    cohort = cohort.loc[cohort.dataset.eq('Development')].copy()
    cohort = cohort.merge(baseline[['eid', 'baseline_date']], on='eid', how='inner', validate='one_to_one')
    if cohort.empty:
        raise ValueError(f'{disease}: empty Development risk set')
    if cohort.loc[cohort.event, 'event_date'].isna().any():
        raise ValueError(f'{disease}: missing incident event dates')
    if (cohort.loc[cohort.event, 'event_date'] <= cohort.loc[cohort.event, 'baseline_date']).any():
        raise ValueError(f'{disease}: incident event on/before baseline')
    if (cohort.loc[cohort.event, 'event_date'] > ADMIN_CENSOR).any():
        raise ValueError(f'{disease}: incident event after administrative censor')
    cens = pd.Series(ADMIN_CENSOR, index=cohort.index)
    if death_file is not None:
        death = pd.read_csv(death_file, usecols=['eid', 'death_date'], dtype={'eid': str})
        death['eid'] = clean_eid(death['eid'])
        death['death_date'] = pd.to_datetime(death.death_date, errors='coerce')
        if death.eid.duplicated().any():
            raise ValueError('Duplicate death date identifiers')
        cohort = cohort.merge(death, on='eid', how='left', validate='one_to_one')
        after_death = cohort.event & cohort.death_date.notna() & cohort.event_date.gt(cohort.death_date)
        if after_death.any():
            raise ValueError(f'{disease}: event after recorded death')
        cens = pd.concat([pd.Series(ADMIN_CENSOR, index=cohort.index), cohort.death_date], axis=1).min(axis=1)
    stop = pd.to_datetime(cens).copy()
    stop.loc[cohort.event] = cohort.loc[cohort.event, 'event_date']
    followup = (stop - cohort.baseline_date).dt.days.to_numpy(float)
    event = cohort.event.to_numpy(bool)
    valid = np.isfinite(followup) & (followup > 0)
    return (event[valid], followup[valid])

def read_test_predictions(disease):
    path = RUN_DIR / 'per_endpoint' / safe_name(disease) / 'test_predictions.tsv.gz'
    if not path.is_file():
        raise FileNotFoundError(f'No saved holdout predictions: {path}')
    d = pd.read_csv(path, sep='\t', compression='gzip', dtype={'eid': 'string'}, low_memory=False)
    expected = {'eid', 'model', 'lambda_rule', 'log_risk', 'event_10y', 'time_days'}
    if not expected.issubset(d.columns):
        raise ValueError(f'{path}: missing {sorted(expected - set(d.columns))}')
    d['eid'] = clean_eid(d['eid'])
    d['model'] = d['model'].astype(str).str.strip()
    d['lambda_rule'] = d['lambda_rule'].astype(str).str.strip()
    base = d.loc[d.model.eq('Age+Sex')].copy()
    preferred = base.loc[base.lambda_rule.eq(LAMBDA_RULE)]
    if not preferred.empty:
        base = preferred
    elif base.duplicated(['eid']).any():
        check = base.groupby('eid')['log_risk'].nunique(dropna=False)
        if (check > 1).any():
            raise ValueError(f'Inconsistent duplicated Age+Sex risks for {disease}')
        base = base.drop_duplicates('eid')
    if base.eid.duplicated().any():
        raise ValueError('Duplicate baseline predictions after lambda selection.')
    base = base[['eid', 'event_10y', 'time_days', 'log_risk']].rename(columns={'log_risk': 'Age+Sex'})
    for cls in CLASSES:
        sub = d.loc[d.model.eq(cls) & d.lambda_rule.eq(LAMBDA_RULE), ['eid', 'log_risk']].copy()
        if sub.eid.duplicated().any():
            raise ValueError(f'Duplicate predictions: {disease} x {cls}')
        sub = sub.rename(columns={'log_risk': cls})
        base = base.merge(sub, on='eid', how='inner', validate='one_to_one')
    if base.empty or base[list(MODELS)].isna().any().any():
        raise ValueError(f'Incomplete four-model test predictions for {disease}')
    for col in list(MODELS) + ['time_days']:
        base[col] = pd.to_numeric(base[col], errors='raise')
    if not np.isfinite(base[list(MODELS) + ['time_days']].to_numpy(float)).all():
        raise ValueError(f'Nonfinite test prediction values for {disease}')
    if (base['time_days'] <= 0).any():
        raise ValueError(f'Nonpositive test follow-up times for {disease}')
    return base

def ipcw_roc(scores, ytest, sample_weights):
    """Case IPCW weighted cumulative/dynamic ROC at fixed 10-year horizon."""
    ev = ytest['event'].astype(bool)
    time = ytest['time'].astype(float)
    cases = ev & (time <= HORIZON_DAYS)
    controls = time > HORIZON_DAYS
    include = cases | controls
    if np.sum(cases) < 10 or np.sum(controls) < 10:
        raise ValueError('Too few test cases/controls for 10-year ROC')
    case_w = np.where(cases, sample_weights, 0.0)
    if not np.isfinite(case_w[cases]).all() or case_w[cases].sum() <= 0:
        raise ValueError('Invalid IPCW case weights')
    r = np.asarray(scores, float)[include]
    a = case_w[include]
    b = np.asarray(controls[include], dtype=float)
    order = np.argsort(-r, kind='mergesort')
    r, a, b = (r[order], a[order], b[order])
    end = np.r_[np.flatnonzero(np.diff(r) != 0), len(r) - 1]
    tpr = np.r_[0.0, np.cumsum(a)[end] / a.sum(), 1.0]
    fpr = np.r_[0.0, np.cumsum(b)[end] / b.sum(), 1.0]
    return (fpr, tpr, float(np.trapezoid(tpr, fpr)))

def prepare_verified_rocs(wide):
    """Recompute 10-year IPCW ROC, validating exact source and risk-set sizes."""
    from sksurv.util import Surv
    from sksurv.nonparametric import CensoringDistributionEstimator
    from sksurv.metrics import cumulative_dynamic_auc
    resolved = resolve_roc_diseases(wide)
    split = read_tsv(SPLIT_FILE, ['eid', 'dataset', 'sex'])
    split['eid'] = clean_eid(split['eid'])
    split['sex'] = pd.to_numeric(split['sex'], errors='raise')
    if split.eid.duplicated().any() or not split.dataset.isin(['Development', 'Test']).all():
        raise ValueError('Invalid or duplicated frozen 80/20 split')
    audit = read_tsv(RUN_DIR / '00_all630_endpoint_split_audit.tsv', ['disease', 'risk_sex', 'N_development',
        'N_test', 'events10_development', 'events10_test', 'controls_after10_test'])
    audit = audit.drop_duplicates('disease').set_index('disease')
    baseline = pd.read_csv(BASELINE_FILE, usecols=['eid', 'date'], dtype={'eid': str})
    baseline['eid'] = clean_eid(baseline['eid'])
    baseline['baseline_date'] = pd.to_datetime(baseline['date'], errors='coerce')
    baseline = baseline.dropna(subset=['baseline_date'])
    if baseline.eid.duplicated().any():
        raise ValueError('Duplicate baseline dates')
    official = wide.set_index('disease')
    results = []
    for disease, label in resolved:
        if disease not in audit.index:
            raise ValueError(f'{disease}: missing from Version-46 endpoint audit')
        row = audit.loc[disease]
        pred = read_test_predictions(disease)
        event_dev, time_dev = make_training_reference(disease, split, baseline, str(row.risk_sex), DEATH_DATES_FILE)
        ytrain = Surv.from_arrays(event=event_dev, time=time_dev)
        ytest = Surv.from_arrays(event=parse_bool(pred.event_10y), time=pred.time_days.to_numpy(float))
        checks = {'N_development': (len(ytrain), int(row.N_development)),
            'events10_development': (int(np.sum(ytrain['event'] & (ytrain['time'] <= HORIZON_DAYS))),
            int(row.events10_development)), 'N_test': (len(ytest), int(row.N_test)),
            'events10_test': (int(np.sum(ytest['event'])), int(row.events10_test)),
            'controls_after10_test': (int(np.sum(ytest['time'] > HORIZON_DAYS)), int(row.controls_after10_test))}
        for name, (actual, expected) in checks.items():
            if actual != expected:
                raise RuntimeError(f'{disease}: {name} mismatch; reconstructed={actual}, 46 audit={expected}. Check split, censoring, and source case files.')
        print(f'[RISK SET CHECK] {disease}: development N={len(ytrain):,}, events<=10y={int(row.events10_development):,}; test N={len(ytest):,}, events<=10y={int(row.events10_test):,}',
            flush=True)
        cens = CensoringDistributionEstimator().fit(ytrain)
        weights = cens.predict_ipcw(ytest)
        curves = {}
        for model in MODELS:
            risk = pred[model].to_numpy(float)
            measured = float(cumulative_dynamic_auc(ytrain, ytest, risk, np.array([HORIZON_DAYS]))[0][0])
            fpr, tpr, integrated = ipcw_roc(risk, ytest, weights)
            reported = float(official.loc[disease, model])
            difference = max(abs(measured - reported), abs(integrated - reported))
            print(f'[ROC VALIDATION] {disease:18s} {model:15s} official={reported:.4f} rebuilt={measured:.4f} ROC_area={integrated:.4f} gap={difference:.4g}',
                flush=True)
            if difference > ROC_AUC_VALIDATION_TOL:
                raise RuntimeError(f'ROC AUC mismatch: {disease} [{model}] official={reported:.4f}, rebuilt={measured:.4f}, ROC integral={integrated:.4f}. This may indicate a split/risk-set/censoring mismatch. No incorrect PDF will be saved.')
            curves[model] = {'fpr': fpr, 'tpr': tpr, 'auc': reported}
        results.append({'disease': disease, 'title': label, 'curves': curves})
    return results

def neat_axes(ax, ylabel=False, xlabel=False):
    """Square ROC axes in a 205 x 100 mm figure; >=7-point text."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#AEBAC8')
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(axis='both', labelsize=7.0, width=0.6, length=2.4, pad=1.7, colors='black')
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis='both', color='#E9EEF2', lw=0.55, zorder=0)
    if ylabel:
        ax.set_ylabel('True positive rate', fontsize=7.2, labelpad=1.6)
    else:
        ax.tick_params(labelleft=False)
    if xlabel:
        ax.set_xlabel('False positive rate', fontsize=7.2, labelpad=1.7)

def plot_heatmap(fig, stats):
    """Panel D. Direct class winner labeling; no stars, no inferential claims."""
    fig.text(0.022, 0.961, 'D', fontsize=13.8, weight='bold', color='black', va='top')
    fig.text(0.054, 0.958, '10-year AUC gains across disease systems', fontsize=9.4, weight='bold', color=NAVY,
        va='top')
    fig.text(0.024, 0.88, 'Mean 10-year test delta AUC (protein model - Age + Sex)', fontsize=7.2, color=MUTED,
        va='top')
    k = len(stats)
    if k != 12:
        raise ValueError(f'Expected exactly 12 systems, got {k}: {stats.system.tolist()}')
    bottom, height = (0.166, 0.62)
    left_label, label_end = (0.021, 0.214)
    heat_left, heat_w = (0.218, 0.15)
    top_left, top_w = (0.386, 0.094)
    row_h = height / k
    palette = mcolors.LinearSegmentedColormap.from_list('warm_delta_auc', ['#F8F8F8', '#F6E3DA', '#F0B8A1',
        '#E18469', '#BF4937'])
    arr = stats[list(CLASSES)].to_numpy(float)
    if not np.isfinite(arr).all():
        raise ValueError('Non-finite system-mean delta AUC')
    vmax_heat = max(0.1, np.ceil(max(0.0, float(arr.max())) * 100) / 100)
    norm = mcolors.Normalize(vmin=min(0.0, float(arr.min())), vmax=vmax_heat, clip=True)
    for i in range(k):
        y0 = bottom + height - (i + 1) * row_h
        fill = '#F1F4F7' if i % 2 == 0 else '#FBFCFD'
        fig.add_artist(Rectangle((left_label, y0 + row_h * 0.055), label_end - left_label, row_h * 0.89,
            transform=fig.transFigure, facecolor=fill, edgecolor='none', zorder=0))
        fig.text(left_label + 0.003, y0 + row_h * 0.5, str(stats.iloc[i]['system']), ha='left', va='center',
            color='#263342', fontsize=7.35)
    ax = fig.add_axes([heat_left, bottom, heat_w, height])
    ax.set_xlim(0, 3)
    ax.set_ylim(0, k)
    ax.invert_yaxis()
    ax.set_axis_off()
    for i in range(k):
        row = stats.iloc[i]
        winner = row['top_class']
        for j, cls in enumerate(CLASSES):
            is_top = cls == winner
            ax.add_patch(Rectangle((j, i), 1, 1, linewidth=0.85 if is_top else 1.05,
                edgecolor='#64798B' if is_top else 'white', facecolor=palette(norm(float(arr[i, j]))), zorder=4 if is_top else 3))
    ax.add_patch(Rectangle((0, 0), 3, k, fill=False, linewidth=0.75, edgecolor='#D6DDE5'))
    for j, (name, col) in enumerate([('Baseline', COLOR['Baseline-only']), ('Dual', COLOR['Dual-response']),
        ('Dynamic', COLOR['Dynamic-only'])]):
        x = heat_left + heat_w * (j + 0.5) / 3
        fig.text(x, 0.798, name, ha='center', va='bottom', fontsize=7.1, weight='bold', color=col)
    fig.text(top_left + top_w / 2, 0.798, 'Top class', ha='center', va='bottom', fontsize=7.2, weight='bold',
        color=NAVY)
    for i, r in stats.iterrows():
        y = bottom + height - (i + 0.5) * row_h
        cls = r['top_class']
        label = {'Baseline-only': 'Baseline', 'Dual-response': 'Dual', 'Dynamic-only': 'Dynamic', 'Tie': 'Tie'}[cls]
        col = COLOR.get(cls, '#7E8995')
        fig.add_artist(plt.Line2D([top_left + 0.009], [y], marker='o', markersize=4.7, color=col,
            markeredgewidth=0, ls='None', transform=fig.transFigure, zorder=5))
        fig.text(top_left + 0.02, y, label, ha='left', va='center', fontsize=7.25, weight='medium', color=col)
    fig.text(top_left + top_w / 2, 0.127, 'Highest mean ΔAUC', ha='center', va='top', fontsize=6.9, color=MUTED)
    cbax = fig.add_axes([0.085, 0.068, 0.335, 0.018])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=palette), cax=cbax, orientation='horizontal')
    color_ticks = [0, 0.05, 0.1]
    if vmax_heat > 0.1 + 1e-09:
        color_ticks.append(vmax_heat)
    cbar.set_ticks(color_ticks)
    cbar.set_ticklabels(['0'] + [f'+{v:.2f}' for v in color_ticks[1:]])
    cbar.ax.tick_params(length=0, labelsize=7.0, pad=2)
    cbar.outline.set_visible(False)
    fig.text(0.2525, 0.019, '10-year ΔAUC', ha='center', fontsize=7.8, color='black')

def annotate_auc_table(ax, row):
    """Compact in-panel AUC key; line colors correspond to the shared legend."""
    values = {m: float(row['curves'][m]['auc']) for m in MODELS}
    ax.text(0.669, 0.338, '10-y AUC', transform=ax.transAxes, color='#32485E', fontsize=7.0, ha='left',
        va='bottom', bbox={'boxstyle': 'square,pad=.05', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9})
    for j, model in enumerate(MODELS):
        y = 0.28 - j * 0.064
        ax.plot([0.636, 0.677], [y, y], transform=ax.transAxes, color=COLOR[model], lw=1.9, solid_capstyle='butt',
            zorder=8, clip_on=False)
        ax.text(0.985, y, f'{values[model]:.3f}', transform=ax.transAxes, color=NAVY, ha='right', va='center',
            fontsize=7.0, bbox={'boxstyle': 'square,pad=.005', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.92},
            zorder=8)

def plot_rocs(fig, roc_rows):
    """Panel E. Four square ROC plots in a compact, two-row layout."""
    if len(roc_rows) != 4:
        raise ValueError(f'Expected four ROC endpoints, got {len(roc_rows)}')
    fig.text(0.505, 0.961, 'E', fontsize=13.8, weight='bold', color='black', va='top')
    fig.text(0.535, 0.958, '10-year time-dependent ROC curves', fontsize=8.7, weight='bold', color=NAVY, va='top')
    handles = [Line2D([0], [0], color=COLOR[m], lw=2.1, label='Age + Sex' if m == 'Age+Sex' else m) for m in MODELS]
    fig.legend(handles=handles, ncol=4, loc='center', frameon=False, bbox_to_anchor=(0.752, 0.87), fontsize=7.0,
        columnspacing=0.75, handlelength=1.25, handletextpad=0.28, borderaxespad=0)
    panels = [(0.528, 0.5, 0.197, 0.29), (0.771, 0.5, 0.197, 0.29), (0.528, 0.13, 0.197, 0.29), (0.771, 0.13,
        0.197, 0.29)]
    for i, (row, dims) in enumerate(zip(roc_rows, panels)):
        ax = fig.add_axes(dims)
        neat_axes(ax, ylabel=i in (0, 2), xlabel=i in (2, 3))
        ax.plot([0, 1], [0, 1], ls='--', lw=0.75, color='#C3CFD9', dashes=(3, 3), zorder=1)
        for model in MODELS:
            curve = row['curves'][model]
            ax.plot(curve['fpr'], curve['tpr'], lw=1.55, color=COLOR[model], zorder=3)
        name = row['title']
        title_size = 7.1 if len(name) >= 25 else 8.1
        title_x = dims[0] + dims[2] / 2
        title_y = dims[1] + dims[3] + 0.009
        fig.text(title_x, title_y, name, color='black', weight='bold', fontsize=title_size, ha='center', va='bottom')
        annotate_auc_table(ax, row)

def main():
    configure_plot()
    if 'independent_holdout_models' not in str(RUN_DIR):
        raise RuntimeError('RUN_DIR must point to the frozen independent-holdout model results')
    wide = load_four_model_complete()
    system_summary = summarize_systems(wide)
    n_better = int((wide['Dynamic-only'] > wide[['Baseline-only', 'Dual-response']].max(axis=1)).sum())
    print(f'[HOLDOUT] Four-model-complete outcomes: {len(wide)}; systems: {len(system_summary)}')
    print(f'[DESCRIPTIVE] Dynamic-only highest: {n_better}/{len(wide)}')
    print('[SYSTEM MEANS] Mean 10-year holdout Delta AUC by disease system:')
    print(system_summary[['system', 'n_diseases', 'Baseline-only', 'Dual-response', 'Dynamic-only',
        'top_class']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('[SYSTEM TOP CLASS COUNTS]')
    print(system_summary['top_class'].value_counts().to_string())
    print('[ROC] Reconstruct and validate ROC / AUC before drawing any PDF.')
    verified = prepare_verified_rocs(wide)
    fig = plt.figure(figsize=FIGURE_INCHES, facecolor='white')
    plot_heatmap(fig, system_summary)
    plot_rocs(fig, verified)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    tmp_pdf = OUTPUT_PDF.with_suffix('.pdf.tmp')
    try:
        fig.savefig(tmp_pdf, format='pdf', facecolor='white')
        tmp_pdf.replace(OUTPUT_PDF)
    finally:
        plt.close(fig)
        if tmp_pdf.exists():
            tmp_pdf.unlink()
    print('\n[PDF OUTPUT]', OUTPUT_PDF)
    print('[DONE] Exactly one PDF. No TSV, PNG or other outputs.')
if __name__ == '__main__':
    main()
