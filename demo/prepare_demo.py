"""
Bedside monitor demo data preparation.
Loads the server model, identifies a suitable test patient who experiences early
sepsis onset, calculates SHAP drivers, and exports demo assets.
"""
import numpy as np, pandas as pd, xgboost as xgb, joblib, json
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
DISPLAY = ['HR', 'Temp', 'Resp', 'SBP', 'MAP', 'O2Sat']     # what the monitor charts
meta_feats = json.load(open('models/sepsis_meta.json'))['features']

raw = pd.read_csv('data/Dataset.csv').rename(columns={'Patient_ID': 'patient_id'})
df = raw[['patient_id'] + VITALS + ['SepsisLabel', 'ICULOS']].copy()
g = df.groupby('patient_id', sort=False)
for v in VITALS:
    df[f'{v}_mean3'] = g[v].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{v}_mean6'] = g[v].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{v}_delta3'] = g[v].diff(3)
    df[f'{v}_missing'] = df[v].isna().astype('int8')

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, groups=df['patient_id']))
test_df = df.iloc[te_idx].copy()

bst = xgb.Booster(); bst.load_model('models/sepsis_booster.json')
iso = joblib.load('models/sepsis_isotonic.joblib')
nb = json.load(open('models/sepsis_meta.json')).get('best_iteration', None)

X = test_df[meta_feats].values
test_df['risk_raw'] = bst.predict(xgb.DMatrix(X, feature_names=meta_feats))
test_df['risk'] = iso.predict(test_df['risk_raw'].values)
print(f'Server test ROC-AUC: {roc_auc_score(test_df.SepsisLabel, test_df.risk_raw):.4f}')
print(f'Calibrated risk: median {np.median(test_df.risk):.3f}, 90th {np.quantile(test_df.risk,0.9):.3f}, 99th {np.quantile(test_df.risk,0.99):.3f}')

THRESH = float(round(np.quantile(test_df['risk'], 0.92), 4))   # "elevated" alarm level
print(f'Alarm threshold (calibrated risk, 92nd pct): {THRESH}')

# pick a sepsis patient: good vital coverage, calm early, risk crosses near/before first label, clear rise
best = None
for pid, p in test_df.groupby('patient_id'):
    p = p.sort_values('ICULOS')
    if p['SepsisLabel'].max() == 0 or not (16 <= len(p) <= 60):
        continue
    cover = np.mean([1 - p[v].isna().mean() for v in ['HR', 'SBP', 'Resp']])  # display-vital coverage
    if cover < 0.7:
        continue
    flp = int(np.argmax(p['SepsisLabel'].values == 1))           # first warning-label hour
    alarms = p['risk'].values >= THRESH
    if not alarms.any():
        continue
    fap = int(np.argmax(alarms))
    lead = flp - fap
    quiet_early = p['risk'].values[:max(1, flp - 4)].max() < THRESH
    if 0 <= lead <= 12 and quiet_early:
        rise = float(p['risk'].values.max() - np.nanmean(p['risk'].values[:4]))
        if best is None or rise > best[0]:
            best = (rise, lead, len(p), pid, flp, fap, cover)

if best is None:
    raise SystemExit('No clean demo patient — loosen criteria.')
rise, lead, n, pid, flp, fap, cover = best
print(f'Chosen {pid}: {n}h, cover {cover:.0%}, warning-label h{flp}, alarm h{fap}, lead {lead}h, rise {rise:.2f}')

pat = test_df[test_df.patient_id == pid].sort_values('ICULOS').reset_index(drop=True)
pat['hour'] = range(len(pat))

# SHAP-driven flags: per-vital contribution = sum of SHAP over that vital's derived features.
# XGBoost computes TreeSHAP natively (pred_contribs) -- no external shap library needed.
contribs = bst.predict(xgb.DMatrix(pat[meta_feats].values, feature_names=meta_feats), pred_contribs=True)
sv = np.asarray(contribs)[:, :-1]                       # (hours, 40); drop bias column
for v in DISPLAY:
    idx = [meta_feats.index(f) for f in meta_feats if f == v or f.startswith(v + '_')]
    pat[f'shap_{v}'] = sv[:, idx].sum(axis=1)
print('Added per-vital SHAP driver columns:', [f'shap_{v}' for v in DISPLAY])

out_cols = ['hour'] + DISPLAY + ['risk_raw', 'risk', 'SepsisLabel'] + [f'shap_{v}' for v in DISPLAY]
pat[out_cols].to_csv('demo/demo_patient.csv', index=False)

meta = {'patient_id': str(pid), 'n_hours': int(n), 'alarm_threshold': THRESH,
        'first_warning_hour': int(flp), 'clinical_onset_hour': int(flp + 6),
        'first_alarm_hour': int(fap), 'lead_hours': int(lead),
        'server_auc': float(roc_auc_score(test_df.SepsisLabel, test_df.risk_raw))}
json.dump(meta, open('demo/demo_meta.json', 'w'), indent=2)
print('Wrote demo/demo_patient.csv + demo/demo_meta.json')
print(json.dumps(meta, indent=2))
