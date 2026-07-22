"""
Generates the on-device bedside-monitor demo data for the BRD2605A (SiWG917).

Mirrors the Vivado patient-streaming testbench (hw/tb_demo_patients.v) and the Dash
dashboard (demo/app.py), but streams REAL held-out test patients through the flagship
model on physical silicon:

  - the septic demo patient from demo/demo_meta.json (same one the dashboard replays)
  - a non-septic control patient (alarm must stay quiet)

Also embeds the isotonic calibration (models/sepsis_isotonic.joblib) as a compact
piecewise-linear table so the device prints the same CALIBRATED risk the dashboard
shows, and the alarm uses the same threshold.

Emits: hw/mcu/patient_stream.h
Run from repo root:  python hw/generate_patient_demo.py
"""
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
FEATURES = (VITALS + [f'{v}_mean3' for v in VITALS] + [f'{v}_mean6' for v in VITALS]
            + [f'{v}_delta3' for v in VITALS] + [f'{v}_missing' for v in VITALS])
DISPLAY = ['HR', 'Temp', 'Resp', 'SBP']   # what the serial monitor prints per hour
ISO_TABLE_MAX = 128

meta = json.load(open('demo/demo_meta.json'))
SEPTIC_PID = int(meta['patient_id'])
THRESH = float(meta['alarm_threshold'])

# ---- rebuild features + test split (same seeds as everywhere else) ----
print('Loading data/Dataset.csv ...')
raw = pd.read_csv('data/Dataset.csv')
df = raw.rename(columns={'Patient_ID': 'patient_id'}).copy()
grp = df.groupby('patient_id', sort=False)
for v in VITALS:
    df[f'{v}_mean3'] = grp[v].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{v}_mean6'] = grp[v].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{v}_delta3'] = grp[v].diff(3)
    df[f'{v}_missing'] = df[v].isna().astype('int8')
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, te_idx = next(gss.split(df, groups=df['patient_id']))
test_df = df.iloc[te_idx].copy()

# ---- model + isotonic calibration ----
bst = xgb.Booster(); bst.load_model('models/sepsis_booster.json')
n_trees = int(bst.attributes()['best_iteration']) + 1
iso = joblib.load('models/sepsis_isotonic.joblib')

X = test_df[FEATURES].values
test_df['risk_raw'] = bst.predict(xgb.DMatrix(X, feature_names=FEATURES),
                                  iteration_range=(0, n_trees))
test_df['risk'] = iso.predict(test_df['risk_raw'].values)

# ---- isotonic as a compact piecewise-linear table ----
xt = np.asarray(iso.X_thresholds_, dtype=np.float64)
yt = np.asarray(iso.y_thresholds_, dtype=np.float64)
if len(xt) > ISO_TABLE_MAX:
    idx = np.unique(np.round(np.linspace(0, len(xt) - 1, ISO_TABLE_MAX)).astype(int))
    xt_s, yt_s = xt[idx], yt[idx]
else:
    xt_s, yt_s = xt, yt
approx = np.interp(test_df['risk_raw'].values, xt_s, yt_s)
iso_err = np.max(np.abs(approx - test_df['risk'].values))
print(f'Isotonic table: {len(xt)} -> {len(xt_s)} points | max abs err on test set = {iso_err:.2e}')
assert iso_err < 1e-3, 'isotonic table too coarse'

# ---- pick a non-septic control patient (alarm must stay quiet) ----
best = None
for pid, p in test_df.groupby('patient_id'):
    p = p.sort_values('ICULOS')
    if p['SepsisLabel'].max() != 0 or not (16 <= len(p) <= 30):
        continue
    cover = np.mean([1 - p[v].isna().mean() for v in ['HR', 'SBP', 'Resp']])
    if cover < 0.8 or p['risk'].max() >= THRESH * 0.8:
        continue
    if best is None or cover > best[0]:
        best = (cover, pid)
assert best is not None, 'no clean control patient found'
CONTROL_PID = best[1]
print(f'Control patient: {CONTROL_PID} (coverage {best[0]:.0%})')

def cfloat(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NAN'
    s = f'{x:.9g}'
    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'
    return s + 'f'

def emit_patient(name, pid):
    p = test_df[test_df.patient_id == pid].sort_values('ICULOS').reset_index(drop=True)
    Xp = p[FEATURES].values.astype(np.float64)
    risk_ref = p['risk'].values
    labels = p['SepsisLabel'].values.astype(int)
    lines = []
    lines.append(f'#define {name}_N_HOURS {len(p)}')
    lines.append(f'static const float {name}_X[{len(p)}][40] = {{')
    for row in Xp:
        lines.append('    {' + ', '.join(cfloat(v) for v in row) + '},')
    lines.append('};')
    lines.append(f'static const float {name}_DISPLAY[{len(p)}][4] = {{  /* HR, Temp, Resp, SBP */')
    for _, r in p.iterrows():
        lines.append('    {' + ', '.join(cfloat(r[v] if pd.notna(r[v]) else float("nan")) for v in DISPLAY) + '},')
    lines.append('};')
    lines.append(f'static const unsigned char {name}_LABEL[{len(p)}] = {{'
                 + ', '.join(str(l) for l in labels) + '};')
    lines.append(f'static const float {name}_RISK_REF[{len(p)}] = {{  /* host calibrated risk */')
    for i in range(0, len(risk_ref), 8):
        lines.append('    ' + ', '.join(cfloat(v) for v in risk_ref[i:i+8]) + ',')
    lines.append('};')
    return lines, len(p)

hdr = []
hdr.append('/* Auto-generated by hw/generate_patient_demo.py -- do not edit by hand.')
hdr.append(' * Real held-out PhysioNet-2019 test patients for the on-device bedside demo. */')
hdr.append('#ifndef SEPSIS_PATIENT_STREAM_H')
hdr.append('#define SEPSIS_PATIENT_STREAM_H')
hdr.append('#include <math.h>')
hdr.append('')
hdr.append(f'#define SEPSIS_ALARM_THRESH {cfloat(THRESH)}   /* calibrated risk, 92nd pct */')
hdr.append(f'#define SEPTIC_FIRST_WARNING_HOUR {meta["first_warning_hour"]}')
hdr.append(f'#define SEPTIC_CLINICAL_ONSET_HOUR {meta["clinical_onset_hour"]}')
hdr.append('')
hdr.append(f'/* isotonic calibration: risk = interp(sigmoid(margin)) over this table */')
hdr.append(f'#define ISO_N {len(xt_s)}')
hdr.append('static const float ISO_X[ISO_N] = {')
for i in range(0, len(xt_s), 8):
    hdr.append('    ' + ', '.join(cfloat(v) for v in xt_s[i:i+8]) + ',')
hdr.append('};')
hdr.append('static const float ISO_Y[ISO_N] = {')
for i in range(0, len(yt_s), 8):
    hdr.append('    ' + ', '.join(cfloat(v) for v in yt_s[i:i+8]) + ',')
hdr.append('};')
hdr.append('')
sep_lines, n_sep = emit_patient('SEPTIC', SEPTIC_PID)
ctl_lines, n_ctl = emit_patient('CONTROL', CONTROL_PID)
hdr += sep_lines + [''] + ctl_lines
hdr.append('')
hdr.append('#endif /* SEPSIS_PATIENT_STREAM_H */')

with open('hw/mcu/patient_stream.h', 'w') as f:
    f.write('\n'.join(hdr))
print(f'Wrote hw/mcu/patient_stream.h  (septic {SEPTIC_PID}: {n_sep}h, control {CONTROL_PID}: {n_ctl}h)')
