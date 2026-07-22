"""
Middle-model sweep for the Cortex-M4 (SiWG917 / BRD2605A) deployment target.

The FPGA edge model (4 features, depth 4) is capped by combinational gate count.
The M4 has an FPU, ~320 KB SRAM and 8 MB flash, so it can afford a bigger model.
This sweeps feature-count x depth to find the "knee" between the 4-feature edge
model (ROC-AUC ~0.674) and the 40-feature server model (ROC-AUC ~0.726).

Uses the exact feature engineering, splits, and scale_pos_weight convention from
01_sepsis_eda_modeling_clean.ipynb so results are directly comparable.
Reads data/Dataset.csv. Writes hw/middle_sweep_results.csv.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

# ---- device: use CUDA if available, else CPU (results are ~identical for hist) ----
def pick_device():
    try:
        d = xgb.DMatrix(np.zeros((4, 1)), label=np.array([0, 1, 0, 1]))
        xgb.train({'tree_method': 'hist', 'device': 'cuda', 'verbosity': 0}, d, num_boost_round=1)
        return 'cuda'
    except Exception:
        return 'cpu'

DEVICE = pick_device()
print(f"Device: {DEVICE}")

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

# ---- load + feature engineering (identical to the notebook) ----
print("Loading data/Dataset.csv ...")
raw = pd.read_csv('data/Dataset.csv')
df = raw.rename(columns={'Patient_ID': 'patient_id'}).copy()

grp = df.groupby('patient_id', sort=False)
for v in VITALS:
    df[f'{v}_mean3'] = grp[v].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{v}_mean6'] = grp[v].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    df[f'{v}_delta3'] = grp[v].diff(3)
    df[f'{v}_missing'] = df[v].isna().astype('int8')

FEATURES_FULL = (VITALS + [f'{v}_mean3' for v in VITALS] + [f'{v}_mean6' for v in VITALS]
                 + [f'{v}_delta3' for v in VITALS] + [f'{v}_missing' for v in VITALS])
FEATURES_EDGE = ['Temp_mean3', 'Resp_mean3', 'HR_mean3', 'SBP_mean3']

# ---- candidate feature tiers (the "in between" ladder) ----
TIERS = {
    'F4_edge':   FEATURES_EDGE,
    'F8_mean3':  [f'{v}_mean3' for v in VITALS],
    'F16_m3d3':  [f'{v}_mean3' for v in VITALS] + [f'{v}_delta3' for v in VITALS],
    'F24_m3d3m6':[f'{v}_mean3' for v in VITALS] + [f'{v}_delta3' for v in VITALS]
                 + [f'{v}_mean6' for v in VITALS],
    'F40_full':  FEATURES_FULL,
}

# ---- splits: identical seeds to the notebook ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, groups=df['patient_id']))
train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
i_tr, i_val = next(inner.split(train_df, groups=train_df['patient_id']))
tr_sub, val_sub = train_df.iloc[i_tr], train_df.iloc[i_val]
assert not (set(train_df.patient_id) & set(test_df.patient_id))
assert not (set(tr_sub.patient_id) & set(val_sub.patient_id))
spw = (tr_sub['SepsisLabel'] == 0).sum() / (tr_sub['SepsisLabel'] == 1).sum()
y_test = test_df['SepsisLabel'].values
print(f"train {train_df.shape[0]:,} | test {test_df.shape[0]:,} | spw {spw:.1f}")

DEPTHS = [4, 5, 6]
M4_MHZ = 180.0
CYCLES_PER_NODE = 10  # rough: load feature, compare, branch on Cortex-M4

def model_bytes(total_nodes):
    # compact quantized C layout: feat idx (1B) + threshold (2B) + 2 children (2B each)
    # + leaf (2B); ~9B/node upper bound. Rounds to a realistic flash figure.
    return total_nodes * 9

def train_eval(feats, depth):
    dtr  = xgb.DMatrix(tr_sub[feats].values,  label=tr_sub['SepsisLabel'].values,  feature_names=feats)
    dval = xgb.DMatrix(val_sub[feats].values, label=val_sub['SepsisLabel'].values, feature_names=feats)
    dtest= xgb.DMatrix(test_df[feats].values, label=y_test,                        feature_names=feats)
    params = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': DEVICE,
              'eval_metric': 'auc', 'scale_pos_weight': spw, 'seed': 42,
              'max_depth': depth, 'learning_rate': 0.1,
              'subsample': 0.8, 'colsample_bytree': 0.9}
    bst = xgb.train(params, dtr, num_boost_round=400,
                    evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
    n_trees = bst.best_iteration + 1
    prob = bst.predict(dtest, iteration_range=(0, n_trees))
    roc = roc_auc_score(y_test, prob)
    pr  = average_precision_score(y_test, prob)
    tdf = bst.trees_to_dataframe()
    tdf = tdf[tdf['Tree'] < n_trees]
    total_nodes = len(tdf)
    kb = model_bytes(total_nodes) / 1024
    lat_us = (n_trees * depth * CYCLES_PER_NODE) / M4_MHZ
    return dict(n_trees=n_trees, roc=roc, pr=pr, nodes=total_nodes, kb=kb, lat_us=lat_us)

rows = []
print(f"\n{'tier':>11} {'nfeat':>5} {'depth':>5} {'trees':>5} {'ROC':>7} {'PR':>7} "
      f"{'nodes':>7} {'flashKB':>8} {'lat_us':>7}")
for name, feats in TIERS.items():
    for d in DEPTHS:
        r = train_eval(feats, d)
        rows.append(dict(tier=name, nfeat=len(feats), depth=d, **r))
        print(f"{name:>11} {len(feats):>5} {d:>5} {r['n_trees']:>5} {r['roc']:>7.4f} "
              f"{r['pr']:>7.4f} {r['nodes']:>7} {r['kb']:>8.1f} {r['lat_us']:>7.1f}")

res = pd.DataFrame(rows)
res.to_csv('hw/middle_sweep_results.csv', index=False)

# ---- knee analysis: gap recovered vs the 4-feature edge anchor ----
edge = res[res.tier == 'F4_edge'].roc.max()
full = res[res.tier == 'F40_full'].roc.max()
gap = full - edge
print(f"\nEdge anchor ROC {edge:.4f} | Server anchor ROC {full:.4f} | gap {gap:.4f}")
res['gap_recovered'] = (res.roc - edge) / gap
mid = res[~res.tier.isin(['F4_edge', 'F40_full'])].copy()
mid = mid.sort_values('roc', ascending=False)
print("\nMiddle candidates by ROC (gap_recovered = fraction of edge->server lift):")
print(mid[['tier', 'depth', 'n_trees', 'roc', 'pr', 'kb', 'lat_us', 'gap_recovered']]
      .to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print("\nWrote hw/middle_sweep_results.csv")
