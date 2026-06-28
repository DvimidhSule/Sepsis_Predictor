"""
Quantizes the edge model features, thresholds, and leaves to fixed-point integers
and evaluates how different bit-widths affect ROC-AUC performance on the test set.
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

EDGE = ['Temp_mean3', 'Resp_mean3', 'HR_mean3', 'SBP_mean3']

# ---- rebuild data + the exact edge model (seeded, deterministic) ----
raw = pd.read_csv('data/Dataset.csv').rename(columns={'Patient_ID': 'patient_id'})
df = raw[['patient_id', 'HR', 'Temp', 'Resp', 'SBP', 'SepsisLabel']].copy()
g = df.groupby('patient_id', sort=False)
for v in ['Temp', 'Resp', 'HR', 'SBP']:
    df[f'{v}_mean3'] = g[v].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, groups=df['patient_id']))
train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
i_tr, i_val = next(gss2.split(train_df, groups=train_df['patient_id']))
tr_sub, val_sub = train_df.iloc[i_tr], train_df.iloc[i_val]

spw = (tr_sub['SepsisLabel'] == 0).sum() / (tr_sub['SepsisLabel'] == 1).sum()
dtr = xgb.DMatrix(tr_sub[EDGE].values, label=tr_sub['SepsisLabel'].values, feature_names=EDGE)
dval = xgb.DMatrix(val_sub[EDGE].values, label=val_sub['SepsisLabel'].values, feature_names=EDGE)
params = {'objective': 'binary:logistic', 'tree_method': 'hist', 'device': 'cuda', 'eval_metric': 'auc',
          'scale_pos_weight': spw, 'max_depth': 4, 'learning_rate': 0.1, 'subsample': 0.8,
          'colsample_bytree': 1.0, 'seed': 42}
bst = xgb.train(params, dtr, num_boost_round=60, evals=[(dval, 'val')],
                early_stopping_rounds=15, verbose_eval=False)
n_trees = bst.best_iteration + 1
print(f'Edge model rebuilt: {n_trees} trees, depth 4')

X_test = test_df[EDGE].values.astype(np.float64)
y_test = test_df['SepsisLabel'].values
auc_xgb = roc_auc_score(y_test, bst.predict(xgb.DMatrix(X_test, feature_names=EDGE),
                                            iteration_range=(0, n_trees)))
print(f'XGBoost float ROC-AUC (reference): {auc_xgb:.4f}')

# ---- parse trees into plain dicts ----
tdf = bst.trees_to_dataframe()
tdf = tdf[tdf['Tree'] < n_trees]
feat_idx = {f: i for i, f in enumerate(EDGE)}
trees = []
for t, sub in tdf.groupby('Tree'):
    nodes = {}
    for _, r in sub.iterrows():
        nid = int(r['Node'])
        if r['Feature'] == 'Leaf':
            nodes[nid] = ('leaf', float(r['Gain']))
        else:
            yes = int(r['Yes'].split('-')[1]); no = int(r['No'].split('-')[1]); mis = int(r['Missing'].split('-')[1])
            nodes[nid] = ('split', feat_idx[r['Feature']], float(r['Split']), yes, no, mis)
    trees.append(nodes)

def margin_float(x):
    s = 0.0
    for nodes in trees:
        nid = 0
        while nodes[nid][0] == 'split':
            _, fi, thr, yes, no, mis = nodes[nid]
            v = x[fi]
            nid = mis if np.isnan(v) else (yes if v < thr else no)
        s += nodes[nid][1]
    return s

# sanity: my float replica must reproduce the XGBoost ranking exactly
m_float = np.array([margin_float(x) for x in X_test])
print(f'Pure-Python float replica ROC-AUC: {roc_auc_score(y_test, m_float):.4f}  (must match reference)')

# ---- collect ranges to size fixed-point ----
all_thr = [n[2] for nodes in trees for n in nodes.values() if n[0] == 'split']
all_leaf = [n[1] for nodes in trees for n in nodes.values() if n[0] == 'leaf']
print(f'\nThreshold range: [{min(all_thr):.2f}, {max(all_thr):.2f}]  | feature max ~{np.nanmax(X_test):.0f}')
print(f'Leaf range: [{min(all_leaf):.4f}, {max(all_leaf):.4f}]  | sum range ~[{m_float.min():.2f}, {m_float.max():.2f}]')

def margin_quant(x, Ff, Fl):
    """integer fixed-point: features/thresholds scaled by 2^Ff, leaves by 2^Fl."""
    sf = 1 << Ff
    s = 0
    for nodes in trees:
        nid = 0
        while nodes[nid][0] == 'split':
            _, fi, thr, yes, no, mis = nodes[nid]
            v = x[fi]
            if np.isnan(v):
                nid = mis
            else:
                vi = int(round(v * sf)); ti = int(round(thr * sf))
                nid = yes if vi < ti else no
        s += int(round(nodes[nid][1] * (1 << Fl)))
    return s

print(f'\n{"Ff":>3} {"Fl":>3} {"ROC-AUC":>9} {"feat bits":>10} {"leaf bits":>10} {"acc bits":>9}')
for Ff in [2, 3, 4, 6]:
    for Fl in [6, 8, 10]:
        mq = np.array([margin_quant(x, Ff, Fl) for x in X_test])
        auc = roc_auc_score(y_test, mq)
        feat_bits = int(np.ceil(np.log2((np.nanmax(X_test) + 1) * (1 << Ff)))) + 1
        leaf_max = max(abs(min(all_leaf)), abs(max(all_leaf)))
        leaf_bits = int(np.ceil(np.log2(leaf_max * (1 << Fl) + 1))) + 1
        acc_bits = int(np.ceil(np.log2(abs(mq).max() + 1))) + 1
        print(f'{Ff:>3} {Fl:>3} {auc:>9.4f} {feat_bits:>10} {leaf_bits:>10} {acc_bits:>9}')

print(f'\nReference float AUC: {auc_xgb:.4f} — pick the smallest (Ff,Fl) that holds it.')
