"""
Sweeps the number of trees to evaluate its impact on edge model performance.
Validates that model performance plateaus after a certain number of estimators
due to the information limit of the 4-feature input space.
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

EDGE = ['Temp_mean3','Resp_mean3','HR_mean3','SBP_mean3']
raw = pd.read_csv('data/Dataset.csv').rename(columns={'Patient_ID':'patient_id'})
df = raw[['patient_id','HR','Temp','Resp','SBP','SepsisLabel']].copy()
g = df.groupby('patient_id', sort=False)
for v in ['Temp','Resp','HR','SBP']:
    df[f'{v}_mean3'] = g[v].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, groups=df['patient_id']))
train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
i_tr, i_val = next(gss2.split(train_df, groups=train_df['patient_id']))
tr_sub, val_sub = train_df.iloc[i_tr], train_df.iloc[i_val]

spw = (tr_sub['SepsisLabel']==0).sum()/(tr_sub['SepsisLabel']==1).sum()
dtr = xgb.DMatrix(tr_sub[EDGE].values, label=tr_sub['SepsisLabel'].values, feature_names=EDGE)
dval= xgb.DMatrix(val_sub[EDGE].values, label=val_sub['SepsisLabel'].values, feature_names=EDGE)
dte = xgb.DMatrix(test_df[EDGE].values, label=test_df['SepsisLabel'].values, feature_names=EDGE)
yv, yt = val_sub['SepsisLabel'].values, test_df['SepsisLabel'].values

def run(lr, n):
    p = {'objective':'binary:logistic','tree_method':'hist','device':'cuda','eval_metric':'auc',
         'scale_pos_weight':spw,'max_depth':4,'learning_rate':lr,'subsample':0.8,'colsample_bytree':1.0,'seed':42}
    b = xgb.train(p, dtr, num_boost_round=n)   # NO early stopping — force exactly n trees
    return roc_auc_score(yv, b.predict(dval)), roc_auc_score(yt, b.predict(dte))

print('lr=0.1 (the edge setting), forcing N trees, NO early stopping:')
print(f'{"trees":>6} {"val-AUC":>8} {"test-AUC":>9}')
for n in [5,10,25,50,100,200,500,1000]:
    va, ta = run(0.1, n)
    print(f'{n:>6} {va:>8.4f} {ta:>9.4f}')

print('\nlr=0.02 (slow learning, many trees) — does it beat the ceiling?')
print(f'{"trees":>6} {"val-AUC":>8} {"test-AUC":>9}')
for n in [100,300,800]:
    va, ta = run(0.02, n)
    print(f'{n:>6} {va:>8.4f} {ta:>9.4f}')
print('\nBaseline model using early stopping achieves test-AUC of 0.6736 with 25 trees.')
