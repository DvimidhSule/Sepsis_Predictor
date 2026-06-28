"""
XGBoost to Verilog Compiler.
Parses the serialized edge XGBoost booster, quantizes thresholds and leaves,
and generates a synthesizable combinational Verilog module (`sepsis_engine.v`).
Also exports `golden_vectors.csv` for test verification.
"""
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

FRAC_FEAT = 4          # features/thresholds scaled by 2^4
FRAC_LEAF = 8          # leaves scaled by 2^8
FW = 14                # feature datapath width (signed)
MW = 16                # margin accumulator width (signed)
EDGE = ['Temp_mean3', 'Resp_mean3', 'HR_mean3', 'SBP_mean3']
FNAME = {f: i for i, f in enumerate(EDGE)}
SHORT = ['temp', 'resp', 'hr', 'sbp']

# ---- rebuild model + splits (deterministic) ----
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
bst = xgb.train(params, dtr, num_boost_round=60, evals=[(dval, 'val')], early_stopping_rounds=15, verbose_eval=False)
n_trees = bst.best_iteration + 1
print(f'Model: {n_trees} trees, depth 4')

# ---- parse + quantize trees ----
tdf = bst.trees_to_dataframe(); tdf = tdf[tdf['Tree'] < n_trees]
trees = []
for t, sub in tdf.groupby('Tree'):
    nodes = {}
    for _, r in sub.iterrows():
        nid = int(r['Node'])
        if r['Feature'] == 'Leaf':
            nodes[nid] = ('leaf', int(round(float(r['Gain']) * (1 << FRAC_LEAF))))
        else:
            yes = int(r['Yes'].split('-')[1]); no = int(r['No'].split('-')[1]); mis = int(r['Missing'].split('-')[1])
            thr_q = int(round(float(r['Split']) * (1 << FRAC_FEAT)))
            nodes[nid] = ('split', FNAME[r['Feature']], thr_q, yes, no, mis)
    trees.append(nodes)

# base margin (intercept): booster margin - sum of my float leaves, constant across rows
Xv = val_sub[EDGE].values
def margin_q(x):
    s = 0
    for nodes in trees:
        nid = 0
        while nodes[nid][0] == 'split':
            _, fi, thr, yes, no, mis = nodes[nid]
            v = x[fi]
            if np.isnan(v): nid = mis
            else: nid = yes if int(round(v*(1<<FRAC_FEAT))) < thr else no
        s += nodes[nid][1]
    return s
bm = bst.predict(xgb.DMatrix(Xv, feature_names=EDGE), output_margin=True, iteration_range=(0, n_trees))
mq = np.array([margin_q(x) for x in Xv]) / (1 << FRAC_LEAF)
base_margin = float(np.mean(bm - mq))
# alarm threshold for p=0.5 operating point: base_margin + sum_leaves_float >= 0
ALARM_THRESHOLD = int(round((0.0 - base_margin) * (1 << FRAC_LEAF)))   # in leaf-fixed-point
print(f'base_margin {base_margin:.4f} -> ALARM_THRESHOLD (p=0.5) = {ALARM_THRESHOLD}')

# ---- emit a tree as a nested ternary ----
def emit(nodes, nid):
    n = nodes[nid]
    if n[0] == 'leaf':
        return f"{MW}'sd{n[1]}" if n[1] >= 0 else f"-{MW}'sd{abs(n[1])}"
    _, fi, thr, yes, no, mis = n
    thr_s = f"{FW}'sd{thr}" if thr >= 0 else f"-{FW}'sd{abs(thr)}"
    y, no_, m = emit(nodes, yes), emit(nodes, no), emit(nodes, mis)
    f, v = SHORT[fi], f"{SHORT[fi]}_v"
    return f"({v} ? (($signed({f}) < {thr_s}) ? {y} : {no_}) : {m})"

lines = []
lines.append("// sepsis_engine.v  -- AUTO-GENERATED from the trained edge GBM by hw/generate_verilog.py")
lines.append("// Do not edit by hand. 25 depth-4 trees, fixed-point. See hw/QUANT_SPEC.md.")
lines.append(f"// FRAC_FEAT={FRAC_FEAT} (x{1<<FRAC_FEAT})  FRAC_LEAF={FRAC_LEAF} (x{1<<FRAC_LEAF})")
lines.append("`timescale 1ns/1ps")
lines.append("")
lines.append("module sepsis_engine (")
lines.append(f"    input  signed [{FW-1}:0] temp, resp, hr, sbp,   // 3h-mean features, Q{FW-FRAC_FEAT}.{FRAC_FEAT} fixed-point")
lines.append("    input  temp_v, resp_v, hr_v, sbp_v,            // per-feature valid bits (low = missing)")
lines.append(f"    output signed [{MW-1}:0] margin,               // summed leaf margin (Q{MW-FRAC_LEAF}.{FRAC_LEAF})")
lines.append("    output sepsis_alarm")
lines.append(");")
lines.append(f"    localparam signed [{MW-1}:0] ALARM_THRESHOLD = {MW}'sd{ALARM_THRESHOLD}; // p=0.5 operating point")
lines.append("")
for ti, nodes in enumerate(trees):
    lines.append(f"    wire signed [{MW-1}:0] t{ti} = {emit(nodes, 0)};")
lines.append("")
lines.append("    assign margin = " + " + ".join(f"t{ti}" for ti in range(len(trees))) + ";")
lines.append("    assign sepsis_alarm = (margin >= ALARM_THRESHOLD);")
lines.append("endmodule")

with open('hw/sepsis_engine.v', 'w') as f:
    f.write("\n".join(lines) + "\n")
print(f'Wrote hw/sepsis_engine.v ({len(trees)} trees, {sum(1 for nd in trees for n in nd.values() if n[0]==chr(115)+"plit" or n[0]=="split")} comparators)')

# ---- golden vectors for 2.3 co-sim: quantized features -> margin, alarm ----
Xt = test_df[EDGE].values
def feat_q(v):  # NaN -> sentinel 0 + valid=0
    return (0, 0) if np.isnan(v) else (int(round(v*(1<<FRAC_FEAT))), 1)
rows = []
for x in Xt[:2000]:
    qv = [feat_q(x[i]) for i in range(4)]
    m = margin_q(x)
    rows.append([qv[0][0],qv[1][0],qv[2][0],qv[3][0], qv[0][1],qv[1][1],qv[2][1],qv[3][1],
                 m, int(m >= ALARM_THRESHOLD)])
gv = pd.DataFrame(rows, columns=['temp','resp','hr','sbp','temp_v','resp_v','hr_v','sbp_v','margin','alarm'])
gv.to_csv('hw/golden_vectors.csv', index=False)
print(f'Wrote hw/golden_vectors.csv ({len(gv)} rows) for bit-exact co-sim')
print(f'Golden alarm rate: {gv.alarm.mean():.1%}')
