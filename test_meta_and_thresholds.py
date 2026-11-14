import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

print("Loading data...")
tr = pd.read_parquet(r'd:\ML Gradient Gemini\cache\train_clean.parquet')
te = pd.read_parquet(r'd:\ML Gradient Gemini\cache\test_clean.parquet')
tr['is_train'] = 1
te['is_train'] = 0
df = pd.concat([tr, te], ignore_index=True)
df['t'] = pd.to_datetime(df['timestamp'])
df['ts'] = (df['t'] - df['t'].min()).dt.total_seconds().astype('int64')
df = df.sort_values('ts', kind='stable').reset_index(drop=True)

is_tr = df['is_train'].values == 1
tr_idx = np.where(is_tr)[0]
month = df['t'].dt.month.values
sept_mask = (month[tr_idx] == 9)
y_all = np.array([L2I[l] for l in df['label'].values[tr_idx]])
y_sept = y_all[sept_mask]

# Load all OOF models
s1 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_1_oof.npy')[sept_mask]
s2 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')[sept_mask]
s3 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')[sept_mask]
om1 = np.load(r'd:\ML Gradient Gemini\checkpoints_omega\oof_stage1.npy')[sept_mask]
om2 = np.load(r'd:\ML Gradient Gemini\checkpoints_omega\oof_stage2.npy')[sept_mask]

print("All 5 OOF models loaded for September holdout.")

# Check baseline blend
p_blend = (s2 + s3 + om2) / 3.0
base_f1 = f1_score(y_sept, p_blend.argmax(axis=1), average='macro')
print(f"Simple Argmax Ensemble September Macro-F1: {base_f1:.4f}")

# Method 1: Train a LogisticRegression Meta-Learner with balanced class weights
print("\n--- Testing Balanced Meta-Learner ---")
X_meta = np.hstack([s1, s2, s3, om1, om2]) # 60 features
# Train on first half of September, evaluate on second half
half = len(y_sept) // 2
clf = LogisticRegression(class_weight='balanced', max_iter=200, C=1.0, random_state=42)
clf.fit(X_meta[:half], y_sept[:half])
meta_preds = clf.predict(X_meta[half:])
val_f1 = f1_score(y_sept[half:], meta_preds, average='macro')
print(f"Meta-Learner Half-Split Val Macro-F1: {val_f1:.4f}")

# Method 2: Optimize Thresholds via Powell on Negative Macro-F1
print("\n--- Testing Powell Threshold Optimization on Probabilities ---")
# Use soft probabilities for the 11 fraud classes vs legit
# Decision rule: assign to fraud class c if P_c * w_c > P_0, argmax over fraud
def loss_func(weights):
    w = np.insert(weights, 0, 1.0)
    preds = (p_blend[:half] * w).argmax(axis=1)
    return -f1_score(y_sept[:half], preds, average='macro')

# Init with previous best weights
w_init = np.array([1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])
res = minimize(loss_func, w_init, method='Powell', options={'maxiter': 5, 'disp': True})
opt_w = np.insert(res.x, 0, 1.0)

test_preds = (p_blend[half:] * opt_w).argmax(axis=1)
opt_val_f1 = f1_score(y_sept[half:], test_preds, average='macro')
print(f"Powell Optimized Half-Split Val Macro-F1: {opt_val_f1:.4f}")

# Full September score with optimized weights
full_preds = (p_blend * opt_w).argmax(axis=1)
full_f1 = f1_score(y_sept, full_preds, average='macro')
print(f"Powell Optimized Full September Macro-F1: {full_f1:.4f}")
print("Per-class F1:")
for l, s in zip(LABELS, f1_score(y_sept, full_preds, average=None)):
    print(f"  {l:20s}: {s:.4f}")
