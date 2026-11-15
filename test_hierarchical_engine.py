import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

print("Loading September holdout and OOF arrays...")
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

# Load Sprint 1, Sprint 2, and Champion OOF
oof1 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\sprint1_oof.npy')[sept_mask]
oof2 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\sprint2_oof.npy')[sept_mask]
oof_champ = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')[sept_mask]

P = 0.45 * oof2 + 0.35 * oof1 + 0.20 * oof_champ

# Previous baseline 0.5054 argmax
w_old = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])
base_preds = (P * w_old).argmax(axis=1)
base_f1 = f1_score(y_sept, base_preds, average='macro')
base_frauds = (base_preds != 0).sum()
print(f"Standard Argmax September Macro-F1: {base_f1:.4f} (Frauds: {base_frauds})")

# TWO-STAGE HIERARCHICAL DECISION ENGINE
# 1. Binary fraud probability
p_total_fraud = 1.0 - P[:, 0]

# 2. Calibrated relative fraud score among the 11 fraud classes
# Normalize fraud probabilities across the 11 fraud classes
P_fraud_only = P[:, 1:].copy()
row_sums = P_fraud_only.sum(axis=1, keepdims=True)
P_fraud_norm = P_fraud_only / np.maximum(row_sums, 1e-6)

# Test different binary thresholds theta
print("\n--- Evaluating Hierarchical Decision Engine ---")
best_score = base_f1
best_theta = 0.50
best_preds = base_preds

for theta in np.linspace(0.40, 0.55, 31):
    # Rule: If p_total_fraud >= theta, assign best fraud class. Else, legit (0)
    is_fraud = (p_total_fraud >= theta)
    preds = np.zeros(len(y_sept), dtype=int)
    
    # Best fraud class among the 11 typologies
    best_c = (P_fraud_only * w_old[1:]).argmax(axis=1) + 1
    preds[is_fraud] = best_c[is_fraud]
    
    sc = f1_score(y_sept, preds, average='macro')
    n_f = is_fraud.sum()
    if sc > best_score:
        best_score = sc
        best_theta = theta
        best_preds = preds
        print(f"  * NEW BEST: theta={theta:.3f} -> Macro-F1 = {sc:.4f} (Frauds: {n_f}, Delta: {sc - base_f1:+.4f})")

print(f"\n=======================================================")
print(f"OPTIMAL HIERARCHICAL SEPTEMBER MACRO-F1: {best_score:.4f} (theta={best_theta:.3f})")
print(f"Base Argmax Macro-F1: {base_f1:.4f} -> Gain: {best_score - base_f1:+.4f}")
print(f"Fraud count: {(best_preds != 0).sum()} (Base: {base_frauds})")
print(f"=======================================================")

print("\nPer-Class F1 comparison:")
pcf1_base = f1_score(y_sept, base_preds, average=None)
pcf1_hier = f1_score(y_sept, best_preds, average=None)
for l, b, h in zip(LABELS, pcf1_base, pcf1_hier):
    print(f"  {l:20s}: Base={b:.4f} -> Hierarchical={h:.4f} ({h-b:+.4f})")
