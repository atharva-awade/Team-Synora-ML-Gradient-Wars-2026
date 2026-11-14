import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

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
te['label'] = 'legit'
te['label_id'] = 0

df = pd.concat([tr, te], ignore_index=True)
df['t'] = pd.to_datetime(df['timestamp'])
df['ts'] = (df['t'] - df['t'].min()).dt.total_seconds().astype('int64')
df = df.sort_values('ts', kind='stable').reset_index(drop=True)

is_tr = df['is_train'].values == 1
tr_idx = np.where(is_tr)[0]
month = df['t'].dt.month.values
sept_mask = (month[tr_idx] == 9)
sept_indices = tr_idx[sept_mask]
y_all = np.array([L2I[l] for l in df['label'].values[tr_idx]])
y_sept = y_all[sept_mask]

# Load OOF arrays
s2_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')
s3_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')
blend = 0.5 * s2_oof + 0.5 * s3_oof

# Multipliers from eval_holdout_clean.py
w_blend = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])
base_preds = (blend[sept_mask] * w_blend).argmax(axis=1)

base_f1 = f1_score(y_sept, base_preds, average='macro')
print(f"Base Tuned GBDT September Macro-F1: {base_f1:.4f}")

# Let's inspect the September dataframe slice
sept_df = df.iloc[sept_indices].copy().reset_index(drop=True)
sept_df['pred'] = base_preds
sept_df['true'] = y_sept
sept_df['pred_label'] = [LABELS[p] for p in base_preds]
sept_df['true_label'] = [LABELS[p] for p in y_sept]

# 1. Test Layering Chain Propagation:
# If pred is layering_chain, find hops where sender == receiver of layering within 30 min and amount matches
layering_pred_idx = np.where(base_preds == L2I['layering_chain'])[0]
print(f"Base layering_chain predictions: {len(layering_pred_idx)}")

# Map receiver -> list of (ts, amount)
layering_txns = sept_df.iloc[layering_pred_idx]
print(f"Unique receivers of predicted layering: {layering_txns['receiver_id'].nunique()}")

# Let's check how many subsequent txns in sept_df match:
cand_hops = sept_df[sept_df['sender_id'].isin(set(layering_txns['receiver_id']))]
m_hops = pd.merge(
    layering_txns[['txn_id', 'receiver_id', 'ts', 'amount']],
    cand_hops[['txn_id', 'sender_id', 'receiver_id', 'ts', 'amount', 'true_label', 'pred_label']],
    left_on='receiver_id',
    right_on='sender_id',
    suffixes=('_hop1', '_hop2')
)
valid_hops = m_hops[
    (m_hops['ts_hop2'] > m_hops['ts_hop1']) &
    (m_hops['ts_hop2'] - m_hops['ts_hop1'] <= 1800) &
    (m_hops['amount_hop2'] <= m_hops['amount_hop1']) &
    (m_hops['amount_hop2'] >= 0.7 * m_hops['amount_hop1'])
]
print(f"Total matching layering hops: {len(valid_hops)}")
print(valid_hops['true_label'].value_counts())
print("\nPredicted labels of these hops before propagation:")
print(valid_hops['pred_label'].value_counts())
