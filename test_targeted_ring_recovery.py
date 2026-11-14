import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}

print("Loading September holdout...")
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
sept_indices = tr_idx[sept_mask]
y_all = np.array([L2I[l] for l in df['label'].values[tr_idx]])
y_sept = y_all[sept_mask]

# Baseline predictions
s2_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')
s3_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')
blend = 0.5 * s2_oof + 0.5 * s3_oof
w_blend = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])

p_sept = blend[sept_mask]
base_preds = (p_sept * w_blend).argmax(axis=1)
base_f1 = f1_score(y_sept, base_preds, average='macro')
print(f"Base September Holdout Macro-F1: {base_f1:.4f}")
print(f"Base fraud count: {(base_preds != 0).sum()} out of {len(base_preds)}")

# Create working dataframe for September
sdf = df.iloc[sept_indices].copy().reset_index(drop=True)
sdf['pred'] = base_preds
sdf['p_mule'] = p_sept[:, L2I['mule_account']]
sdf['p_layer'] = p_sept[:, L2I['layering_chain']]
sdf['p_bot'] = p_sept[:, L2I['velocity_bot']]
sdf['p_refund'] = p_sept[:, L2I['refund_abuse']]
sdf['p_legit'] = p_sept[:, 0]

new_preds = base_preds.copy()

# 1. High-confidence Layering Chain Forward/Backward Hop Recovery
# Confirmed layering: base pred is layering with high score
conf_layer_idx = np.where((base_preds == L2I['layering_chain']) & (sdf['p_layer'] > 0.40))[0]
conf_layer = sdf.iloc[conf_layer_idx]
print(f"High confidence layering seeds: {len(conf_layer)}")

# Next hop: receiver of conf_layer sends money out within 1800s, amount in [0.75*amt, amt]
cand_layer_next = sdf[(sdf['sender_id'].isin(set(conf_layer['receiver_id']))) & (new_preds == 0)]
m_next = pd.merge(
    conf_layer[['receiver_id', 'ts', 'amount']].rename(columns={'receiver_id': 'mid'}),
    cand_layer_next[['sender_id', 'ts', 'amount', 'p_layer']].reset_index().rename(columns={'sender_id': 'mid'}),
    on='mid', suffixes=('_seed', '_next')
)
valid_next = m_next[
    (m_next['ts_next'] > m_next['ts_seed']) &
    (m_next['ts_next'] - m_next['ts_seed'] <= 1800) &
    (m_next['amount_next'] <= m_next['amount_seed']) &
    (m_next['amount_next'] >= 0.70 * m_next['amount_seed']) &
    (m_next['p_layer'] >= 0.10) # at least slight model suspicion
]
recovered_layer_idx = valid_next['index'].unique()
print(f"Recovered layering hops: {len(recovered_layer_idx)}")
# Check accuracy of recovered hops against ground truth:
true_recovered = (y_sept[recovered_layer_idx] == L2I['layering_chain']).sum()
print(f"Accuracy of recovered layering: {true_recovered} / {len(recovered_layer_idx)} ({true_recovered/max(1, len(recovered_layer_idx)):.1%})")
new_preds[recovered_layer_idx] = L2I['layering_chain']

# 2. High-confidence Mule Inflow Recovery
# Confirmed mule receiver: receiver has high confidence mule txns
mule_conf_receivers = sdf[(base_preds == L2I['mule_account']) & (sdf['p_mule'] > 0.50)]['receiver_id'].value_counts()
top_mule_rec = set(mule_conf_receivers[mule_conf_receivers >= 1].index)
print(f"Confirmed mule receiver accounts: {len(top_mule_rec)}")

# Incoming transactions to these mule accounts where model was suspicious (p_mule >= 0.20)
cand_mule_in = sdf[(sdf['receiver_id'].isin(top_mule_rec)) & (new_preds == 0) & (sdf['p_mule'] >= 0.20)]
recovered_mule_idx = cand_mule_in.index.values
print(f"Recovered mule inflow txns: {len(recovered_mule_idx)}")
true_mule = (y_sept[recovered_mule_idx] == L2I['mule_account']).sum()
print(f"Accuracy of recovered mule inflow: {true_mule} / {len(recovered_mule_idx)} ({true_mule/max(1, len(recovered_mule_idx)):.1%})")
new_preds[recovered_mule_idx] = L2I['mule_account']

# 3. High-confidence Velocity Bot Burst Recovery
conf_bot_devices = set(sdf[(base_preds == L2I['velocity_bot']) & (sdf['p_bot'] > 0.50)]['device_id'].unique())
print(f"Confirmed bot devices: {len(conf_bot_devices)}")
cand_bot = sdf[(sdf['device_id'].isin(conf_bot_devices)) & (new_preds == 0) & (sdf['p_bot'] >= 0.15)]
recovered_bot_idx = cand_bot.index.values
print(f"Recovered bot txns: {len(recovered_bot_idx)}")
true_bot = (y_sept[recovered_bot_idx] == L2I['velocity_bot']).sum()
print(f"Accuracy of recovered bot: {true_bot} / {len(recovered_bot_idx)} ({true_bot/max(1, len(recovered_bot_idx)):.1%})")
new_preds[recovered_bot_idx] = L2I['velocity_bot']

# 4. High-confidence Refund Abuse Account Recovery
conf_refund_senders = set(sdf[(base_preds == L2I['refund_abuse']) & (sdf['p_refund'] > 0.50)]['sender_id'].unique())
cand_refund = sdf[(sdf['sender_id'].isin(conf_refund_senders)) & (sdf['txn_type'] == 'REFUND') & (new_preds == 0) & (sdf['p_refund'] >= 0.15)]
recovered_refund_idx = cand_refund.index.values
print(f"Recovered refund abuse txns: {len(recovered_refund_idx)}")
true_ref = (y_sept[recovered_refund_idx] == L2I['refund_abuse']).sum()
print(f"Accuracy of recovered refund: {true_ref} / {len(recovered_refund_idx)} ({true_ref/max(1, len(recovered_refund_idx)):.1%})")
new_preds[recovered_refund_idx] = L2I['refund_abuse']

eval_f1 = f1_score(y_sept, new_preds, average='macro')
print(f"\n==========================================")
print(f"NEW SEPTEMBER HOLDOUT MACRO-F1: {eval_f1:.4f} (Base: {base_f1:.4f}, Delta: {eval_f1 - base_f1:+.4f})")
print(f"Total fraud count: {(new_preds != 0).sum()} (Base: {(base_preds != 0).sum()})")
print(f"==========================================")
