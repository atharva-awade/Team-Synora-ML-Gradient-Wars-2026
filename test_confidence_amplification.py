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

# Baseline probabilities for September
s2 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')[sept_mask]
s3 = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')[sept_mask]
P = 0.5 * s2 + 0.5 * s3
w = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])

base_preds = (P * w).argmax(axis=1)
base_f1 = f1_score(y_sept, base_preds, average='macro')
print(f"Baseline September Macro-F1: {base_f1:.4f}")

# Extract graph IDs for September slice
sdf = df.iloc[tr_idx[sept_mask]].copy().reset_index(drop=True)
sdf['pred'] = base_preds
sdf['max_p'] = (P * w).max(axis=1)

# Confident fraud calls: fraud prediction with max_p > 0.60
is_conf_fraud = (base_preds != 0) & (sdf['max_p'] > 0.50)
print(f"Confident fraud transactions in September: {is_conf_fraud.sum()}")

# Map of confident entities:
# For each entity, what is its dominant confirmed fraud class?
conf_df = sdf[is_conf_fraud]

for boost_weight in [1.2, 1.5, 2.0, 3.0]:
    P_boosted = P.copy()
    
    # 1. Receiver-level contagion:
    # If a receiver received confirmed fraud of class C, boost class C for other txns to that receiver
    r_fraud_class = conf_df.groupby('receiver_id')['pred'].agg(lambda x: x.mode()[0]).to_dict()
    # Find transactions paying these receivers
    matched_r = sdf['receiver_id'].map(r_fraud_class)
    has_match_r = matched_r.notnull()
    for c in range(1, K):
        mask_c = has_match_r & (matched_r == c)
        P_boosted[mask_c.values, c] *= boost_weight
        
    # 2. Device-level contagion for velocity_bot
    bot_devices = set(conf_df[conf_df['pred'] == L2I['velocity_bot']]['device_id'].unique())
    is_bot_dev = sdf['device_id'].isin(bot_devices).values
    P_boosted[is_bot_dev, L2I['velocity_bot']] *= boost_weight
    
    # 3. Sender-level contagion for refund_abuse
    ref_senders = set(conf_df[conf_df['pred'] == L2I['refund_abuse']]['sender_id'].unique())
    is_ref_sender = (sdf['sender_id'].isin(ref_senders) & (sdf['txn_type'] == 'REFUND')).values
    P_boosted[is_ref_sender, L2I['refund_abuse']] *= boost_weight
    
    boost_preds = (P_boosted * w).argmax(axis=1)
    new_f1 = f1_score(y_sept, boost_preds, average='macro')
    diff = (boost_preds != base_preds).sum()
    print(f"Boost Weight {boost_weight:.1f} -> New Macro-F1: {new_f1:.4f} (Delta: {new_f1 - base_f1:+.4f}, Modified rows: {diff})")
