import numpy as np
import pandas as pd

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}

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

s2_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')
s3_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')
blend = 0.5 * s2_oof + 0.5 * s3_oof
w_blend = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])
preds = (blend[sept_mask] * w_blend).argmax(axis=1)

sept_df = df.iloc[sept_indices].copy().reset_index(drop=True)
sept_df['pred'] = preds
sept_df['true'] = y_sept
sept_df['p_fraud'] = 1.0 - blend[sept_mask, 0] # probability of being any fraud

# Highly confident predicted fraud
conf_fraud = sept_df[sept_df['pred'] != 0]
print(f"Confidently predicted fraud transactions: {len(conf_fraud)}")

# Check entities with confident fraud
for entity in ['device_id', 'receiver_id', 'sender_id']:
    fraud_entities = conf_fraud[entity].unique()
    # Transactions predicted as legit that share this entity
    pred_legit_same_entity = sept_df[(sept_df['pred'] == 0) & (sept_df[entity].isin(fraud_entities))]
    true_fraud_in_these = (pred_legit_same_entity['true'] != 0).sum()
    total_in_these = len(pred_legit_same_entity)
    print(f"\n{entity}:")
    print(f"  Unique fraud entities: {len(fraud_entities)}")
    print(f"  Transactions predicted as legit sharing these entities: {total_in_these}")
    print(f"  True fraud among them: {true_fraud_in_these} ({true_fraud_in_these/total_in_these:.1%})")
    if total_in_these > 0:
        print("  Breakdown of true labels in these 'legit' predictions:")
        print(pred_legit_same_entity['true'].value_counts().head(5))
