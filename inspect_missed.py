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
# Add probabilities
for i, l in enumerate(LABELS):
    sept_df['p_' + l] = blend[sept_mask, i]

# Look at missed mule_account (true == mule, pred == legit)
mule_caught = sept_df[(sept_df['true'] == L2I['mule_account']) & (sept_df['pred'] == L2I['mule_account'])]
mule_missed = sept_df[(sept_df['true'] == L2I['mule_account']) & (sept_df['pred'] == 0)]

print(f"Mule caught: {len(mule_caught)}, Mule missed: {len(mule_missed)}")
print("\nMule CAUGHT mean probabilities:")
print(mule_caught[['p_legit', 'p_mule_account', 'p_layering_chain']].mean())
print("\nMule MISSED mean probabilities:")
print(mule_missed[['p_legit', 'p_mule_account', 'p_layering_chain']].mean())

# Percentiles of p_mule_account in missed mules:
print("\nPercentiles of p_mule_account in MISSED mules:")
print(np.percentile(mule_missed['p_mule_account'], [10, 25, 50, 75, 90]))

# Percentiles of p_mule_account in TRUE LEGIT:
true_legit = sept_df[sept_df['true'] == 0]
print("\nPercentiles of p_mule_account in TRUE LEGIT:")
print(np.percentile(true_legit['p_mule_account'], [90, 95, 99, 99.5, 99.9]))
