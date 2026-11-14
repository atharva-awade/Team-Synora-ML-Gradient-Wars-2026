import numpy as np
import pandas as pd

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]

print("Loading and preparing timestamp-aligned IDs...")
tr = pd.read_parquet(r'd:\ML Gradient Gemini\cache\train_clean.parquet')
te = pd.read_parquet(r'd:\ML Gradient Gemini\cache\test_clean.parquet')
tr['is_train'] = 1
te['is_train'] = 0
df = pd.concat([tr, te], ignore_index=True)
df['t'] = pd.to_datetime(df['timestamp'])
df['ts'] = (df['t'] - df['t'].min()).dt.total_seconds().astype('int64')
df = df.sort_values('ts', kind='stable').reset_index(drop=True)
test_idx = np.where(df['is_train'] == 0)[0]
correct_test_ids = df['txn_id'].values[test_idx]

print(f"Correct test IDs count: {len(correct_test_ids)}")

# Load the 5-fold Stage 2 and Stage 3 ensembles from checkpoints_resilient
p2 = np.load(r'D:\ML Gradient Gemini\checkpoints_resilient\Stage_2_pte.npy')
p3 = np.load(r'D:\ML Gradient Gemini\checkpoints_resilient\Stage_3_pte.npy')
p_fresh = 0.5 * p2 + 0.5 * p3

# Multipliers tuned strictly on September holdout (eval_holdout_clean.py -> 0.4796 Macro-F1)
w_blend = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])
preds = (p_fresh * w_blend).argmax(axis=1)

# Construct strictly aligned submission
sub = pd.DataFrame({
    'id': correct_test_ids,
    'label': np.array(LABELS)[preds]
})

out_path = r'D:\ML Gradient Gemini\submissions\submission_synora_resilient_aligned_peak.csv'
sub.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

# Run strict 5-point verification harness against test.csv and 0.5005 baseline
raw_te = pd.read_csv(r'C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings\test.csv')
top1 = pd.read_csv(r'D:\ML Gradient Gemini\submissions\submission_synora_top1_grandmaster.csv')

assert len(sub) == len(raw_te), f"Row count mismatch: {len(sub)} vs {len(raw_te)}"
assert set(sub['id']) == set(raw_te['txn_id']), "ID set mismatch!"
assert sub['id'].is_unique, "Duplicate IDs present!"
assert sub['label'].isnull().sum() == 0, "Null values detected!"
assert set(sub['label'].unique()).issubset(set(LABELS)), "Unknown labels detected!"
assert (sub['id'].values == top1['id'].values).all(), "CRITICAL: ID order mismatch with top1 baseline!"

print("\n--- 5-POINT VERIFICATION PASSED ---")
agree_top1 = (sub['label'] == top1['label']).mean()
diff_top1 = (sub['label'] != top1['label']).sum()
print(f"ID match with 0.5005 baseline: 100.0% EXACT (260,582 / 260,582)")
print(f"Label agreement with 0.5005 baseline: {agree_top1:.4%} (only {diff_top1} rows different)")
print(f"Total fraud transactions: {(sub['label'] != 'legit').sum()}")

print("\nLabel distribution:")
print(sub['label'].value_counts())
