import os
import sys
import pickle
import numpy as np
import pandas as pd

SUB_DIR = r"D:\ML Gradient Gemini\submissions"
DATA_DIR = r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings"

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]

# Load submissions 1, 2, and 3
s1 = pd.read_csv(os.path.join(SUB_DIR, "submission_synora_hierarchical_5184.csv"))
s2 = pd.read_csv(os.path.join(SUB_DIR, "submission_synora_hierarchical_5606.csv"))
s3 = pd.read_csv(os.path.join(SUB_DIR, "submission_synora_bayes_optimal_6155.csv"))
top1 = pd.read_csv(os.path.join(SUB_DIR, "submission_synora_top1_grandmaster.csv"))
raw_te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), usecols=['txn_id'])

# Consensus blend:
# If s1 and s2 both predict a fraud class, it is guaranteed high-confidence fraud.
# If s2 predicts fraud and s1 predicts legit, accept it if s3 also predicts that fraud!
preds_emperor = s1['label'].values.copy()

# Add confirmed consensus from s2 and s3
promote_mask = (s1['label'] == 'legit') & (s2['label'] != 'legit') & (s2['label'] == s3['label'])
preds_emperor[promote_mask] = s2.loc[promote_mask, 'label'].values

sub_emp = pd.DataFrame({'id': s1['id'].values, 'label': preds_emperor})
out_path = os.path.join(SUB_DIR, "submission_synora_emperor_crown_no1.csv")
sub_emp.to_csv(out_path, index=False)

# 5-point harness
assert len(sub_emp) == len(raw_te)
assert (sub_emp['id'].values == top1['id'].values).all()
assert sub_emp['label'].isnull().sum() == 0
assert set(sub_emp['label'].unique()).issubset(set(LABELS))
assert len(sub_emp['label'].unique()) == 12

fraud_cnt = (sub_emp['label'] != 'legit').sum()
print("=" * 70)
print(f"[VERIFIED & SAVED] -> submission_synora_emperor_crown_no1.csv")
print(f"  Total Frauds: {fraud_cnt}")
print(f"  Agreement with S1 (5,281): {(sub_emp['label'] == s1['label']).mean():.2%}")
print(f"  Agreement with S2 (5,703): {(sub_emp['label'] == s2['label']).mean():.2%}")
print("  Class Distribution:")
print(sub_emp['label'].value_counts())
print("=" * 70)
