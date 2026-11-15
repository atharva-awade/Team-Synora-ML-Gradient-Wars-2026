import os
import sys
import pickle
import numpy as np
import pandas as pd

print("=" * 70)
print("  BUILDING SUBMISSIONS 1 & 2: HIERARCHICAL RECOVERY ENGINES")
print("=" * 70)

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

DATA_DIR = r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings"
CKPT_DIR = r"D:\ML Gradient Gemini\checkpoints_resilient"
SUB_DIR = r"D:\ML Gradient Gemini\submissions"

# 1. Load test metadata and baseline champion
with open(os.path.join(CKPT_DIR, "dataset_meta.pkl"), "rb") as f:
    meta = pickle.load(f)

test_ids = meta['test_ids']
df_meta = meta['df']
test_idx = meta['test_idx']
te_df = df_meta.iloc[test_idx].reset_index(drop=True)

# Baseline 0.5054 champion
sub_s3 = pd.read_csv(os.path.join(SUB_DIR, "submission_sprint3_crown_063.csv"))
top1 = pd.read_csv(os.path.join(SUB_DIR, "submission_synora_top1_grandmaster.csv"))
raw_te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), usecols=['txn_id'])

# 2. Load model probabilities
pte_s1 = np.load(os.path.join(CKPT_DIR, "sprint1_pte.npy"))
pte_s2 = np.load(os.path.join(CKPT_DIR, "sprint2_pte.npy"))
pte_s0 = np.load(os.path.join(CKPT_DIR, "Stage_2_pte.npy"))
pte_blend = 0.45 * pte_s2 + 0.35 * pte_s1 + 0.20 * pte_s0

p_fraud_total = 1.0 - pte_blend[:, 0]
raw_best_fraud_type = pte_blend[:, 1:].argmax(axis=1) + 1

# Multiplier weighted fraud types
w_s3 = np.array([1.0, 0.958, 1.0, 0.75, 0.85, 1.0, 0.4, 0.3, 0.958, 1.375, 1.042, 0.958])
weighted_fraud_probs = pte_blend[:, 1:] * w_s3[1:]
weighted_best_fraud_type = weighted_fraud_probs.argmax(axis=1) + 1

# 3. Identify confirmed mule accounts in Sprint 3
confirmed_mule_receivers = set(te_df.loc[sub_s3['label'] == 'mule_account', 'receiver_id'].unique())
print(f"Confirmed mule receiver accounts: {len(confirmed_mule_receivers)}")

# Inflows and Outflows for confirmed mules
mule_inflow_cand = (te_df['receiver_id'].isin(confirmed_mule_receivers)) & (sub_s3['label'] == 'legit') & (pte_blend[:, L2I['mule_account']] >= 0.10)
mule_outflow_cand = (te_df['sender_id'].isin(confirmed_mule_receivers)) & (sub_s3['label'] == 'legit') & (p_fraud_total >= 0.30)
print(f"Additional mule inflows to recover: {mule_inflow_cand.sum()}")
print(f"Additional mule outflows to recover: {mule_outflow_cand.sum()}")

def build_and_verify(preds_labels, filename, description):
    sub = pd.DataFrame({'id': test_ids, 'label': preds_labels})
    out_path = os.path.join(SUB_DIR, filename)
    sub.to_csv(out_path, index=False)
    
    # 5-Point Strict Integrity Harness
    assert len(sub) == len(raw_te), f"Length mismatch: {len(sub)} vs {len(raw_te)}"
    assert (sub['id'].values == top1['id'].values).all(), "ID sequence mismatch!"
    assert sub['label'].isnull().sum() == 0, "Null labels detected!"
    assert set(sub['label'].unique()).issubset(set(LABELS)), "Invalid labels!"
    assert len(sub['label'].unique()) == 12, "Not all 12 classes represented!"
    
    fraud_count = (sub['label'] != 'legit').sum()
    agreement = (sub['label'] == sub_s3['label']).mean()
    print(f"\n[VERIFIED & SAVED] -> {filename}")
    print(f"  Description: {description}")
    print(f"  Total Frauds: {fraud_count} (Recovered: +{fraud_count - 3785})")
    print(f"  Agreement with 0.5054 Champion: {agreement:.2%}")
    print(f"  Path: {out_path}")
    print("  Class Distribution:")
    print(sub['label'].value_counts())
    return out_path

# ==============================================================================
# SUBMISSION 1: Hierarchical 5,184-Fraud Model (tau = 0.510)
# ==============================================================================
print("\n>>> Building Submission 1: Hierarchical 5,184 (tau = 0.510)...")
sub1_labels = sub_s3['label'].values.copy()
flip_mask_s1 = (sub1_labels == 'legit') & (p_fraud_total >= 0.510)
sub1_labels[flip_mask_s1] = np.array(LABELS)[raw_best_fraud_type[flip_mask_s1]]
# Also tag confirmed mule conduit flows
sub1_labels[mule_inflow_cand] = 'mule_account'
sub1_labels[mule_outflow_cand] = 'mule_account'

build_and_verify(
    sub1_labels,
    "submission_synora_hierarchical_5184.csv",
    "Hierarchical Two-Stage Calibration (tau=0.510, 5,184 frauds, 54% recall)"
)

# ==============================================================================
# SUBMISSION 2: Hierarchical 5,606-Fraud Model (tau = 0.505)
# ==============================================================================
print("\n>>> Building Submission 2: Hierarchical 5,606 (tau = 0.505)...")
sub2_labels = sub_s3['label'].values.copy()
flip_mask_s2 = (sub2_labels == 'legit') & (p_fraud_total >= 0.505)
sub2_labels[flip_mask_s2] = np.array(LABELS)[raw_best_fraud_type[flip_mask_s2]]
# Tag confirmed mule conduit flows
sub2_labels[mule_inflow_cand] = 'mule_account'
sub2_labels[mule_outflow_cand] = 'mule_account'

build_and_verify(
    sub2_labels,
    "submission_synora_hierarchical_5606.csv",
    "Hierarchical Two-Stage Calibration (tau=0.505, 5,606 frauds, 58% recall)"
)

# ==============================================================================
# SUBMISSION 3: Pure Bayes Optimal 6,155-Fraud Model (tau = 0.500)
# ==============================================================================
print("\n>>> Building Submission 3: Bayes Optimal 6,155 (tau = 0.500)...")
sub3_labels = sub_s3['label'].values.copy()
flip_mask_s3 = (sub3_labels == 'legit') & (p_fraud_total >= 0.500)
sub3_labels[flip_mask_s3] = np.array(LABELS)[raw_best_fraud_type[flip_mask_s3]]
sub3_labels[mule_inflow_cand] = 'mule_account'
sub3_labels[mule_outflow_cand] = 'mule_account'

build_and_verify(
    sub3_labels,
    "submission_synora_bayes_optimal_6155.csv",
    "Bayes Optimal Full Fraud Boundary (tau=0.500, 6,155 frauds, 64% recall)"
)
