import os
import sys
import numpy as np
import pandas as pd
from src.config import (
    TEST_PARQUET, CACHE_DIR, SUBMISSIONS_DIR,
    LABELS, LABEL2ID, ID2LABEL, EXPECTED_TEST_COUNTS
)

sys.stdout.reconfigure(encoding='utf-8')

def calibrate_submission_quotas():
    print("[POST-PROCESSING] Running Global Constrained Quota Matching on Test Predictions...", flush=True)

    # Load test dataframe for txn_id
    test_df = pd.read_parquet(TEST_PARQUET)
    test_ids = test_df['txn_id'].values
    n_test = len(test_ids)

    # Load baseline submission to compare
    base_sub_path = os.path.join(SUBMISSIONS_DIR, "submission_vortex12_winning.csv")
    base_sub = pd.read_csv(base_sub_path)

    # Target class quotas scaled by a conservative 0.65x multiplier to prioritize precision
    TARGET_QUOTAS = {
        'mule_account': 1100,
        'velocity_bot': 900,
        'qr_swap': 800,
        'layering_chain': 750,
        'phishing_collect': 750,
        'sim_swap': 450,
        'account_takeover': 450,
        'refund_abuse': 350,
        'merchant_collusion': 280,
        'fake_cashback': 280,
        'loan_app_extortion': 220,
    }

    # Load typology feature scores as secondary high-precision rankers
    typology_df = pd.read_parquet(os.path.join(CACHE_DIR, "typology_features.parquet")).iloc[-n_test:].reset_index(drop=True)

    # Ranker score combinations for under-represented classes
    # 1. merchant_collusion
    graph_df = pd.read_parquet(os.path.join(CACHE_DIR, "graph_features.parquet")).iloc[-n_test:].reset_index(drop=True)
    collusion_score = (typology_df['is_round_1000'] * 2.0 + typology_df['is_round_500'] + (graph_df['r_out_tx_cnt'] < 5).astype(float)).values

    # 2. velocity_bot
    bot_score = typology_df['score_velocity_bot'].values

    # 3. loan_app_extortion
    loan_score = typology_df['score_loan_extortion'].values

    # 4. qr_swap
    qr_score = typology_df['score_qr_swap'].values

    preds = base_sub['label'].values.copy()

    def boost_class(label_name, score_arr, target_quota):
        current_cnt = (preds == label_name).sum()
        needed = target_quota - current_cnt
        if needed <= 0:
            return
        # Find candidates currently predicted as 'legit' with highest score
        cand_indices = np.where(preds == 'legit')[0]
        cand_scores = score_arr[cand_indices]
        top_k_cand = cand_indices[np.argsort(cand_scores)[::-1][:needed]]
        preds[top_k_cand] = label_name
        print(f"  -> Quota boost for {label_name:<20}: {current_cnt:>5} -> {(preds == label_name).sum():>5} rows (target ~{target_quota})")

    boost_class('merchant_collusion', collusion_score, TARGET_QUOTAS['merchant_collusion'])
    boost_class('velocity_bot', bot_score, TARGET_QUOTAS['velocity_bot'])
    boost_class('loan_app_extortion', loan_score, TARGET_QUOTAS['loan_app_extortion'])
    boost_class('qr_swap', qr_score, TARGET_QUOTAS['qr_swap'])
    boost_class('fake_cashback', typology_df['is_micro_amount'].values, TARGET_QUOTAS['fake_cashback'])

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_vortex12_calibrated_quota.csv")
    pd.DataFrame({'id': test_ids, 'label': preds}).to_csv(out_path, index=False)
    print(f"\n[SUCCESS] Wrote calibrated submission to: {out_path}")
    return out_path

if __name__ == "__main__":
    calibrate_submission_quotas()

