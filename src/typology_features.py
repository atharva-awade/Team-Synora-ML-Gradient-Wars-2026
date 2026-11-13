import os
import gc
import time
import numpy as np
import pandas as pd
from src.config import TRAIN_PARQUET, TEST_PARQUET, ACCOUNTS_PARQUET, CACHE_DIR

TYPOLOGY_FEATS_PARQUET = os.path.join(CACHE_DIR, "typology_features.parquet")

def compute_window_counts(keys: np.ndarray, time_sec: np.ndarray, window_sec: int) -> np.ndarray:
    """
    Computes bidirectional window row counts in O(n log n) using searchsorted.
    Returns counts within [t - window_sec, t + window_sec] for each key.
    """
    n = len(keys)
    codes, _ = pd.factorize(keys)
    codes = codes.astype(np.int64)

    BIG = 20_000_000
    comp_key = codes * BIG + time_sec
    order = np.argsort(comp_key, kind="mergesort")
    sorted_comp_key = comp_key[order]
    sorted_codes = codes[order]

    num_keys = int(codes.max()) + 1
    starts = np.searchsorted(sorted_codes, np.arange(num_keys), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(num_keys), side="right")

    r_idx = np.minimum(np.searchsorted(sorted_comp_key, sorted_comp_key + window_sec, side="right"), ends[sorted_codes])
    l_idx = np.maximum(np.searchsorted(sorted_comp_key, sorted_comp_key - window_sec, side="left"), starts[sorted_codes])

    counts = np.empty(n, dtype=np.int32)
    counts[order] = (r_idx - l_idx)
    return counts

def compute_typology_and_window_features(full_df: pd.DataFrame, accounts_df: pd.DataFrame) -> pd.DataFrame:
    print("[TYPOLOGY] Computing Generator Inversion and Vectorized Window Features...", flush=True)
    t0 = time.time()
    n = len(full_df)

    ts = pd.to_datetime(full_df['timestamp'])
    time_sec = (ts.astype("int64") // 10**9).values
    hour = ts.dt.hour.values
    weekday = ts.dt.weekday.values

    # Accounts metadata join
    home_city_map = dict(zip(accounts_df['account_id'], accounts_df['home_city']))
    sender_home_city = full_df['sender_id'].map(home_city_map).fillna("UNKNOWN")
    away_from_home = (full_df['device_city'] != sender_home_city).astype(np.float32)

    amount = full_df['amount'].fillna(0.0).values
    balance = full_df['balance_before'].fillna(0.0).values
    ratio_bal = np.where(balance > 0, amount / (balance + 1e-4), 0.0)

    feats = pd.DataFrame(index=full_df.index)

    # 1. Cyclical Time & Night
    feats['hour'] = hour.astype(np.float32)
    feats['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    feats['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    feats['weekday'] = weekday.astype(np.float32)
    feats['is_weekend'] = (weekday >= 5).astype(np.float32)
    feats['is_night'] = ((hour >= 0) & (hour <= 5)).astype(np.float32)
    feats['is_evening'] = ((hour >= 20) & (hour <= 23)).astype(np.float32)

    # 2. Amount & Balance Dynamics
    feats['amt_log'] = np.log1p(np.maximum(0.0, amount)).astype(np.float32)
    feats['bal_log'] = np.log1p(np.maximum(0.0, balance)).astype(np.float32)
    feats['amt_to_bal_ratio'] = np.clip(ratio_bal, 0.0, 10.0).astype(np.float32)
    feats['amt_gt_bal'] = (amount > balance).astype(np.float32)

    # Whole-number & round limits (phishing, collusion, bot signatures)
    feats['is_int'] = (np.mod(amount, 1.0) == 0.0).astype(np.float32)
    feats['is_round_100'] = (np.mod(amount, 100.0) == 0.0).astype(np.float32)
    feats['is_round_500'] = (np.mod(amount, 500.0) == 0.0).astype(np.float32)
    feats['is_round_1000'] = (np.mod(amount, 1000.0) == 0.0).astype(np.float32)
    feats['is_micro_amount'] = (amount <= 10.0).astype(np.float32)

    # Phishing collect: just-under limits
    feats['just_under_2k'] = ((amount >= 1990) & (amount < 2000)).astype(np.float32)
    feats['just_under_5k'] = ((amount >= 4990) & (amount < 5000)).astype(np.float32)
    feats['just_under_10k'] = ((amount >= 9990) & (amount < 10000)).astype(np.float32)
    feats['just_under_20k'] = ((amount >= 19900) & (amount < 20000)).astype(np.float32)
    feats['just_under_50k'] = ((amount >= 49900) & (amount < 50000)).astype(np.float32)

    # Phishing drain ratio template: amount / balance ≈ 0.909 (10/11)
    feats['is_phishing_drain_ratio'] = (np.abs(ratio_bal - 0.909) < 0.015).astype(np.float32)

    # 3. Security, Hardware & Location Shifts
    feats['away_from_home'] = away_from_home.values
    feats['sim_age_hours_log'] = np.log1p(full_df['sim_age_hours'].fillna(0.0)).astype(np.float32)
    feats['device_age_days_log'] = np.log1p(full_df['device_age_days'].fillna(0.0)).astype(np.float32)
    feats['sender_acc_age_log'] = np.log1p(full_df['sender_account_age_days'].fillna(0.0)).astype(np.float32)
    feats['login_to_txn_sec_log'] = np.log1p(full_df['login_to_txn_sec'].fillna(0.0)).astype(np.float32)

    feats['new_payee'] = full_df['new_payee'].fillna(0.0).astype(np.float32)
    feats['pin_retries'] = full_df['pin_retries'].fillna(0.0).astype(np.float32)
    feats['pin_retries_gt0'] = (feats['pin_retries'] > 0).astype(np.float32)
    feats['ip_risk_score'] = full_df['ip_risk_score'].fillna(0.2).astype(np.float32)

    # QR mismatch (57x lift on qr_swap)
    feats['qr_vpa_mismatch'] = full_df['qr_vpa_mismatch'].fillna(0.0).astype(np.float32)

    # 4. Explicit Typology Detectors (Symbolic AI Formulas)
    txn_type = full_df['txn_type'].fillna('NONE').values
    merch_cat = full_df['merchant_category'].fillna('NONE').values

    feats['is_collect'] = (txn_type == 'COLLECT').astype(np.float32)
    feats['is_refund'] = (txn_type == 'REFUND').astype(np.float32)
    feats['is_qr'] = (txn_type == 'QR').astype(np.float32)
    feats['is_pay'] = (txn_type == 'PAY').astype(np.float32)

    feats['is_lending_cat'] = (merch_cat == 'lending').astype(np.float32)
    feats['is_ecommerce_cat'] = (merch_cat == 'ecommerce').astype(np.float32)
    feats['is_gaming_cat'] = (merch_cat == 'gaming').astype(np.float32)

    # Exact Symbolic detectors
    feats['score_qr_swap'] = (feats['qr_vpa_mismatch'] * 3.0 + feats['is_qr'] * 1.5 + feats['is_round_100']).astype(np.float32)
    feats['score_phishing'] = (feats['is_collect'] * 2.5 + feats['is_phishing_drain_ratio'] * 3.0 + (feats['just_under_2k'] + feats['just_under_5k'] + feats['just_under_10k']) * 2.0).astype(np.float32)
    feats['score_loan_extortion'] = (feats['is_lending_cat'] * 4.0 + feats['is_evening'] * 1.5 + feats['is_night'] * 2.0).astype(np.float32)
    feats['score_refund_abuse'] = (feats['is_refund'] * 4.0 + feats['is_ecommerce_cat'] * 2.0).astype(np.float32)
    feats['score_sim_swap'] = ((full_df['sim_age_hours'] < 72).astype(np.float32) * 3.0 + feats['is_night'] * 2.0 + feats['away_from_home'] * 1.5).astype(np.float32)
    feats['score_ato'] = (feats['new_payee'] * 2.0 + feats['pin_retries_gt0'] * 2.0 + feats['away_from_home'] * 2.0 + (full_df['device_age_days'] < 3).astype(np.float32) * 2.0).astype(np.float32)

    # 5. Vectorized Rolling Window Burst Counts (O(n log n))
    print("  -> Computing rolling burst windows for device, sender, and receiver...", flush=True)
    dev_keys = full_df['device_id'].fillna("MISSING").astype(str).values
    snd_keys = full_df['sender_id'].astype(str).values
    rcv_keys = full_df['receiver_id'].astype(str).values

    feats['dev_burst_cnt_2m'] = compute_window_counts(dev_keys, time_sec, window_sec=120)
    feats['dev_burst_cnt_10m'] = compute_window_counts(dev_keys, time_sec, window_sec=600)
    feats['snd_burst_cnt_15m'] = compute_window_counts(snd_keys, time_sec, window_sec=900)
    feats['snd_burst_cnt_1h'] = compute_window_counts(snd_keys, time_sec, window_sec=3600)
    feats['rcv_fanin_cnt_10m'] = compute_window_counts(rcv_keys, time_sec, window_sec=600)
    feats['rcv_fanin_cnt_1h'] = compute_window_counts(rcv_keys, time_sec, window_sec=3600)

    # Velocity bot detector using burst counts
    feats['score_velocity_bot'] = ((feats['dev_burst_cnt_2m'] >= 5).astype(np.float32) * 3.0 + feats['is_int'] * 1.5 + (full_df['login_to_txn_sec'] < 30).astype(np.float32)).astype(np.float32)

    print(f"[TYPOLOGY] Generated {feats.shape[1]} typology features in {time.time() - t0:.1f}s.", flush=True)
    return feats

def get_or_create_typology_features(train_df: pd.DataFrame, test_df: pd.DataFrame, accounts_df: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(TYPOLOGY_FEATS_PARQUET):
        print(f"[TYPOLOGY] Loading precomputed typology features from {TYPOLOGY_FEATS_PARQUET}")
        return pd.read_parquet(TYPOLOGY_FEATS_PARQUET)

    cols = ['timestamp', 'sender_id', 'receiver_id', 'amount', 'balance_before', 'device_id', 'device_city',
            'sim_age_hours', 'device_age_days', 'sender_account_age_days', 'login_to_txn_sec',
            'new_payee', 'pin_retries', 'ip_risk_score', 'qr_vpa_mismatch', 'txn_type', 'merchant_category']
    full_df = pd.concat([train_df[cols], test_df[cols]], axis=0, ignore_index=True)
    feats = compute_typology_and_window_features(full_df, accounts_df)
    feats.to_parquet(TYPOLOGY_FEATS_PARQUET, compression="zstd", index=False)
    print(f"[TYPOLOGY] Saved typology features to {TYPOLOGY_FEATS_PARQUET}")
    return feats

if __name__ == "__main__":
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    accounts_df = pd.read_parquet(ACCOUNTS_PARQUET)
    feats = get_or_create_typology_features(train_df, test_df, accounts_df, force=True)
    print("Typology feats sample:")
    print(feats.head(3))

