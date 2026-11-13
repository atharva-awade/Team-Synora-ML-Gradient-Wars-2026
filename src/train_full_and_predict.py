import os
import gc
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
from src.config import (
    TRAIN_PARQUET, TEST_PARQUET, CACHE_DIR, SUBMISSIONS_DIR,
    LABELS, LABEL2ID, ID2LABEL, EXPECTED_TEST_COUNTS, SEED
)

LEDGER_FEATS_PARQUET = os.path.join(CACHE_DIR, "ledger_features.parquet")
GRAPH_FEATS_PARQUET = os.path.join(CACHE_DIR, "graph_features.parquet")
TYPOLOGY_FEATS_PARQUET = os.path.join(CACHE_DIR, "typology_features.parquet")
SPECTRAL_FEATS_PARQUET = os.path.join(CACHE_DIR, "spectral_features.parquet")
TE_FEATS_PARQUET = os.path.join(CACHE_DIR, "target_encoding_features.parquet")

def assemble_master_features():
    print("[PIPELINE] Assembling 139 Master Features from all 5 feature engines...", flush=True)
    t0 = time.time()
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    n_train = len(train_df)
    n_test = len(test_df)
    n_total = n_train + n_test
    print(f"  -> Total Stream: {n_total:,} ({n_train:,} Train + {n_test:,} Test)", flush=True)

    ledger_df = pd.read_parquet(LEDGER_FEATS_PARQUET)
    graph_df = pd.read_parquet(GRAPH_FEATS_PARQUET)
    typology_df = pd.read_parquet(TYPOLOGY_FEATS_PARQUET)
    spectral_df = pd.read_parquet(SPECTRAL_FEATS_PARQUET)
    te_df = pd.read_parquet(TE_FEATS_PARQUET)

    # Basic categoricals
    full_cats = pd.concat([
        train_df[['receiver_type', 'txn_type', 'merchant_category', 'app']],
        test_df[['receiver_type', 'txn_type', 'merchant_category', 'app']]
    ], axis=0, ignore_index=True)

    cat_cols_df = pd.DataFrame(index=full_cats.index)
    for c in ['receiver_type', 'txn_type', 'merchant_category', 'app']:
        s = full_cats[c].fillna('missing').astype('category')
        cat_cols_df[f'cat_{c}'] = s.cat.codes.astype(np.int32)

    X_full = pd.concat([cat_cols_df, te_df, typology_df, ledger_df, graph_df, spectral_df], axis=1)
    X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    print(f"  -> Final Master Feature Matrix: {X_full.shape} in {time.time() - t0:.1f}s", flush=True)

    X_train = X_full.iloc[:n_train].reset_index(drop=True)
    X_test = X_full.iloc[n_train:].reset_index(drop=True)

    y_train = train_df['label_id'].values.astype(np.int8)
    timestamps_train = train_df['timestamp'].values
    test_ids = test_df['txn_id'].values
    test_timestamps = test_df['timestamp'].values
    test_devices = test_df['device_id'].values
    test_senders = test_df['sender_id'].values

    return X_train, y_train, timestamps_train, X_test, test_ids, test_timestamps, test_devices, test_senders

def run_grandmaster_pipeline():
    X_train, y_train, timestamps_train, X_test, test_ids, test_timestamps, test_devices, test_senders = assemble_master_features()
    n_train = len(X_train)
    n_test = len(X_test)

    # =========================================================================
    # STEP 1: VALIDATION & DECISION CALIBRATION ON SEP HOLDOUT
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Calibrating Per-Class Biases on Sep 2026 Holdout Fold...")
    print("=" * 70, flush=True)

    is_sep = pd.Series(timestamps_train).astype(str).str.startswith('2026-09').values
    tr_idx = np.where(~is_sep)[0]
    va_idx = np.where(is_sep)[0]

    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_val = X_train.iloc[va_idx], y_train[va_idx]

    # Fast validation slice for early stopping
    np.random.seed(SEED)
    va_slice_idx = np.random.choice(len(va_idx), size=25000, replace=False)
    X_va_slice = X_va.iloc[va_slice_idx]
    y_va_slice_bin = (y_val[va_slice_idx] > 0).astype(np.int32)

    # 1. Tier 1 Model (Binary Gate)
    y_tr_bin = (y_tr > 0).astype(np.int32)
    bin_params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'learning_rate': 0.07,
        'num_leaves': 127,
        'max_depth': 8,
        'min_child_samples': 40,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'scale_pos_weight': 4.0,
        'n_estimators': 400,
        'n_jobs': -1,
        'random_state': SEED,
        'verbose': -1
    }
    model_bin_val = lgb.LGBMClassifier(**bin_params)
    model_bin_val.fit(
        X_tr, y_tr_bin,
        eval_set=[(X_va_slice, y_va_slice_bin)],
        callbacks=[lgb.early_stopping(25, verbose=False)]
    )
    p_va_fraud = model_bin_val.predict_proba(X_va)[:, 1]

    # 2. Tier 2 Model (11-Way Conditional Fraud Disambiguator)
    fraud_tr_mask = (y_tr > 0)
    legit_tr_indices = np.where(~fraud_tr_mask)[0]
    sampled_legit = np.random.choice(legit_tr_indices, size=int(fraud_tr_mask.sum() * 1.5), replace=False)
    t2_train_indices = np.concatenate([np.where(fraud_tr_mask)[0], sampled_legit])
    np.random.shuffle(t2_train_indices)

    # Fast val slice for Tier 2
    fraud_va_mask = (y_val > 0)
    legit_va_indices = np.where(~fraud_va_mask)[0]
    sampled_va_legit = np.random.choice(legit_va_indices, size=int(fraud_va_mask.sum() * 1.5), replace=False)
    t2_va_indices = np.concatenate([np.where(fraud_va_mask)[0], sampled_va_legit])

    t2_params = {
        'objective': 'multiclass',
        'num_class': 12,
        'boosting_type': 'gbdt',
        'learning_rate': 0.07,
        'num_leaves': 75,
        'max_depth': 8,
        'min_child_samples': 25,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'class_weight': 'balanced',
        'n_estimators': 300,
        'n_jobs': -1,
        'random_state': SEED,
        'verbose': -1
    }
    model_t2_val = lgb.LGBMClassifier(**t2_params)
    model_t2_val.fit(
        X_tr.iloc[t2_train_indices], y_tr[t2_train_indices],
        eval_set=[(X_va.iloc[t2_va_indices], y_val[t2_va_indices])],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    p_va_t2 = model_t2_val.predict_proba(X_va)

    # Joint Validation Probabilities
    fraud_sum_va = np.maximum(p_va_t2[:, 1:].sum(axis=1, keepdims=True), 1e-12)
    fraud_cond_va = p_va_t2[:, 1:] / fraud_sum_va
    p_va_joint = np.zeros((len(X_va), 12), dtype=np.float32)
    p_va_joint[:, 0] = 1.0 - p_va_fraud
    p_va_joint[:, 1:] = p_va_fraud[:, None] * fraud_cond_va

    # Optimize Logit Biases for Macro-F1
    eps = 1e-7
    log_p_va = np.log(np.clip(p_va_joint, eps, 1.0))
    b = np.zeros(12, dtype=np.float64)
    for c in range(1, 12):
        b[c] = 1.4

    best_macro_f1 = f1_score(y_val, np.argmax(log_p_va + b, axis=1), average='macro')
    for step in range(3):
        improved = False
        for c in range(1, 12):
            for delta in [-0.3, -0.15, 0.15, 0.3]:
                b_cand = b.copy()
                b_cand[c] += delta
                score = f1_score(y_val, np.argmax(log_p_va + b_cand, axis=1), average='macro')
                if score > best_macro_f1:
                    best_macro_f1 = score
                    b = b_cand
                    improved = True
        if not improved:
            break

    print(f"  -> Calibrated Logit Biases on Sep Holdout: Macro-F1 = {best_macro_f1:.4f}", flush=True)

    del X_tr, y_tr, X_va, y_val, model_bin_val, model_t2_val, p_va_joint, log_p_va
    gc.collect()

    # =========================================================================
    # STEP 2: FULL DATA RETRAINING (JUL + AUG + SEP = 1.56 MILLION ROWS)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Retraining on FULL 1,563,913 Samples (Jul + Aug + Sep)...")
    print("=" * 70, flush=True)

    # 1. Full Tier 1 Training
    y_full_bin = (y_train > 0).astype(np.int32)
    bin_params['n_estimators'] = 380
    full_bin_model = lgb.LGBMClassifier(**bin_params)
    full_bin_model.fit(X_train, y_full_bin)
    p_te_fraud = full_bin_model.predict_proba(X_test)[:, 1]
    print(f"  -> Full Tier 1 Model Fitted on {len(X_train):,} samples", flush=True)

    # 2. Full Tier 2 Training
    fraud_full_mask = (y_train > 0)
    legit_full_indices = np.where(~fraud_full_mask)[0]
    sampled_full_legit = np.random.choice(legit_full_indices, size=int(fraud_full_mask.sum() * 1.5), replace=False)
    t2_full_train_indices = np.concatenate([np.where(fraud_full_mask)[0], sampled_full_legit])
    np.random.shuffle(t2_full_train_indices)

    t2_params['n_estimators'] = 280
    full_t2_model = lgb.LGBMClassifier(**t2_params)
    full_t2_model.fit(X_train.iloc[t2_full_train_indices], y_train[t2_full_train_indices])
    p_te_t2 = full_t2_model.predict_proba(X_test)
    print(f"  -> Full Tier 2 Model Fitted on {len(t2_full_train_indices):,} samples", flush=True)

    # 3. Full Joint Probabilities on Test
    fraud_sum_te = np.maximum(p_te_t2[:, 1:].sum(axis=1, keepdims=True), 1e-12)
    fraud_cond_te = p_te_t2[:, 1:] / fraud_sum_te
    p_te_joint = np.zeros((n_test, 12), dtype=np.float32)
    p_te_joint[:, 0] = 1.0 - p_te_fraud
    p_te_joint[:, 1:] = p_te_fraud[:, None] * fraud_cond_te

    # =========================================================================
    # STEP 3: LOGIT BIASING & EPISODE BURST CONSENSUS POOLING
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Applying Calibrated Decision Layer & Episode Pooling...")
    print("=" * 70, flush=True)

    log_p_te = np.log(np.clip(p_te_joint, eps, 1.0))
    # Apply calibrated b shrunk slightly by 5%
    test_scaled_logits = log_p_te + (b * 0.95)
    raw_test_preds = np.argmax(test_scaled_logits, axis=1)

    # Episode Clustering by Device within 5 minutes
    ts_test = pd.to_datetime(test_timestamps)
    time_sec_test = (ts_test.astype("int64") // 10**9).values

    dev_str = test_devices.astype(str)
    codes, _ = pd.factorize(dev_str)
    BIG = 20_000_000
    comp_key = codes * BIG + time_sec_test
    order = np.argsort(comp_key)
    sorted_key = comp_key[order]
    sorted_codes = codes[order]
    sorted_preds = raw_test_preds[order]
    sorted_p_fraud = p_te_fraud[order]

    # Cluster consecutive rows on same device with gap <= 300s
    time_diff = np.diff(time_sec_test[order], prepend=time_sec_test[order][0])
    code_diff = np.diff(sorted_codes, prepend=sorted_codes[0])
    is_new_cluster = (code_diff != 0) | (time_diff > 300)
    cluster_ids = np.cumsum(is_new_cluster)

    # Burst Homophily Smoothing:
    # If a cluster has size >= 3 and any member has high fraud confidence, assign consensus
    cluster_series = pd.Series(cluster_ids)
    cluster_sizes = cluster_series.map(cluster_series.value_counts()).values

    pooled_preds = sorted_preds.copy()
    burst_mask = (cluster_sizes >= 3) & (cluster_sizes <= 60)

    # Inside bursts, if max fraud prob >= 0.50, pool to majority non-legit prediction
    for cid in np.unique(cluster_ids[burst_mask]):
        c_mask = (cluster_ids == cid)
        c_p = sorted_p_fraud[c_mask]
        if np.max(c_p) >= 0.45:
            # find most confident fraud class in this burst
            fraud_candidates = sorted_preds[c_mask]
            non_legit = fraud_candidates[fraud_candidates > 0]
            if len(non_legit) > 0:
                majority_fraud = pd.Series(non_legit).mode()[0]
                pooled_preds[c_mask] = majority_fraud

    final_preds = np.empty(n_test, dtype=np.int32)
    final_preds[order] = pooled_preds

    # Map integer IDs to canonical strings
    final_labels = [ID2LABEL[p] for p in final_preds]

    out_path = os.path.join(SUBMISSIONS_DIR, "submission_vortex12_grandmaster_top1.csv")
    pd.DataFrame({'id': test_ids, 'label': final_labels}).to_csv(out_path, index=False)

    print(f"\n[SUCCESS] Generated Grandmaster #1 Submission: {out_path}")
    print(f"  -> Total Rows: {len(final_labels):,}")
    print("\n--- Final Test Predicted Class Distribution ---")
    counts = pd.Series(final_labels).value_counts()
    for lbl in LABELS:
        print(f"  {lbl:<20}: {counts.get(lbl, 0):>6} rows (Expected prior ~{EXPECTED_TEST_COUNTS.get(lbl, 0):>6})")

    return out_path

if __name__ == "__main__":
    run_grandmaster_pipeline()

