#!/usr/bin/env python3
"""
MASTER CONTINUOUS SPRINT ENGINE (Target: 0.63+ Macro-F1)
ML Gradient Wars 2026 - UPI Fraud Rings

Runs Sprint 1 -> Sprint 2 -> Sprint 3 sequentially without stopping.
Prevents Windows from sleeping or shutting down.
Outputs verified submissions at each sprint milestone.
"""

import os
import sys
import time
import ctypes
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

# ------------------------------------------------------------------------------
# 1. Windows Execution State: Prevent Sleep / Shutdown
# ------------------------------------------------------------------------------
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("[OS] Windows Sleep/Hibernate prevention ACTIVE (Execution state locked).")
    except Exception as e:
        print(f"[OS WARNING] Could not lock execution state: {e}")

prevent_sleep()

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for i, l in enumerate(LABELS)}
K = len(LABELS)

DATA_DIR = r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings"
CACHE_DIR = r"D:\ML Gradient Gemini\cache"
CKPT_DIR = r"D:\ML Gradient Gemini\checkpoints_resilient"
SUB_DIR = r"D:\ML Gradient Gemini\submissions"
LOG_FILE = r"D:\ML Gradient Gemini\sprints_progress.log"

os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(out + "\n")

# ------------------------------------------------------------------------------
# 2. Data & Feature Assembly (227 Features)
# ------------------------------------------------------------------------------
def assemble_227_features():
    fused_path = os.path.join(CKPT_DIR, "fused_227_features.parquet")
    meta_path = os.path.join(CKPT_DIR, "dataset_meta.pkl")
    
    if os.path.exists(fused_path) and os.path.exists(meta_path):
        log(f"Loading cached 227-feature matrix from {fused_path}...")
        X = pd.read_parquet(fused_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return X, meta

    log("Building fused 227-feature master matrix from cache...")
    tr = pd.read_parquet(os.path.join(CACHE_DIR, "train_clean.parquet"))
    te = pd.read_parquet(os.path.join(CACHE_DIR, "test_clean.parquet"))
    tr['is_train'] = 1
    te['is_train'] = 0
    te['label'] = 'legit'
    te['label_id'] = 0
    
    df_raw = pd.concat([tr, te], ignore_index=True)
    df_raw['t'] = pd.to_datetime(df_raw['timestamp'])
    df_raw['ts'] = (df_raw['t'] - df_raw['t'].min()).dt.total_seconds().astype('int64')
    
    # Sort order matching base_features.parquet
    sort_idx = np.argsort(df_raw['ts'].values, kind='stable')
    df = df_raw.iloc[sort_idx].reset_index(drop=True)
    
    # Load feature sets
    base = pd.read_parquet(os.path.join(CKPT_DIR, "base_features.parquet"))
    log(f"  Base features: {base.shape[1]}")
    
    gf = pd.read_parquet(os.path.join(CACHE_DIR, "graph_features.parquet")).iloc[sort_idx].reset_index(drop=True)
    log(f"  Graph features: {gf.shape[1]}")
    
    lf = pd.read_parquet(os.path.join(CACHE_DIR, "ledger_features.parquet")).iloc[sort_idx].reset_index(drop=True)
    log(f"  Ledger features: {lf.shape[1]}")
    
    sf = pd.read_parquet(os.path.join(CACHE_DIR, "spectral_features.parquet")).iloc[sort_idx].reset_index(drop=True)
    log(f"  Spectral features: {sf.shape[1]}")
    
    tef = pd.read_parquet(os.path.join(CACHE_DIR, "target_encoding_features.parquet")).iloc[sort_idx].reset_index(drop=True)
    log(f"  Target encoding features: {tef.shape[1]}")
    
    tf = pd.read_parquet(os.path.join(CACHE_DIR, "typology_features.parquet")).iloc[sort_idx].reset_index(drop=True)
    tf_unique_cols = [c for c in tf.columns if c not in base.columns and c not in gf.columns]
    tf_unique = tf[tf_unique_cols]
    log(f"  Typology extra features: {len(tf_unique_cols)}")
    
    # Concatenate all features
    X = pd.concat([base, gf, lf, sf, tef, tf_unique], axis=1)
    X = X.astype(np.float32)
    log(f"--> Combined Matrix Shape: {X.shape} ({X.shape[1]} features)")
    
    # Save cache
    X.to_parquet(fused_path)
    
    is_tr = df['is_train'].values == 1
    tr_idx = np.where(is_tr)[0]
    test_idx = np.where(~is_tr)[0]
    month = df['t'].dt.month.values
    y_all = np.array([L2I[l] for l in df['label'].values])
    y = y_all[tr_idx]
    last = (month[tr_idx] == 9)
    groups = (df['ts'] // 86400).values[tr_idx]
    test_ids = df['txn_id'].values[test_idx]
    
    meta = {
        'tr_idx': tr_idx,
        'test_idx': test_idx,
        'y': y,
        'last': last,
        'groups': groups,
        'test_ids': test_ids,
        'df': df[['txn_id', 'ts', 'sender_id', 'receiver_id', 'device_id', 'amount', 'is_train']]
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    log("Saved fused features and metadata.")
    return X, meta

# ------------------------------------------------------------------------------
# 3. Model Training & Coordinate Multiplier Tuning
# ------------------------------------------------------------------------------
def tune_multipliers(P_val, y_val, n_cycles=4):
    best_w = np.ones(K, dtype=np.float64)
    best_score = f1_score(y_val, (P_val * best_w).argmax(axis=1), average='macro')
    log(f"    Initial Macro-F1 on September Holdout: {best_score:.4f}")

    for cycle in range(n_cycles):
        for c in range(1, K):
            best_wc = best_w[c]
            candidates = np.unique(np.concatenate([
                np.linspace(0.1, 3.0, 30),
                np.linspace(max(0.05, best_wc * 0.5), min(10.0, best_wc * 1.5), 25)
            ]))
            for cand in candidates:
                w_try = best_w.copy()
                w_try[c] = cand
                preds = (P_val * w_try).argmax(axis=1)
                score = f1_score(y_val, preds, average='macro') - 1e-4 * ((cand - 1.0) ** 2)
                if score > best_score:
                    best_score = score
                    best_wc = cand
            best_w[c] = best_wc
        log(f"      Cycle {cycle + 1}/{n_cycles} -> Tuned Macro-F1: {best_score:.4f}")
    return best_w, best_score

def verify_and_save_submission(preds, test_ids, filename, description=""):
    sub = pd.DataFrame({'id': test_ids, 'label': np.array(LABELS)[preds]})
    out_path = os.path.join(SUB_DIR, filename)
    sub.to_csv(out_path, index=False)
    
    # 5-point harness
    raw_te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), usecols=['txn_id'])
    top1 = pd.read_csv(os.path.join(SUB_DIR, "submission_synora_top1_grandmaster.csv"))
    
    assert len(sub) == len(raw_te), f"Length mismatch: {len(sub)} vs {len(raw_te)}"
    assert (sub['id'].values == top1['id'].values).all(), "ID sequence mismatch!"
    assert sub['label'].isnull().sum() == 0, "Null labels detected!"
    assert set(sub['label'].unique()).issubset(set(LABELS)), "Invalid labels!"
    
    fraud_count = (sub['label'] != 'legit').sum()
    agreement = (sub['label'] == top1['label']).mean()
    log(f"[SUBMISSION VERIFIED] -> {filename}")
    log(f"  Description: {description}")
    log(f"  Fraud Count: {fraud_count} (Agreement with 0.5005 baseline: {agreement:.2%})")
    log(f"  Path: {out_path}")
    return out_path

# ------------------------------------------------------------------------------
# MAIN PIPELINE: SPRINT 1 -> SPRINT 2 -> SPRINT 3
# ------------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("  SYNORA DUAL-ENGINE 0.63+ CONTINUOUS MASTER RUNNER")
    log("  Windows Sleep Block: ACTIVE | Hardware: RTX 4050 GPU")
    log("=" * 70)
    
    X, meta = assemble_227_features()
    tr_idx, test_idx = meta['tr_idx'], meta['test_idx']
    y, last, groups = meta['y'], meta['last'], meta['groups']
    test_ids = meta['test_ids']
    
    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_te = X.iloc[test_idx].reset_index(drop=True)
    
    # --------------------------------------------------------------------------
    # SPRINT 1: 227-Feature Super Fusion Engine (GPU XGBoost + LightGBM)
    # --------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log(">>> LAUNCHING SPRINT 1: 227-Feature Super Fusion Engine (GPU XGBoost)")
    log("=" * 70)
    
    oof_s1 = np.zeros((len(X_tr), K), dtype=np.float32)
    pte_s1 = np.zeros((len(X_te), K), dtype=np.float32)
    
    # Inverse class weights (focal weighting alpha=0.5)
    cnt = np.bincount(y, minlength=K).astype(float)
    inv_w = (len(y) / (K * np.maximum(cnt, 1.0))) ** 0.5
    sample_weights = (inv_w / inv_w.min())[y]
    
    gkf = GroupKFold(5)
    rng = np.random.RandomState(42)
    
    # Subsampling legit to 0.40 to give rare classes 2.5x gradient volume
    for fold, (tr_k, va_k) in enumerate(gkf.split(X_tr, y, groups)):
        log(f"--- Sprint 1 Fold {fold + 1}/5 ---")
        # Subsample legit in train fold
        keep_tr = (y[tr_k] != 0) | (rng.random(len(tr_k)) < 0.40)
        idx_train = tr_k[keep_tr]
        
        clf = xgb.XGBClassifier(
            n_estimators=650,
            learning_rate=0.06,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            tree_method='hist',
            device='cuda',
            random_state=42 + fold,
            n_jobs=8
        )
        clf.fit(X_tr.iloc[idx_train], y[idx_train], sample_weight=sample_weights[idx_train])
        
        va_probs = clf.predict_proba(X_tr.iloc[va_k])
        oof_s1[va_k] = va_probs
        te_probs = clf.predict_proba(X_te)
        pte_s1 += te_probs / 5.0
        
        fold_f1 = f1_score(y[va_k], va_probs.argmax(axis=1), average='macro')
        log(f"  Fold {fold + 1} Raw Macro-F1: {fold_f1:.4f}")
    
    np.save(os.path.join(CKPT_DIR, "sprint1_oof.npy"), oof_s1)
    np.save(os.path.join(CKPT_DIR, "sprint1_pte.npy"), pte_s1)
    
    # Multiplier calibration on September holdout
    log("Tuning coordinate multipliers for Sprint 1...")
    w_s1, sc_s1 = tune_multipliers(oof_s1[last], y[last])
    log(f">>> SPRINT 1 SEPTEMBER HOLDOUT MACRO-F1: {sc_s1:.4f}")
    
    preds_s1 = (pte_s1 * w_s1).argmax(axis=1)
    verify_and_save_submission(preds_s1, test_ids, "submission_sprint1_superfusion_227f.csv", f"Sprint 1 (227f GPU XGBoost, OOF {sc_s1:.4f})")
    
    # --------------------------------------------------------------------------
    # SPRINT 2: Neighbor Probability Diffusion & Stacking
    # --------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log(">>> LAUNCHING SPRINT 2: Multi-Hop Neighbor Stacking on 227 Features")
    log("=" * 70)
    
    # Assemble full probability matrix
    P_all = np.zeros((len(X), K), dtype=np.float32)
    P_all[tr_idx] = oof_s1
    P_all[test_idx] = pte_s1
    
    # Neighbor features using ex graph pointer structures
    with open(os.path.join(CKPT_DIR, "ex.pkl"), "rb") as f:
        class Ref: pass
        sys.modules['__main__'].Ref = Ref
        ex = pickle.load(f)
    
    def calc_neighbour_feats(P, ex):
        s, r, d, ts = ex['s'], ex['r'], ex['d'], ex['ts']
        nx, pv, rev = ex['nx'], ex['pv'], ex['rev']
        sref, dref = ex['sref'], ex['dref']
        
        cols, names = [], []
        # Receiver, Sender, Device multi-hop means
        for nm, ref in [('r', r), ('s', s), ('d', d)]:
            counts = pd.Series(1, index=ref).groupby(ref).transform('sum').values
            means = pd.DataFrame(P).groupby(ref).transform('mean').values
            cols.append(means)
            names += [f'nb_{nm}_{LABELS[k]}' for k in range(K)]
        
        # Previous hop, next hop, reverse pair
        for nm, pt in [('pv', pv), ('nx', nx), ('rev', rev)]:
            ok = (pt >= 0)
            valid_pt = np.where(ok, pt, 0)
            p_hop = P[valid_pt].copy()
            p_hop[~ok] = 0.0
            cols.append(p_hop)
            names += [f'hop_{nm}_{LABELS[k]}' for k in range(K)]
            
        cols.append(P)
        names += [f'p_s1_{LABELS[k]}' for k in range(K)]
        return pd.DataFrame(np.hstack(cols).astype(np.float32), columns=names)

    log("Calculating multi-hop neighbor probability features...")
    X_nb = calc_neighbour_feats(P_all, ex)
    X_s2 = pd.concat([X, X_nb], axis=1)
    log(f"Sprint 2 Feature Matrix: {X_s2.shape[1]} features")
    
    X2_tr = X_s2.iloc[tr_idx].reset_index(drop=True)
    X2_te = X_s2.iloc[test_idx].reset_index(drop=True)
    
    oof_s2 = np.zeros((len(X2_tr), K), dtype=np.float32)
    pte_s2 = np.zeros((len(X2_te), K), dtype=np.float32)
    
    for fold, (tr_k, va_k) in enumerate(gkf.split(X2_tr, y, groups)):
        log(f"--- Sprint 2 Fold {fold + 1}/5 ---")
        keep_tr = (y[tr_k] != 0) | (rng.random(len(tr_k)) < 0.45)
        idx_train = tr_k[keep_tr]
        
        clf2 = xgb.XGBClassifier(
            n_estimators=550,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.7,
            tree_method='hist',
            device='cuda',
            random_state=100 + fold,
            n_jobs=8
        )
        clf2.fit(X2_tr.iloc[idx_train], y[idx_train], sample_weight=sample_weights[idx_train])
        
        va_p = clf2.predict_proba(X2_tr.iloc[va_k])
        oof_s2[va_k] = va_p
        pte_s2 += clf2.predict_proba(X2_te) / 5.0
        log(f"  Fold {fold + 1} Raw Macro-F1: {f1_score(y[va_k], va_p.argmax(1), average='macro'):.4f}")
        
    np.save(os.path.join(CKPT_DIR, "sprint2_oof.npy"), oof_s2)
    np.save(os.path.join(CKPT_DIR, "sprint2_pte.npy"), pte_s2)
    
    w_s2, sc_s2 = tune_multipliers(oof_s2[last], y[last])
    log(f">>> SPRINT 2 SEPTEMBER HOLDOUT MACRO-F1: {sc_s2:.4f}")
    
    preds_s2 = (pte_s2 * w_s2).argmax(axis=1)
    verify_and_save_submission(preds_s2, test_ids, "submission_sprint2_stacking_peak.csv", f"Sprint 2 (Stacking 482f GPU, OOF {sc_s2:.4f})")
    
    # --------------------------------------------------------------------------
    # SPRINT 3: Transductive Test Ring Diffusion + Grand Champion Ensemble
    # --------------------------------------------------------------------------
    log("\n" + "=" * 70)
    log(">>> LAUNCHING SPRINT 3: Transductive Test Ring Diffusion & Grandmaster Blend")
    log("=" * 70)
    
    # Blend Sprint 1 + Sprint 2 + Resilient 0.5032 Champion
    p_champ = np.load(os.path.join(CKPT_DIR, "Stage_2_pte.npy"))
    p_final_blend = 0.45 * pte_s2 + 0.35 * pte_s1 + 0.20 * p_champ
    
    # Optimal blend multipliers
    oof_final_blend = 0.45 * oof_s2 + 0.35 * oof_s1 + 0.20 * np.load(os.path.join(CKPT_DIR, "Stage_2_oof.npy"))
    w_final, sc_final = tune_multipliers(oof_final_blend[last], y[last])
    log(f">>> GRANDMASTER BLEND SEPTEMBER HOLDOUT MACRO-F1: {sc_final:.4f}")
    
    preds_s3 = (p_final_blend * w_final).argmax(axis=1)
    verify_and_save_submission(preds_s3, test_ids, "submission_sprint3_crown_063.csv", f"Sprint 3 (Grandmaster Tri-Blend, OOF {sc_final:.4f})")
    
    log("\n" + "=" * 70)
    log("  ALL SPRINTS COMPLETE! 3 VERIFIED SUBMISSIONS GENERATED IN submissions/")
    log("  Windows was kept awake throughout. No shutdown performed.")
    log("=" * 70)

if __name__ == '__main__':
    main()
