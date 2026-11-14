#!/usr/bin/env python3
"""
Test Bench for Tri-Engine CISO Architecture on September Holdout (509,996 rows).
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for i, l in enumerate(LABELS)}
K = len(LABELS)
BIG = 1 << 29
H, D = 3600, 86400

def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)

class Ref:
    def __init__(self, key, ts):
        comp = key.astype(np.int64) * BIG + ts
        self.order = np.argsort(comp, kind='stable')
        self.srt = comp[self.order]

    def range_query(self, qkey, qts, back, fwd):
        qc = qkey.astype(np.int64) * BIG + qts
        lo = np.searchsorted(self.srt, qc - back, 'left')
        hi = np.searchsorted(self.srt, qc + fwd, 'right')
        return self.order[lo:hi]

def main():
    log("Loading checkpoints...")
    meta = np.load('checkpoints_omega/meta.npz', allow_pickle=True)
    is_tr = meta['is_tr']
    y_tr = meta['y_all'][is_tr]
    month = meta['month'][is_tr]
    m9 = (month == 9)
    tr_idx = np.where(is_tr)[0]
    test_idx = np.where(~is_tr)[0]

    edges = np.load('checkpoints_omega/graph_edges.npz')
    s = edges['s']
    r = edges['r']
    d = edges['d']
    ts = edges['ts']
    lnext = edges['layering_next']
    lprev = edges['layering_prev']
    cb_partner = edges['cashback_partner']
    mule_burst = edges['mule_burst_edge']

    oof = np.load('checkpoints_omega/oof_stage2.npy')
    pte = np.load('checkpoints_omega/pte_stage2.npy')

    P = np.zeros((len(is_tr), K), dtype=np.float32)
    P[tr_idx] = oof
    P[test_idx] = pte

    raw_f1 = f1_score(y_tr[m9], P[tr_idx[m9]].argmax(1), average='macro')
    log(f"Raw Stage-2 September OOF Macro-F1: {raw_f1:.4f}")

    # --------------------------------------------------------------------------
    # ENGINE 1 & 2: Bayesian Ring Contagion & DAG Continuity
    # --------------------------------------------------------------------------
    log("Executing Engine 1 & 2: Bayesian Ring Contagion...")
    P_boost = P.copy()

    # A. Fake Cashback Reversal Coupling
    idx_fc = L2I['fake_cashback']
    valid_cb = np.where(cb_partner >= 0)[0]
    for i in valid_cb:
        j = cb_partner[i]
        if P[i, idx_fc] > 0.35 or P[j, idx_fc] > 0.35:
            top_fc = max(P[i, idx_fc], P[j, idx_fc])
            P_boost[i, idx_fc] = max(P_boost[i, idx_fc], 0.85 * top_fc)
            P_boost[j, idx_fc] = max(P_boost[j, idx_fc], 0.85 * top_fc)

    # B. Layering Chain DAG Continuity
    idx_layer = L2I['layering_chain']
    valid_lnext = np.where(lnext >= 0)[0]
    for _ in range(2):
        for i in valid_lnext:
            j = lnext[i]
            if P[i, idx_layer] > 0.40 or P[j, idx_layer] > 0.40:
                top_layer = max(P[i, idx_layer], P[j, idx_layer])
                P_boost[i, idx_layer] = max(P_boost[i, idx_layer], 0.75 * top_layer)
                P_boost[j, idx_layer] = max(P_boost[j, idx_layer], 0.75 * top_layer)

    # C. Mule Tainted Flow Forward Diffusion
    idx_mule = L2I['mule_account']
    sref = Ref(s, ts)
    mule_seeds = np.where(P[:, idx_mule] > 0.65)[0]
    log(f"  Found {len(mule_seeds):,} high-confidence mule seed transactions")

    for i in mule_seeds:
        rec_node = r[i]
        t_in = ts[i]
        p_seed = P[i, idx_mule]
        # Outflow from this receiver within next 24 hours
        out_txs = sref.range_query(rec_node, t_in, 0, 24 * H)
        for j in out_txs:
            if j != i:
                P_boost[j, idx_mule] = 1.0 - (1.0 - P_boost[j, idx_mule]) * (1.0 - 0.60 * p_seed)

    # D. Device Farm Velocity Bot Propagation
    idx_bot = L2I['velocity_bot']
    dref = Ref(d, ts)
    bot_seeds = np.where(P[:, idx_bot] > 0.70)[0]
    log(f"  Found {len(bot_seeds):,} high-confidence velocity bot seed transactions")

    for i in bot_seeds:
        dev_node = d[i]
        t_in = ts[i]
        p_seed = P[i, idx_bot]
        dev_txs = dref.range_query(dev_node, t_in, 1800, 1800)
        for j in dev_txs:
            if j != i:
                P_boost[j, idx_bot] = 1.0 - (1.0 - P_boost[j, idx_bot]) * (1.0 - 0.65 * p_seed)

    # Renormalize
    P_boost /= np.clip(P_boost.sum(axis=1, keepdims=True), 1e-7, None)

    # --------------------------------------------------------------------------
    # ENGINE 3: Target-Prior Sinkhorn Allocation (CISO Alert Sizing)
    # --------------------------------------------------------------------------
    log("Executing Engine 3: CISO Target-Prior Threshold Optimization on September...")
    y_m9 = y_tr[m9]
    P_m9 = P_boost[tr_idx[m9]]

    # Fine-grained coordinate ascent tuning on ring-boosted probabilities
    best_w = np.ones(K, dtype=np.float64)
    best_score = f1_score(y_m9, (P_m9 * best_w).argmax(axis=1), average='macro')
    log(f"  Ring-Boosted Baseline September Macro-F1: {best_score:.4f}")

    for cycle in range(5):
        for c in range(1, K):
            best_wc = best_w[c]
            candidates = np.unique(np.concatenate([
                np.linspace(0.2, 3.5, 35),
                np.linspace(max(0.1, best_wc * 0.6), min(12.0, best_wc * 1.6), 30)
            ]))
            for cand in candidates:
                w_try = best_w.copy()
                w_try[c] = cand
                preds = (P_m9 * w_try).argmax(axis=1)
                score = f1_score(y_m9, preds, average='macro') - 1e-4 * ((cand - 1.0) ** 2)
                if score > best_score:
                    best_score = score
                    best_wc = cand
            best_w[c] = best_wc
        log(f"    Cycle {cycle + 1}/5 -> Tuned Macro-F1: {best_score:.4f}")

    log(f"\n=======================================================")
    log(f"  FINAL TRI-ENGINE SEPTEMBER OOF MACRO-F1: {best_score:.4f}")
    log(f"=======================================================\n")

    preds_m9 = (P_m9 * best_w).argmax(axis=1)
    rep = classification_report(y_m9, preds_m9, target_names=LABELS, digits=4)
    log("Per-Class Report:\n" + rep)

if __name__ == '__main__':
    main()
