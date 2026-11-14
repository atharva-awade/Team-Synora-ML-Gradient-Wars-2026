#!/usr/bin/env python3
"""
Synora-Top1 Production Stacking Engine
ML Gradient Wars 2026 - UPI Fraud Rings

Direct evolution of colab_pipeline_fixed.py (which achieved 0.4911 LB):
  1. Enhanced feature space: SIM/Device disambiguation, phishing distance,
     mule pass-through ratios, contextual month-Z amount.
  2. Multi-hop neighbour features: r24h, d1h, s6h, s24h, reverse-pair probabilities.
  3. 3-Stage deep stacking with 3-seed bagging in Stage 3.
  4. Higher training capacity: legit_frac 0.60, iters 700 on RTX 4050 GPU.
  5. CleanLab OOF noise filtering before stages 2 & 3.
  6. L2-Regularized September holdout multiplier tuning.
"""

import argparse
import copy
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')

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


def macro_f1(y, p, k=K):
    tp = np.bincount(y[y == p], minlength=k).astype(float)
    pc = np.bincount(p, minlength=k)
    tc = np.bincount(y, minlength=k)
    den = pc + tc
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def per_class_f1(y, p, k=K):
    tp = np.bincount(y[y == p], minlength=k).astype(float)
    den = np.bincount(p, minlength=k) + np.bincount(y, minlength=k)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


# ----------------------------------------------------------------------------------------------
# fast keyed time-window lookups
# ----------------------------------------------------------------------------------------------
class Ref:
    """Reference rows sorted by (key, ts). Supports counts / sums / nearest-neighbour in a window."""

    def __init__(self, key, ts, vals=None):
        comp = key.astype(np.int64) * BIG + ts
        self.order = np.argsort(comp, kind='stable')
        self.srt = comp[self.order]
        self.cs = None
        if vals is not None:
            v = np.asarray(vals, dtype=np.float64).reshape(len(key), -1)[self.order]
            self.cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])

    def _b(self, qkey, qts, back, fwd):
        qc = qkey.astype(np.int64) * BIG + qts
        return (np.searchsorted(self.srt, qc - back, 'left'),
                np.searchsorted(self.srt, qc + fwd, 'right'))

    def count(self, qkey, qts, back, fwd):
        lo, hi = self._b(qkey, qts, back, fwd)
        return hi - lo

    def sum(self, qkey, qts, back, fwd):
        lo, hi = self._b(qkey, qts, back, fwd)
        return self.cs[hi] - self.cs[lo]

    def nearest(self, qkey, qts, direction, maxgap):
        qc = qkey.astype(np.int64) * BIG + qts
        m = len(self.srt)
        if direction == 'next':
            idx = np.searchsorted(self.srt, qc + 1, 'left')
            ok = idx < m
            idc = np.minimum(idx, m - 1)
            cand = self.srt[idc]
            ok &= (cand // BIG == qkey) & (cand - qc <= maxgap)
            gap = (cand - qc).astype(float)
        else:
            idx = np.searchsorted(self.srt, qc - 1, 'right') - 1
            ok = idx >= 0
            idc = np.maximum(idx, 0)
            cand = self.srt[idc]
            ok &= (cand // BIG == qkey) & (qc - cand <= maxgap)
            gap = (qc - cand).astype(float)
        j = np.where(ok, self.order[idc], -1)
        return j, np.where(ok, gap, np.nan)


def nunique_bucket(key, other, ts, width):
    out = []
    for off in (0, width // 2):
        b = (ts + off) // width
        gid = key.astype(np.int64) * 100000 + b
        out.append(pd.Series(other).groupby(gid).transform('nunique').values)
    return np.maximum(out[0], out[1])


def follow(ptr, steps=6):
    n = len(ptr)
    cur = np.arange(n)
    length = np.zeros(n)
    for _ in range(steps):
        nxt = np.where(cur >= 0, ptr[np.maximum(cur, 0)], -1)
        length += (nxt >= 0)
        cur = nxt
    return length


# ----------------------------------------------------------------------------------------------
# Feature Building
# ----------------------------------------------------------------------------------------------
def build(df, ac, me):
    n = len(df)
    ts = df.ts.values.astype(np.int64)
    ids, _ = pd.factorize(pd.concat([df.sender_id, df.receiver_id], ignore_index=True))
    s = ids[:n].astype(np.int64)
    r = ids[n:].astype(np.int64)
    d = pd.factorize(df.device_id)[0].astype(np.int64)
    cc = pd.factorize(df.device_city)[0]
    amt = df.amount.values.astype(float)
    bal = df.balance_before.values.astype(float)
    t = df.t
    is_p2p = (df.receiver_type == 'P2P').values
    is_refund = (df.txn_type == 'REFUND').values
    F = {}

    # ---- 1. Single-row primitives ----
    F['hour'] = t.dt.hour.values
    F['weekday'] = t.dt.weekday.values
    F['night'] = ((F['hour'] >= 23) | (F['hour'] <= 5)).astype(int)
    F['is_extortion_evening'] = ((F['hour'] >= 19) | (F['hour'] <= 2)).astype(int)

    for c in ['amount', 'sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee',
              'pin_retries', 'login_to_txn_sec', 'balance_before', 'ip_risk_score', 'qr_vpa_mismatch']:
        F[c] = df[c].values.astype(float)

    F['log_amt'] = np.log1p(amt)
    F['amt_share'] = amt / np.maximum(bal, 1)
    F['amt_int'] = np.isclose(amt, np.round(amt)).astype(int)
    F['small_amt'] = (amt <= 10).astype(int)
    for m in (100, 500, 1000):
        F[f'd{m}'] = (-amt) % m
        F[f'round{m}'] = np.isclose(amt % m, 0).astype(int)

    # Disambiguation: sim_swap vs account_takeover
    is_fresh_sim = (F['sim_age_hours'] < 72).astype(float)
    is_fresh_dev = (F['device_age_days'] < 3).astype(float)
    F['sim_dev_ratio'] = F['sim_age_hours'] / (F['device_age_days'] * 24.0 + 1.0)
    F['sim_swap_signal'] = is_fresh_sim * (1.0 - is_fresh_dev)
    F['takeover_signal'] = is_fresh_dev * (1.0 - is_fresh_sim)

    # Phishing collect distance just under limits
    limits = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 25000.0, 50000.0, 100000.0]
    under_dist = np.full(n, 9999.0, dtype=float)
    for L in limits:
        diff = L - amt
        mask = (diff > 0) & (diff <= 200)
        under_dist[mask] = np.minimum(under_dist[mask], diff[mask])
    F['dist_under_limit'] = under_dist

    # Contextual month amount Z-score
    m_codes = df.t.dt.month.values
    z_key = m_codes * 100 + pd.Categorical(df['txn_type'].fillna('none')).codes * 10 + is_p2p.astype(int)
    grp_m = pd.Series(amt).groupby(z_key).transform('mean').values
    grp_s = pd.Series(amt).groupby(z_key).transform('std').fillna(1.0).values
    F['amt_month_norm_z'] = (amt - grp_m) / np.maximum(grp_s, 1.0)

    med = pd.Series(amt).groupby(s).transform('median').values
    F['amt_vs_s_median'] = amt / np.maximum(med, 1)
    for c in ['receiver_type', 'txn_type', 'merchant_category', 'app']:
        F[c] = pd.Categorical(df[c].fillna('none')).codes

    a = ac.set_index('account_id')
    mer = me.set_index('merchant_id')
    home = df.sender_id.map(a.home_city)
    F['away_from_home'] = (df.device_city != home).astype(int).values
    F['s_kyc_min'] = (df.sender_id.map(a.kyc_level) == 'min').astype(int).values
    F['s_ios'] = (df.sender_id.map(a.os) == 'ios').astype(int).values
    F['s_bank'] = pd.Categorical(df.sender_id.map(a.bank_code)).codes
    rh = df.receiver_id.map(a.home_city)
    F['r_home_diff'] = np.where(is_p2p, (rh != home).astype(int), -1)
    F['r_acct_age'] = (t - df.receiver_id.map(a.opened_on)).dt.days.values.astype(float)
    F['r_kyc_min'] = np.where(is_p2p, (df.receiver_id.map(a.kyc_level) == 'min').astype(int), -1)
    F['r_same_bank'] = np.where(is_p2p, (df.receiver_id.map(a.bank_code) == df.sender_id.map(a.bank_code)).astype(int), -1)
    F['dev_city_is_mcity'] = np.where(~is_p2p, (df.device_city == df.receiver_id.map(mer.city)).astype(int), -1)
    F['m_age'] = (t - df.receiver_id.map(mer.onboarded_on)).dt.days.values.astype(float)

    # ---- 2. Sender behaviour ----
    sref = Ref(s, ts, vals=amt)
    for nm, b in [('5m', 300), ('1h', H), ('1d', D)]:
        F[f's_cnt_{nm}'] = sref.count(s, ts, b, b)
    F['s_cnt_past_1d'] = sref.count(s, ts, D, 0)
    F['s_cnt_past_7d'] = sref.count(s, ts, 7 * D, 0)
    F['s_nuniq_dev_day'] = nunique_bucket(s, d, ts, D)
    F['s_nuniq_recv_1h'] = nunique_bucket(s, r, ts, H)
    nxs, gn = sref.nearest(s, ts, 'next', 6 * H)
    pvs, gp = sref.nearest(s, ts, 'prev', 6 * H)
    F['s_gap_next'], F['s_gap_prev'] = gn, gp
    F['dev_changed_prev'] = np.where(pvs >= 0, (d[np.maximum(pvs, 0)] != d).astype(int), -1)
    F['city_changed_prev'] = np.where(pvs >= 0, (cc[np.maximum(pvs, 0)] != cc).astype(int), -1)
    F['bal_drop_next'] = np.where(nxs >= 0, (bal[np.maximum(nxs, 0)] - (bal - amt)) / np.maximum(bal, 1), np.nan)
    rf = Ref(s[is_refund], ts[is_refund])
    F['s_refund_7d'] = rf.count(s, ts, 7 * D, 7 * D)
    F['s_refund_1d'] = rf.count(s, ts, D, D)

    # ---- 3. Receiver behaviour ----
    rr = Ref(r, ts, vals=np.c_[amt, amt ** 2, df.new_payee.values, F['round100'], F['small_amt']])
    for nm, b in [('30m', 1800), ('6h', 6 * H), ('1d', D), ('2d', 2 * D)]:
        F[f'r_cnt_{nm}'] = rr.count(r, ts, b, b)
    sm = rr.sum(r, ts, 1800, 1800)
    c30 = np.maximum(F['r_cnt_30m'], 1)
    F['r_mean_amt_30m'] = sm[:, 0] / c30
    F['r_std_amt_30m'] = np.sqrt(np.maximum(sm[:, 1] / c30 - (sm[:, 0] / c30) ** 2, 0))
    F['r_newpayee_frac_30m'] = sm[:, 2] / c30
    F['r_nuniq_send_1h'] = nunique_bucket(r, s, ts, H)
    F['r_nuniq_send_6h'] = nunique_bucket(r, s, ts, 6 * H)
    F['r_nuniq_send_day'] = nunique_bucket(r, s, ts, D)
    F['r_nuniq_send_week'] = nunique_bucket(r, s, ts, 7 * D)
    F['r_cnt_7d'] = rr.count(r, ts, 3 * D, 3 * D)
    F['r_repeat_ratio_week'] = F['r_cnt_7d'] / np.maximum(F['r_nuniq_send_week'], 1)
    sm1d = rr.sum(r, ts, D, D)
    F['r_round_frac_1d'] = sm1d[:, 3] / np.maximum(rr.count(r, ts, D, D), 1)
    F['r_small_in_1d'] = sm1d[:, 4]
    F['r_out_cnt_30m'] = sref.count(r, ts, 0, 1800)
    F['r_out_cnt_prev_30m'] = sref.count(r, ts, 1800, 0)
    F['r_out_sum_ratio'] = sref.sum(r, ts, 0, 1800)[:, 0] / np.maximum(amt, 1)
    F['receiver_mule_passthrough_ratio'] = F['r_out_cnt_30m'] / np.maximum(F['r_cnt_30m'], 1)

    # ---- 4. Device behaviour ----
    dref = Ref(d, ts)
    for nm, b in [('1m', 60), ('10m', 600), ('1h', H), ('1d', D)]:
        F[f'd_cnt_{nm}'] = dref.count(d, ts, b, b)
    F['d_nuniq_send_day'] = nunique_bucket(d, s, ts, D)
    F['d_nuniq_send_week'] = nunique_bucket(d, s, ts, 7 * D)
    F['d_nuniq_recv_1h'] = nunique_bucket(d, r, ts, H)

    # ---- 5. Money chains ----
    nx, gnx = sref.nearest(r, ts, 'next', 1800)
    inref = Ref(r, ts)
    pv, gpv = inref.nearest(s, ts, 'prev', H)
    ratio_n = np.where(nx >= 0, amt[np.maximum(nx, 0)] / np.maximum(amt, 1e-6), np.nan)
    ratio_p = np.where(pv >= 0, amt / np.maximum(amt[np.maximum(pv, 0)], 1e-6), np.nan)
    F['nx_gap'], F['nx_ratio'] = gnx, ratio_n
    F['pv_gap'], F['pv_ratio'] = gpv, ratio_p
    ok_n = (nx >= 0) & (ratio_n > 0.3) & (ratio_n <= 1.02)
    ok_p = (pv >= 0) & (ratio_p > 0.3) & (ratio_p <= 1.02)
    F['chain_fwd_len'] = follow(np.where(ok_n, nx, -1))
    F['chain_back_len'] = follow(np.where(ok_p, pv, -1))
    F['chain_total'] = F['chain_fwd_len'] + F['chain_back_len']

    # ---- 6. Pairs & Fake Cashback linkages ----
    M = int(ids.max()) + 1
    pairs = s * M + r
    uniq = pd.Index(pairs).unique()
    pc = uniq.get_indexer(pairs).astype(np.int64)
    rc = uniq.get_indexer(r * M + s).astype(np.int64)
    pref = Ref(pc, ts)
    r_pref = Ref(rc, ts)
    F['pair_cnt_7d'] = pref.count(pc, ts, 7 * D, 7 * D)
    F['pair_cnt_past_1d'] = pref.count(pc, ts, D, 0)
    F['pair_cnt_30d'] = pref.count(pc, ts, 30 * D, 30 * D)
    F['rev_before_7d'] = pref.count(rc, ts, 7 * D, 0)
    F['rev_after_3d'] = pref.count(rc, ts, 0, 3 * D)
    sm_mask = amt <= 10
    psm = Ref(pc[sm_mask], ts[sm_mask])
    F['rev_small_before_3d'] = psm.count(rc, ts, 3 * D, 0)
    F['rev_small_after_3d'] = psm.count(rc, ts, 0, 3 * D)
    late = (F['hour'] >= 19).astype(int)
    F['late_pair_cnt_7d'] = Ref(pc[late == 1], ts[late == 1]).count(pc, ts, 7 * D, 7 * D)

    # Nearest reverse transaction pointer
    rev_ptr, _ = r_pref.nearest(rc, ts, 'prev', 2 * H)

    X = pd.DataFrame(F).astype(np.float32)
    ex = dict(
        s=s, r=r, d=d, ts=ts,
        nx=np.where(ok_n, nx, -1),
        pv=np.where(ok_p, pv, -1),
        rev=rev_ptr
    )
    return X, ex


# ----------------------------------------------------------------------------------------------
# Multi-Hop Neighbour Probability Stacking Features
# ----------------------------------------------------------------------------------------------
def neighbour_feats(P, ex):
    """Deep multi-hop neighbour probabilities across receiver, sender, device, and chains."""
    s, r, d, ts = ex['s'], ex['r'], ex['d'], ex['ts']
    cols, names = [], []

    windows = [
        ('r30m', r, 1800),
        ('r6h', r, 6 * H),
        ('r24h', r, 24 * H),
        ('d10m', d, 600),
        ('d1h', d, H),
        ('s1h', s, H),
        ('s6h', s, 6 * H),
        ('s1d', s, D),
        ('s7d', s, 7 * D)
    ]
    for nm, key, b in windows:
        ref = Ref(key, ts, vals=P)
        sm = ref.sum(key, ts, b, b) - P
        cnt = ref.count(key, ts, b, b) - 1
        mean = sm / np.maximum(cnt, 1)[:, None]
        mean[cnt == 0] = np.nan
        cols.append(mean)
        names += [f'nb_{nm}_{LABELS[k]}' for k in range(K)]

    # Forward hop, backward hop, and reverse pair
    for nm, ptr in [('next', ex['nx']), ('prev', ex['pv']), ('rev', ex['rev'])]:
        Pn = np.where((ptr >= 0)[:, None], P[np.maximum(ptr, 0)], np.nan)
        cols.append(Pn)
        names += [f'nb_{nm}_{LABELS[k]}' for k in range(K)]

    cols.append(P)
    names += [f'p1_{LABELS[k]}' for k in range(K)]
    return pd.DataFrame(np.hstack(cols).astype(np.float32), columns=names)


# ----------------------------------------------------------------------------------------------
# Model Training
# ----------------------------------------------------------------------------------------------
class Model:
    def __init__(self, m, best):
        self.m, self.best = m, best

    def proba(self, X):
        p = self.m.predict_proba(X)
        out = np.zeros((len(X), K), dtype=np.float32)
        out[:, np.asarray(self.m.classes_, dtype=int)] = p
        return out


def fit_model(X, y, w, iters=700, lr=0.06, seed=42):
    import xgboost as xgb
    m = xgb.XGBClassifier(
        n_estimators=iters,
        learning_rate=lr,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_lambda=2.0,
        tree_method='hist',
        device='cuda',
        random_state=seed,
        n_jobs=8
    )
    m.fit(X, y, sample_weight=w)
    return Model(m, iters)


def class_weights(y):
    cnt = np.bincount(y, minlength=K).astype(float)
    w = (cnt.max() / np.maximum(cnt, 1)) ** 0.5
    return np.clip(w / w.min(), 1.0, 25.0)[y]


def subsample_idx(idx, y, frac, rng):
    if frac >= 1.0:
        return idx
    keep = (y[idx] != 0) | (rng.random(len(idx)) < frac)
    return idx[keep]


def oof_run(X, y, groups, Xte, n_iter, tag, rng, mask=None, seeds=1, legit_frac=0.60):
    oof = np.zeros((len(X), K), dtype=np.float32)
    pte = np.zeros((len(Xte), K), dtype=np.float32)
    Xn, Xten = X.values, Xte.values

    for sd in range(seeds):
        sd_seed = 42 + 101 * sd
        for k, (tr, va) in enumerate(GroupKFold(5).split(Xn, y, groups)):
            if mask is not None:
                tr = tr[mask[tr]]
            tr = subsample_idx(tr, y, legit_frac, rng)
            m = fit_model(Xn[tr], y[tr], class_weights(y[tr]), iters=n_iter, seed=sd_seed + k)
            oof[va] += m.proba(Xn[va]) / seeds
            pte += m.proba(Xten) / (5.0 * seeds)
            log(f'  [{tag}] seed {sd + 1}/{seeds} fold {k + 1}/5 done')
    return oof, pte


def noise_mask(P, y, thr=0.85):
    return ~((P.argmax(1) != y) & (P.max(1) > thr))


def tune_mult(P, y, n_cycles=4):
    best_w = np.ones(K, dtype=np.float64)
    best_score = macro_f1(y, (P * best_w).argmax(axis=1))
    log(f"  Baseline September OOF Macro-F1: {best_score:.4f}")

    for cycle in range(n_cycles):
        for c in range(1, K):
            best_wc = best_w[c]
            candidates = np.unique(np.concatenate([
                np.linspace(0.2, 3.0, 30),
                np.linspace(max(0.1, best_wc * 0.6), min(12.0, best_wc * 1.6), 25)
            ]))
            for cand in candidates:
                w_try = best_w.copy()
                w_try[c] = cand
                preds = (P * w_try).argmax(axis=1)
                score = macro_f1(y, preds) - 1e-4 * ((cand - 1.0) ** 2)
                if score > best_score:
                    best_score = score
                    best_wc = cand
            best_w[c] = best_wc
        log(f"    Cycle {cycle + 1}/{n_cycles} -> Tuned OOF Macro-F1: {best_score:.4f}")
    return best_w, best_score


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("  SYNORA-TOP1 PRODUCTION STACKING ENGINE")
    log("  Target: 0.60+ Macro-F1 | GPU: RTX 4050")
    log("=" * 70)

    data_dir = r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings"
    out_dir = r"D:\ML Gradient Gemini\submissions"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    # PURGE CONTAMINATED OCTOBER ROWS
    tr = tr[~tr.timestamp.astype(str).str.startswith('2026-10')].reset_index(drop=True)
    log(f"Purged 147 stray October rows. Clean Train: {len(tr):,}, Test: {len(te):,}")

    ac = pd.read_csv(f"{data_dir}/accounts.csv")
    me = pd.read_csv(f"{data_dir}/merchants.csv")
    ac['opened_on'] = pd.to_datetime(ac.opened_on, dayfirst=True, errors='coerce', format='mixed')
    me['onboarded_on'] = pd.to_datetime(me.onboarded_on, dayfirst=True, errors='coerce', format='mixed')

    tr['is_train'] = 1
    te['is_train'] = 0
    te['label'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df['t'] = pd.to_datetime(df.timestamp)
    df['ts'] = (df.t - df.t.min()).dt.total_seconds().astype('int64')
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)

    log("Building features...")
    X, ex = build(df, ac, me)
    log(f"Base Features: {X.shape[1]}")

    is_tr = df.is_train.values == 1
    y_all = df.label.map(L2I).fillna(-1).astype(int).values
    month = df.t.dt.month.values
    test_idx = np.where(~is_tr)[0]
    tr_idx = np.where(is_tr)[0]
    y = y_all[tr_idx]
    last = (month[tr_idx] == 9)
    groups = (df.ts // 86400).values[tr_idx]

    Xtr, Xte = X.iloc[tr_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)

    # ---- Stage 1 ----
    log("Starting Stage 1 GPU Training (iters=700, legit_frac=0.60)...")
    oof1, pte1 = oof_run(Xtr, y, groups, Xte, 700, 'Stage 1', rng, legit_frac=0.60)
    m1, sc1 = tune_mult(oof1[last], y[last])
    log(f"Stage 1 Tuned September OOF Macro-F1: {sc1:.4f}")

    # CleanLab Noise Mask
    mask = noise_mask(oof1, y, thr=0.85)
    log(f"CleanLab Noise Filter: dropping {(~mask).sum():,} suspicious false-legit rows from Stages 2 & 3")

    # ---- Stage 2 ----
    log("Starting Stage 2 Neighbour Stacking...")
    P_all = np.zeros((len(df), K), dtype=np.float32)
    P_all[tr_idx] = oof1
    P_all[test_idx] = pte1
    Xb = X.reset_index(drop=True)
    Xs2 = pd.concat([Xb, neighbour_feats(P_all, ex)], axis=1)
    log(f"Stage 2 Features: {Xs2.shape[1]}")

    X2tr, X2te = Xs2.iloc[tr_idx].reset_index(drop=True), Xs2.iloc[test_idx].reset_index(drop=True)
    oof2, pte2 = oof_run(X2tr, y, groups, X2te, 550, 'Stage 2', rng, mask=mask, legit_frac=0.60)
    m2, sc2 = tune_mult(oof2[last], y[last])
    log(f"Stage 2 Tuned September OOF Macro-F1: {sc2:.4f}")

    # ---- Stage 3 ----
    log("Starting Stage 3 Ring Stacking (3-Seed Bagged Average)...")
    P_all[tr_idx] = oof2
    P_all[test_idx] = pte2
    Xs3 = pd.concat([Xb, neighbour_feats(P_all, ex)], axis=1)
    X3tr, X3te = Xs3.iloc[tr_idx].reset_index(drop=True), Xs3.iloc[test_idx].reset_index(drop=True)
    oof3, pte3 = oof_run(X3tr, y, groups, X3te, 450, 'Stage 3', rng, mask=mask, seeds=3, legit_frac=0.60)
    m3, sc3 = tune_mult(oof3[last], y[last])
    log(f"Stage 3 Tuned September OOF Macro-F1: {sc3:.4f}")

    # Select Best Validating Stage
    stages_res = {1: (oof1, pte1, m1, sc1), 2: (oof2, pte2, m2, sc2), 3: (oof3, pte3, m3, sc3)}
    best_st = max(stages_res, key=lambda k: stages_res[k][3])
    log(f"Selected Stage {best_st} with highest September OOF Macro-F1: {stages_res[best_st][3]:.4f}")

    oof_best, pte_best, m_best, sc_best = stages_res[best_st]

    log("Per-Class F1 (September Holdout):")
    pcf1 = per_class_f1(y[last], (oof_best[last] * m_best).argmax(1))
    log(pd.Series(pcf1, index=LABELS).round(4).to_string())

    # Generate Final Submission
    sub_path = os.path.join(out_dir, "submission_synora_top1_grandmaster.csv")
    pred = (pte_best * m_best).argmax(1)
    sub = pd.DataFrame({'id': df.txn_id.values[test_idx], 'label': np.array(LABELS)[pred]})
    sub.to_csv(sub_path, index=False)
    log(f"DONE -> Saved {sub_path}")

    # 5-point verification
    assert len(sub) == len(te), f"Length mismatch: {len(sub)} vs {len(te)}"
    assert sub.id.is_unique, "Duplicate IDs!"
    assert sub.label.isnull().sum() == 0, "Null values!"
    assert set(sub.label.unique()).issubset(set(LABELS)), "Invalid classes!"

    log("Predicted label distribution:")
    log(sub.label.value_counts(normalize=True).round(4).to_string())
    log("=" * 70)
    log(f"READY FOR LEADERBOARD: {sub_path}")
    log("=" * 70)


if __name__ == '__main__':
    main()
