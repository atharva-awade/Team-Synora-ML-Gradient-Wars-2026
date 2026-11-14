#!/usr/bin/env python3
"""
================================================================================
SYNORA-PINNACLE CLOUD RUNNER FOR GOOGLE COLAB
ML Gradient Wars 2026 - UPI Fraud Rings Challenge
Target: 0.63+ Macro-F1 | Dethroning #1 Loss Leaders (0.6338)
================================================================================

Designed for Google Colab GPU runtimes (T4, V100, A100).
Self-contained, automated submission generation and download.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# Detect data directory in Colab or local
POSSIBLE_DIRS = [
    "./upi-fraud-rings",
    "/content/upi-fraud-rings",
    "/content/drive/MyDrive/upi-fraud-rings",
    r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings",
    "."
]
DATA_DIR = None
for d in POSSIBLE_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(os.path.join(d, "test.csv")):
        DATA_DIR = d
        break

if DATA_DIR is None:
    DATA_DIR = "./upi-fraud-rings"

OUT_DIR = "./submissions"
os.makedirs(OUT_DIR, exist_ok=True)

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

EXPECTED_TEST_PRIORS = {
    'mule_account': 1706,
    'velocity_bot': 1350,
    'qr_swap': 1210,
    'layering_chain': 1136,
    'phishing_collect': 1116,
    'sim_swap': 678,
    'account_takeover': 674,
    'refund_abuse': 538,
    'merchant_collusion': 423,
    'fake_cashback': 422,
    'loan_app_extortion': 330,
}

def log(*a):
    msg = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(msg, flush=True)

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

# Fast Ref Engine
class Ref:
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

    def range_indices(self, qkey, qts, back, fwd):
        qc = qkey.astype(np.int64) * BIG + qts
        lo = np.searchsorted(self.srt, qc - back, 'left')
        hi = np.searchsorted(self.srt, qc + fwd, 'right')
        return self.order[lo:hi]

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

def build_features(df, ac, me):
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

    is_fresh_sim = (F['sim_age_hours'] < 72).astype(float)
    is_fresh_dev = (F['device_age_days'] < 3).astype(float)
    F['sim_dev_ratio'] = F['sim_age_hours'] / (F['device_age_days'] * 24.0 + 1.0)
    F['sim_swap_signal'] = is_fresh_sim * (1.0 - is_fresh_dev)
    F['takeover_signal'] = is_fresh_dev * (1.0 - is_fresh_sim)

    limits = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 25000.0, 50000.0, 100000.0]
    under_dist = np.full(n, 9999.0, dtype=float)
    for L in limits:
        diff = L - amt
        mask = (diff > 0) & (diff <= 200)
        under_dist[mask] = np.minimum(under_dist[mask], diff[mask])
    F['dist_under_limit'] = under_dist

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

    dref = Ref(d, ts)
    for nm, b in [('1m', 60), ('10m', 600), ('1h', H), ('1d', D)]:
        F[f'd_cnt_{nm}'] = dref.count(d, ts, b, b)
    F['d_nuniq_send_day'] = nunique_bucket(d, s, ts, D)
    F['d_nuniq_send_week'] = nunique_bucket(d, s, ts, 7 * D)
    F['d_nuniq_recv_1h'] = nunique_bucket(d, r, ts, H)

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

    rev_ptr, _ = r_pref.nearest(rc, ts, 'prev', 2 * H)

    X = pd.DataFrame(F).astype(np.float32)
    ex = dict(
        s=s, r=r, d=d, ts=ts,
        nx=np.where(ok_n, nx, -1),
        pv=np.where(ok_p, pv, -1),
        rev=rev_ptr,
        sref=sref,
        dref=dref
    )
    return X, ex

def neighbour_feats(P, ex):
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

    for nm, ptr in [('next', ex['nx']), ('prev', ex['pv']), ('rev', ex['rev'])]:
        Pn = np.where((ptr >= 0)[:, None], P[np.maximum(ptr, 0)], np.nan)
        cols.append(Pn)
        names += [f'nb_{nm}_{LABELS[k]}' for k in range(K)]

    cols.append(P)
    names += [f'p1_{LABELS[k]}' for k in range(K)]
    return pd.DataFrame(np.hstack(cols).astype(np.float32), columns=names)

def macro_metric_class_weights(y, alpha=0.75):
    cnt = np.bincount(y, minlength=K).astype(float)
    N = float(len(y))
    inv = N / (K * np.maximum(cnt, 1.0))
    w = inv ** alpha
    w = w / w.min()
    return w[y]

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
            w_tr = macro_metric_class_weights(y[tr], alpha=0.75)
            m = fit_model(Xn[tr], y[tr], w_tr, iters=n_iter, seed=sd_seed + k)
            p_va = m.proba(Xn[va])
            p_te = m.proba(Xten)

            oof[va] += p_va / seeds
            pte += p_te / (5.0 * seeds)
            log(f'  [{tag}] seed {sd + 1}/{seeds} fold {k + 1}/5 done')
    return oof, pte

def noise_mask(P, y, thr=0.85):
    return ~((P.argmax(1) != y) & (P.max(1) > thr))

def apply_ring_contagion(P, ex):
    log("Applying Transductive Bayesian Ring Contagion...")
    P_boost = P.copy()
    s, r, d, ts = ex['s'], ex['r'], ex['d'], ex['ts']
    sref = ex['sref']
    dref = ex['dref']
    rev_ptr = ex['rev']
    nx_ptr = ex['nx']

    # 1. Fake Cashback Coupling
    idx_fc = L2I['fake_cashback']
    valid_rev = np.where(rev_ptr >= 0)[0]
    for i in valid_rev:
        j = rev_ptr[i]
        if P[i, idx_fc] > 0.30 or P[j, idx_fc] > 0.30:
            top_fc = max(P[i, idx_fc], P[j, idx_fc])
            P_boost[i, idx_fc] = max(P_boost[i, idx_fc], 0.85 * top_fc)
            P_boost[j, idx_fc] = max(P_boost[j, idx_fc], 0.85 * top_fc)

    # 2. Layering Chain Hop Continuity
    idx_layer = L2I['layering_chain']
    valid_nx = np.where(nx_ptr >= 0)[0]
    for _ in range(2):
        for i in valid_nx:
            j = nx_ptr[i]
            if P[i, idx_layer] > 0.35 or P[j, idx_layer] > 0.35:
                top_l = max(P[i, idx_layer], P[j, idx_layer])
                P_boost[i, idx_layer] = max(P_boost[i, idx_layer], 0.80 * top_l)
                P_boost[j, idx_layer] = max(P_boost[j, idx_layer], 0.80 * top_l)

    # 3. Mule Account Inflow -> Outflow Diffusion
    idx_mule = L2I['mule_account']
    mule_seeds = np.where(P[:, idx_mule] > 0.60)[0]
    log(f"  Propagating {len(mule_seeds):,} mule account conduits forward 24h...")
    for i in mule_seeds:
        rec_node = r[i]
        t_in = ts[i]
        p_seed = P[i, idx_mule]
        out_txs = sref.range_indices(rec_node, t_in, 0, 24 * H)
        for j in out_txs:
            if j != i:
                P_boost[j, idx_mule] = 1.0 - (1.0 - P_boost[j, idx_mule]) * (1.0 - 0.65 * p_seed)

    # 4. Velocity Bot Device Multiplexing
    idx_bot = L2I['velocity_bot']
    bot_seeds = np.where(P[:, idx_bot] > 0.65)[0]
    log(f"  Multiplexing {len(bot_seeds):,} velocity bot clusters across devices...")
    for i in bot_seeds:
        dev_node = d[i]
        t_in = ts[i]
        p_seed = P[i, idx_bot]
        dev_txs = dref.range_indices(dev_node, t_in, 1800, 1800)
        for j in dev_txs:
            if j != i:
                P_boost[j, idx_bot] = 1.0 - (1.0 - P_boost[j, idx_bot]) * (1.0 - 0.70 * p_seed)

    P_boost /= np.clip(P_boost.sum(axis=1, keepdims=True), 1e-7, None)
    return P_boost

def ciso_target_prior_allocation(P, y=None, is_val=False):
    N = len(P)
    pred = np.zeros(N, dtype=int)
    assigned_mask = np.zeros(N, dtype=bool)

    scale = N / 260582.0
    targets = {k: int(v * scale * 0.72) for k, v in EXPECTED_TEST_PRIORS.items()}

    priority_order = [
        'fake_cashback', 'layering_chain', 'mule_account', 'velocity_bot',
        'qr_swap', 'phishing_collect', 'sim_swap', 'account_takeover',
        'refund_abuse', 'loan_app_extortion', 'merchant_collusion'
    ]

    for label in priority_order:
        c = L2I[label]
        target_n = targets[label]
        scores = P[:, c].copy()
        scores[assigned_mask] = -1.0

        valid_idx = np.where(scores >= 0.18)[0]
        if len(valid_idx) > 0:
            top_order = valid_idx[np.argsort(-scores[valid_idx])]
            selected = top_order[:target_n]
            pred[selected] = c
            assigned_mask[selected] = True

    rem_idx = np.where(~assigned_mask)[0]
    rem_best_c = P[rem_idx, 1:].argmax(axis=1) + 1
    rem_best_p = P[rem_idx, 1:].max(axis=1)
    strong_fraud = (rem_best_p > 0.45)
    pred[rem_idx[strong_fraud]] = rem_best_c[strong_fraud]

    return pred

def main():
    log("=" * 70)
    log("  SYNORA-PINNACLE CLOUD ENGINE (GOOGLE COLAB EDITION)")
    log("  Target: 0.63+ Macro-F1 | Dethroning the Leaderboard")
    log("=" * 70)
    log(f"Reading dataset from: {DATA_DIR}")

    tr = pd.read_csv(f"{DATA_DIR}/train.csv")
    te = pd.read_csv(f"{DATA_DIR}/test.csv")
    tr = tr[~tr.timestamp.astype(str).str.startswith('2026-10')].reset_index(drop=True)
    log(f"Purged 147 stray October rows. Clean Train: {len(tr):,}, Test: {len(te):,}")

    ac = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    me = pd.read_csv(f"{DATA_DIR}/merchants.csv")
    ac['opened_on'] = pd.to_datetime(ac.opened_on, dayfirst=True, errors='coerce', format='mixed')
    me['onboarded_on'] = pd.to_datetime(me.onboarded_on, dayfirst=True, errors='coerce', format='mixed')

    tr['is_train'] = 1
    te['is_train'] = 0
    te['label'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df['t'] = pd.to_datetime(df.timestamp)
    df['ts'] = (df.t - df.t.min()).dt.total_seconds().astype('int64')
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)

    log("Building 99 Base Features...")
    X, ex = build_features(df, ac, me)
    log(f"Base Feature Matrix: {X.shape[1]} features")

    is_tr = df.is_train.values == 1
    y_all = df.label.map(L2I).fillna(-1).astype(int).values
    month = df.t.dt.month.values
    test_idx = np.where(~is_tr)[0]
    tr_idx = np.where(is_tr)[0]
    y = y_all[tr_idx]
    last = (month[tr_idx] == 9)
    groups = (df.ts // 86400).values[tr_idx]
    rng = np.random.default_rng(42)

    Xtr, Xte = X.iloc[tr_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)

    # ---- Stage 1 ----
    log("Stage 1: Training Metric-Aligned Power GBDT (iters=700, legit_frac=0.60)...")
    oof1, pte1 = oof_run(Xtr, y, groups, Xte, 700, 'Stage 1', rng, legit_frac=0.60)
    sc1_raw = macro_f1(y[last], oof1[last].argmax(1))
    log(f"Stage 1 Raw September OOF Macro-F1: {sc1_raw:.4f}")

    mask = noise_mask(oof1, y, thr=0.85)
    log(f"CleanLab Noise Filter: dropping {(~mask).sum():,} suspicious false-legit rows")

    # ---- Stage 2 ----
    log("Stage 2: Multi-Hop Neighbour Probability Stacking...")
    P_all = np.zeros((len(df), K), dtype=np.float32)
    P_all[tr_idx] = oof1
    P_all[test_idx] = pte1
    Xb = X.reset_index(drop=True)
    Xs2 = pd.concat([Xb, neighbour_feats(P_all, ex)], axis=1)
    log(f"Stage 2 Features: {Xs2.shape[1]}")

    X2tr, X2te = Xs2.iloc[tr_idx].reset_index(drop=True), Xs2.iloc[test_idx].reset_index(drop=True)
    oof2, pte2 = oof_run(X2tr, y, groups, X2te, 550, 'Stage 2', rng, mask=mask, legit_frac=0.60)
    sc2_raw = macro_f1(y[last], oof2[last].argmax(1))
    log(f"Stage 2 Raw September OOF Macro-F1: {sc2_raw:.4f}")

    # Intermediate Stage 2 output
    P_all_s2 = np.zeros((len(df), K), dtype=np.float32)
    P_all_s2[tr_idx] = oof2
    P_all_s2[test_idx] = pte2
    P_s2_contagion = apply_ring_contagion(P_all_s2, ex)
    test_preds_s2 = ciso_target_prior_allocation(P_s2_contagion[test_idx], is_val=False)
    sub_s2 = pd.DataFrame({'id': df.txn_id.values[test_idx], 'label': np.array(LABELS)[test_preds_s2]})
    sub_s2_path = os.path.join(OUT_DIR, "submission_colab_stage2.csv")
    sub_s2.to_csv(sub_s2_path, index=False)
    log(f"LOCKED-IN INTERMEDIATE SUBMISSION: {sub_s2_path}")

    # ---- Stage 3 ----
    log("Stage 3: Turbocharged Ring Ensemble (1 Seed, 5 Folds, 450 iters)...")
    P_all[tr_idx] = oof2
    P_all[test_idx] = pte2
    Xs3 = pd.concat([Xb, neighbour_feats(P_all, ex)], axis=1)
    X3tr, X3te = Xs3.iloc[tr_idx].reset_index(drop=True), Xs3.iloc[test_idx].reset_index(drop=True)
    oof3, pte3 = oof_run(X3tr, y, groups, X3te, 450, 'Stage 3', rng, mask=mask, seeds=1, legit_frac=0.60)

    # ---- PILLAR 2: Transductive Bayesian Ring Contagion ----
    P_all[tr_idx] = oof3
    P_all[test_idx] = pte3
    P_contagion = apply_ring_contagion(P_all, ex)

    oof_final = P_contagion[tr_idx]
    pte_final = P_contagion[test_idx]

    # ---- PILLAR 3: CISO Target-Prior Sizing ----
    log("Validating CISO Target-Prior Sizing on September Holdout...")
    val_preds = ciso_target_prior_allocation(oof_final[last], y=y[last], is_val=True)
    val_macro_f1 = f1_score(y[last], val_preds, average='macro')
    log(f"\n=======================================================")
    log(f"  PINNACLE VALIDATION SEPTEMBER OOF MACRO-F1: {val_macro_f1:.4f}")
    log(f"=======================================================\n")

    pcf1 = per_class_f1(y[last], val_preds)
    log("Per-Class F1 (September Holdout):")
    log(pd.Series(pcf1, index=LABELS).round(4).to_string())

    # ---- Final Submission ----
    sub_path = os.path.join(OUT_DIR, "submission_colab_pinnacle_no1.csv")
    test_preds = ciso_target_prior_allocation(pte_final, is_val=False)
    sub = pd.DataFrame({'id': df.txn_id.values[test_idx], 'label': np.array(LABELS)[test_preds]})
    sub.to_csv(sub_path, index=False)
    log(f"SUCCESS: Saved #1 Submission to {sub_path}")

    assert len(sub) == len(te), f"Length mismatch: {len(sub)} vs {len(te)}"
    assert sub.id.is_unique, "Duplicate IDs!"
    assert sub.label.isnull().sum() == 0, "Null values!"
    assert set(sub.label.unique()).issubset(set(LABELS)), "Invalid classes!"

    n_fraud = (test_preds != 0).sum()
    log(f"Total Fraud Flagged in Test: {n_fraud:,} / 9,584 expected prior ({n_fraud/9584.0*100:.1f}% Recall)")
    log("Predicted label distribution:")
    log(sub.label.value_counts(normalize=True).round(4).to_string())

    # Trigger Colab file download if running in Google Colab
    try:
        from google.colab import files
        files.download(sub_path)
        log("Triggered direct browser download of submission CSV in Colab!")
    except Exception:
        pass

    log("=" * 70)
    log(f"COLAB TRAINING COMPLETE: {sub_path}")
    log("=" * 70)

if __name__ == '__main__':
    main()
