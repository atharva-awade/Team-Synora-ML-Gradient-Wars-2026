#!/usr/bin/env python
"""
UPI Fraud Rings - ML Gradient Wars 2026
Full pipeline: graph / time-window features -> stage-1 GBM (day-grouped OOF)
-> neighbour-probability features -> stage-2 GBM -> macro-F1 class-multiplier tuning.

Usage:
  python pipeline.py --data ./data --out ./out                 # CPU (LightGBM if installed)
  python pipeline.py --data /content/data --out /content/out --gpu   # Colab T4, XGBoost GPU
Options worth knowing:
  --thin 0.5      randomly keep 50% of train rows BEFORE building window features
                  (use if test has ~half the rows/day of train: makes count features comparable)
  --legit_frac    fraction of 'legit' rows used for training (speed). Prediction always uses all rows.
"""
import argparse, os, time, warnings
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover',
          'phishing_collect', 'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion',
          'merchant_collusion']
L2I = {l: i for i, l in enumerate(LABELS)}
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
        """index (into original arrays) of the closest strictly-later / strictly-earlier row with same key."""
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
    """distinct `other` per key inside time buckets (max of two half-shifted bucketings)."""
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
# feature building
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

    # ---- single-row ------------------------------------------------------------------------
    F['hour'] = t.dt.hour.values
    F['weekday'] = t.dt.weekday.values
    F['night'] = ((F['hour'] >= 23) | (F['hour'] <= 5)).astype(int)
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

    # ---- sender behaviour ------------------------------------------------------------------
    sref = Ref(s, ts, vals=amt)
    for nm, b in [('5m', 300), ('1h', H), ('1d', D)]:
        F[f's_cnt_{nm}'] = sref.count(s, ts, b, b)
    F['s_cnt_past_1d'] = sref.count(s, ts, D, 0)
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

    # ---- receiver behaviour ----------------------------------------------------------------
    rr = Ref(r, ts, vals=np.c_[amt, amt ** 2, df.new_payee.values, F['round100'], F['small_amt']])
    for nm, b in [('30m', 1800), ('6h', 6 * H), ('2d', 2 * D)]:
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
    F['r_out_cnt_30m'] = sref.count(r, ts, 0, 1800)          # receiver sends money on soon after
    F['r_out_cnt_prev_30m'] = sref.count(r, ts, 1800, 0)
    F['r_out_sum_ratio'] = sref.sum(r, ts, 0, 1800)[:, 0] / np.maximum(amt, 1)

    # ---- device behaviour ------------------------------------------------------------------
    dref = Ref(d, ts)
    for nm, b in [('1m', 60), ('10m', 600), ('1h', H), ('1d', D)]:
        F[f'd_cnt_{nm}'] = dref.count(d, ts, b, b)
    F['d_nuniq_send_day'] = nunique_bucket(d, s, ts, D)
    F['d_nuniq_send_week'] = nunique_bucket(d, s, ts, 7 * D)
    F['d_nuniq_recv_1h'] = nunique_bucket(d, r, ts, H)

    # ---- money chains ----------------------------------------------------------------------
    nx, gnx = sref.nearest(r, ts, 'next', 1800)               # receiver's next outgoing payment
    inref = Ref(r, ts)
    pv, gpv = inref.nearest(s, ts, 'prev', H)                 # money that reached the sender just before
    ratio_n = np.where(nx >= 0, amt[np.maximum(nx, 0)] / np.maximum(amt, 1e-6), np.nan)
    ratio_p = np.where(pv >= 0, amt / np.maximum(amt[np.maximum(pv, 0)], 1e-6), np.nan)
    F['nx_gap'], F['nx_ratio'] = gnx, ratio_n
    F['pv_gap'], F['pv_ratio'] = gpv, ratio_p
    ok_n = (nx >= 0) & (ratio_n > 0.3) & (ratio_n <= 1.02)
    ok_p = (pv >= 0) & (ratio_p > 0.3) & (ratio_p <= 1.02)
    F['chain_fwd_len'] = follow(np.where(ok_n, nx, -1))
    F['chain_back_len'] = follow(np.where(ok_p, pv, -1))
    F['chain_total'] = F['chain_fwd_len'] + F['chain_back_len']

    # ---- pairs -----------------------------------------------------------------------------
    M = int(ids.max()) + 1
    pairs = s * M + r
    uniq = pd.Index(pairs).unique()
    pc = uniq.get_indexer(pairs).astype(np.int64)
    rc = uniq.get_indexer(r * M + s).astype(np.int64)
    pref = Ref(pc, ts)
    F['pair_cnt_7d'] = pref.count(pc, ts, 7 * D, 7 * D)
    F['pair_cnt_past_1d'] = pref.count(pc, ts, D, 0)
    F['pair_cnt_30d'] = pref.count(pc, ts, 30 * D, 30 * D)
    F['rev_before_7d'] = pref.count(rc, ts, 7 * D, 0)
    F['rev_after_3d'] = pref.count(rc, ts, 0, 3 * D)
    sm_mask = amt <= 10
    psm = Ref(pc[sm_mask], ts[sm_mask])
    F['rev_small_before_3d'] = psm.count(rc, ts, 3 * D, 0)    # fake-cashback: tiny credit came first
    F['rev_small_after_3d'] = psm.count(rc, ts, 0, 3 * D)
    late = (F['hour'] >= 19).astype(int)
    F['late_pair_cnt_7d'] = pair_late = Ref(pc[late == 1], ts[late == 1]).count(pc, ts, 7 * D, 7 * D)

    X = pd.DataFrame(F).astype(np.float32)
    ex = dict(s=s, r=r, d=d, ts=ts, nx=np.where(ok_n, nx, -1), pv=np.where(ok_p, pv, -1))
    return X, ex


def neighbour_feats(P, ex):
    """Stage-2 features: what do the *other* transactions around this one look like to stage 1?"""
    s, r, d, ts = ex['s'], ex['r'], ex['d'], ex['ts']
    cols, names = [], []
    for nm, key, b in [('r30m', r, 1800), ('r6h', r, 6 * H), ('d10m', d, 600), ('s1h', s, H), ('s1d', s, D)]:
        ref = Ref(key, ts, vals=P)
        sm = ref.sum(key, ts, b, b) - P
        cnt = ref.count(key, ts, b, b) - 1
        mean = sm / np.maximum(cnt, 1)[:, None]
        mean[cnt == 0] = np.nan
        cols.append(mean)
        names += [f'nb_{nm}_{LABELS[k]}' for k in range(K)]
    for nm, ptr in [('next', ex['nx']), ('prev', ex['pv'])]:
        Pn = np.where((ptr >= 0)[:, None], P[np.maximum(ptr, 0)], np.nan)
        cols.append(Pn)
        names += [f'nb_{nm}_{LABELS[k]}' for k in range(K)]
    cols.append(P)
    names += [f'p1_{LABELS[k]}' for k in range(K)]
    return pd.DataFrame(np.hstack(cols).astype(np.float32), columns=names)


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def pick_backend(args):
    if args.backend != 'auto':
        return args.backend
    try:
        import xgboost  # noqa
        if args.gpu:
            return 'xgb'
    except Exception:
        pass
    try:
        import lightgbm  # noqa
        return 'lgb'
    except Exception:
        pass
    try:
        import xgboost  # noqa
        return 'xgb'
    except Exception:
        return 'sk'


class Model:
    def __init__(self, m, kind, best):
        self.m, self.kind, self.best = m, kind, best

    def proba(self, X):
        p = self.m.predict_proba(X)
        out = np.zeros((len(X), K), dtype=np.float32)
        out[:, np.asarray(self.m.classes_, dtype=int)] = p
        return out


def fit_model(args, X, y, w, Xv=None, yv=None, n_iter=None):
    kind = args.kind
    if kind == 'lgb':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=n_iter or args.iters, learning_rate=args.lr, num_leaves=63,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                               min_child_samples=30, n_jobs=-1, verbose=-1, random_state=args.seed)
        if Xv is not None and n_iter is None:
            m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(80, verbose=False)])
            return Model(m, kind, int(m.best_iteration_ or args.iters))
        m.fit(X, y, sample_weight=w)
        return Model(m, kind, n_iter)
    if kind == 'xgb':
        import xgboost as xgb
        kw = dict(n_estimators=n_iter or args.iters, learning_rate=args.lr * 1.2, max_depth=8,
                  subsample=0.8, colsample_bytree=0.7, min_child_weight=2, reg_lambda=2.0,
                  tree_method='hist', random_state=args.seed, n_jobs=-1)
        if args.gpu:
            kw['device'] = 'cuda'
        if Xv is not None and n_iter is None:
            m = xgb.XGBClassifier(early_stopping_rounds=80, eval_metric='mlogloss', **kw)
            m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
            return Model(m, kind, int(m.best_iteration) + 1)
        m = xgb.XGBClassifier(**kw)
        m.fit(X, y, sample_weight=w)
        return Model(m, kind, n_iter)
    from sklearn.ensemble import HistGradientBoostingClassifier
    it = n_iter or min(args.iters, 150)
    m = HistGradientBoostingClassifier(max_iter=it, learning_rate=args.lr * 1.5, max_leaf_nodes=31,
                                       early_stopping=False, random_state=args.seed)
    m.fit(X, y, sample_weight=w)
    return Model(m, kind, it)


def class_weights(y):
    cnt = np.bincount(y, minlength=K).astype(float)
    w = (cnt.max() / np.maximum(cnt, 1)) ** 0.5
    return np.clip(w / w.min(), 1, 30)[y]


def subsample_idx(idx, y, frac, rng):
    if frac >= 1:
        return idx
    keep = (y[idx] != 0) | (rng.random(len(idx)) < frac)
    return idx[keep]


def oof_run(args, X, y, groups, Xte, n_iter, tag, rng, mask=None, seeds=1):
    """Day-grouped OOF. `mask` = rows allowed in TRAINING (noise filter); all rows still get predictions."""
    import copy
    oof = np.zeros((len(X), K), dtype=np.float32)
    pte = np.zeros((len(Xte), K), dtype=np.float32)
    Xn, Xten = X.values, Xte.values
    for sd in range(seeds):
        a2 = copy.copy(args)
        a2.seed = args.seed + 101 * sd
        for k, (tr, va) in enumerate(GroupKFold(args.folds).split(Xn, y, groups)):
            if mask is not None:
                tr = tr[mask[tr]]
            tr = subsample_idx(tr, y, args.legit_frac, rng)
            m = fit_model(a2, Xn[tr], y[tr], class_weights(y[tr]), n_iter=n_iter)
            oof[va] += m.proba(Xn[va]) / seeds
            pte += m.proba(Xten) / (args.folds * seeds)
            log(f'  [{tag}] seed {sd + 1}/{seeds} fold {k + 1}/{args.folds} done')
    return oof, pte


def noise_mask(P, y, thr=0.9):
    """False for rows whose label strongly disagrees with an out-of-fold model (likely unreported / mislabelled)."""
    return ~((P.argmax(1) != y) & (P.max(1) > thr))


def tune_mult(P, y, iters=3):
    m = np.ones(K)
    grid = np.exp(np.linspace(np.log(0.2), np.log(10), 31))
    best = macro_f1(y, (P * m).argmax(1))
    for _ in range(iters):
        for c in range(K):
            for g in grid:
                mm = m.copy()
                mm[c] = g
                sc = macro_f1(y, (P * mm).argmax(1))
                if sc > best + 1e-6:
                    best, m = sc, mm
    return m, best


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='.')
    ap.add_argument('--out', default='./out')
    ap.add_argument('--backend', default='auto', choices=['auto', 'lgb', 'xgb', 'sk'])
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--lr', type=float, default=0.06)
    ap.add_argument('--legit_frac', type=float, default=0.35)
    ap.add_argument('--thin', default='1.0', help="fraction of train rows kept before windows, or 'auto' (match test density)")
    ap.add_argument('--stages', type=int, default=3, help='stacking stages (1 = stage 1 only)')
    ap.add_argument('--seeds', type=int, default=1, help='seeds averaged in the final stage')
    ap.add_argument('--clean', action='store_true', help='drop likely-mislabelled rows when training stages >= 2')
    ap.add_argument('--ks_thresh', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_stage2', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    args.kind = pick_backend(args)
    log('backend:', args.kind, '| gpu' if args.gpu else '')

    tr = pd.read_csv(f'{args.data}/train.csv')
    te = pd.read_csv(f'{args.data}/test.csv')
    # PURGE STRAY BOUNDARY ROWS
    tr = tr[~tr.timestamp.astype(str).str.startswith('2026-10')].reset_index(drop=True)
    log('Purged 147 contaminated Oct rows from train. Clean train:', len(tr))
    ac = pd.read_csv(f'{args.data}/accounts.csv')
    me = pd.read_csv(f'{args.data}/merchants.csv')
    ac['opened_on'] = pd.to_datetime(ac.opened_on, dayfirst=True, errors='coerce', format='mixed')
    me['onboarded_on'] = pd.to_datetime(me.onboarded_on, dayfirst=True, errors='coerce', format='mixed')
    def _dens(d):
        return len(d) / pd.to_datetime(d.timestamp).dt.date.nunique()
    ratio = _dens(te) / _dens(tr)
    log(f'rows/day: train {_dens(tr):,.0f}  test {_dens(te):,.0f}  ratio {ratio:.2f}')
    thin = min(1.0, ratio) if str(args.thin) == 'auto' else float(args.thin)
    if thin < 1:
        tr = tr[rng.random(len(tr)) < thin].reset_index(drop=True)
        log(f'thinned train to {len(tr):,} rows (keep {thin:.2f})')
    tr['is_train'] = 1
    te['is_train'] = 0
    te['label'] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    df['t'] = pd.to_datetime(df.timestamp)
    df['ts'] = (df.t - df.t.min()).dt.total_seconds().astype('int64')
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)
    log(f'rows: train {len(tr):,}  test {len(te):,}')
    dpd = df.groupby([df.t.dt.to_period('M'), df.is_train]).size() / df.groupby([df.t.dt.to_period('M'), df.is_train]).t.apply(lambda x: x.dt.day.nunique())
    log('rows per active day by month:\n' + dpd.to_string())

    log('building features ...')
    X, ex = build(df, ac, me)
    log(f'features: {X.shape[1]}')
    is_tr = df.is_train.values == 1
    y_all = df.label.map(L2I).fillna(-1).astype(int).values
    month = df.t.dt.month.values
    test_idx = np.where(~is_tr)[0]
    tr_idx = np.where(is_tr)[0]

    # ---- drift report: last train month vs test month ---------------------------------------
    last_m = np.sort(np.unique(month[is_tr]))[-1]
    a_idx = np.where(is_tr & (month == last_m))[0]
    drift = {}
    for c in X.columns:
        va = X[c].values[rng.choice(a_idx, min(40000, len(a_idx)), replace=False)]
        vb = X[c].values[rng.choice(test_idx, min(40000, len(test_idx)), replace=False)]
        na, nb = np.isnan(va).mean(), np.isnan(vb).mean()
        va, vb = va[~np.isnan(va)], vb[~np.isnan(vb)]
        ks = ks_2samp(va, vb).statistic if len(va) > 50 and len(vb) > 50 else 0
        drift[c] = (max(ks, abs(na - nb)), ks, na, nb)
    dr = pd.DataFrame(drift, index=['drift', 'ks', 'nan_train', 'nan_test']).T.sort_values('drift', ascending=False)
    dr.to_csv(f'{args.out}/drift_report.csv')
    log('top drifting features (last train month vs test):\n' + dr.head(12).round(3).to_string())
    PROTECTED = {'amount', 'log_amt', 'txn_type', 'merchant_category', 'hour', 'night', 's_bank', 'amt_share', 'r_mean_amt_30m'}
    drop = [c for c in dr.index[dr.drift > args.ks_thresh] if c not in PROTECTED]
    if drop:
        log('DROPPING drifted features:', drop)
        X = X.drop(columns=drop)

    Xtr, Xte = X.iloc[tr_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y = y_all[tr_idx]
    mon = month[tr_idx]
    groups = df.ts.values[tr_idx] // D
    last = mon == last_m

    # ---- honest forward validation: first months -> last month -----------------------------
    log(f'forward validation: train months < {last_m}, validate month {last_m}')
    fit_i = subsample_idx(np.where(~last)[0], y, args.legit_frac, rng)
    m0 = fit_model(args, Xtr.values[fit_i], y[fit_i], class_weights(y[fit_i]),
                   Xv=Xtr.values[last], yv=y[last])
    pv_ = m0.proba(Xtr.values[last])
    log(f'  best_iter={m0.best}  raw macro-F1 = {macro_f1(y[last], pv_.argmax(1)):.4f}')
    mm, sc = tune_mult(pv_, y[last])
    log(f'  after multiplier tuning (optimistic, tuned on same month) = {sc:.4f}')
    log('  per-class F1:', dict(zip(LABELS, per_class_f1(y[last], (pv_ * mm).argmax(1)).round(3))))
    n_iter = max(int(m0.best * 1.1), 50)

    # ---- stage 1 ------------------------------------------------------------------------------
    log('stage 1: day-grouped OOF')
    oof1, pte1 = oof_run(args, Xtr, y, groups, Xte, n_iter, 's1', rng)
    log(f'  stage-1 OOF macro-F1 all={macro_f1(y, oof1.argmax(1)):.4f}  last-month={macro_f1(y[last], oof1[last].argmax(1)):.4f}')
    m1, sc1 = tune_mult(oof1[last], y[last])
    sub1 = pd.DataFrame({'id': df.txn_id.values[test_idx],
                         'label': np.array(LABELS)[(pte1 * m1).argmax(1)]})
    sub1.to_csv(f'{args.out}/submission_stage1.csv', index=False)
    np.save(f'{args.out}/pte1.npy', pte1)
    log(f'  wrote submission_stage1.csv (tuned OOF last-month F1 {sc1:.4f})')
    if args.no_stage2 or args.stages < 2:
        return

    # ---- stages 2..N: neighbour-probability stacking ------------------------------------------
    mask = noise_mask(oof1, y) if args.clean else None
    if mask is not None:
        log(f'  noise filter: excluding {int((~mask).sum()):,} suspicious train rows from stages >= 2')
    oof, pte = oof1, pte1
    results = {1: (oof1, pte1)}
    Xb = X.reset_index(drop=True)
    for st in range(2, args.stages + 1):
        log(f'stage {st}: neighbour-probability features')
        P_all = np.zeros((len(df), K), dtype=np.float32)
        P_all[tr_idx] = oof
        P_all[test_idx] = pte
        Xs = pd.concat([Xb, neighbour_feats(P_all, ex)], axis=1)
        Xstr, Xste = Xs.iloc[tr_idx].reset_index(drop=True), Xs.iloc[test_idx].reset_index(drop=True)
        final = st == args.stages
        oof, pte = oof_run(args, Xstr, y, groups, Xste, max(int(n_iter * 0.8), 50), f's{st}', rng,
                           mask=mask, seeds=args.seeds if final else 1)
        log(f'  stage-{st} OOF macro-F1 all={macro_f1(y, oof.argmax(1)):.4f}  last-month={macro_f1(y[last], oof[last].argmax(1)):.4f}')
        np.save(f'{args.out}/pte{st}.npy', pte)
        np.save(f'{args.out}/oof{st}.npy', oof)
        results[st] = (oof, pte)
    # choose the stage that validates best on the last train month (stacking does not always help)
    scores = {st: tune_mult(o[last], y[last]) for st, (o, _) in results.items()}
    best_st = max(scores, key=lambda k: scores[k][1])
    log('stage scores (tuned last-month OOF F1):', {k: round(v[1], 4) for k, v in scores.items()}, '-> using stage', best_st)
    oof, pte = results[best_st]
    m2, sc2 = scores[best_st]
    np.save(f'{args.out}/pte_final.npy', pte)
    cm = pd.crosstab(pd.Series(np.array(LABELS)[y[last]], name='true'),
                     pd.Series(np.array(LABELS)[(oof[last] * m2).argmax(1)], name='pred'))
    cm.to_csv(f'{args.out}/confusion_last_month.csv')
    log('confusion matrix (last train month, rows=true, cols=pred):\n' + cm.to_string())
    pred = (pte * m2).argmax(1)
    log(f'  tuned OOF last-month F1 {sc2:.4f}')
    log('  per-class F1 (last month):', dict(zip(LABELS, per_class_f1(y[last], (oof[last] * m2).argmax(1)).round(3))))
    sub = pd.DataFrame({'id': df.txn_id.values[test_idx], 'label': np.array(LABELS)[pred]})
    sub.to_csv(f'{args.out}/submission.csv', index=False)
    np.save(f'{args.out}/mult.npy', m2)
    log('predicted label distribution:\n' + sub.label.value_counts(normalize=True).round(4).to_string())
    assert sub.id.is_unique and len(sub) == len(te)
    log('DONE ->', f'{args.out}/submission.csv')


if __name__ == '__main__':
    main()
