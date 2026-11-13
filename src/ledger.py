import os
import gc
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from src.config import TRAIN_PARQUET, TEST_PARQUET, CACHE_DIR

LEDGER_FEATS_PARQUET = os.path.join(CACHE_DIR, "ledger_features.parquet")

def compute_bidirectional_ledger(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes exact chronological pass-through latency, balance retention ratio,
    running balance, and conduit index across both sender and receiver roles.
    """
    print("[LEDGER] Initializing Bidirectional Account Ledger Engine...")
    t0 = time.time()
    n = len(full_df)

    senders = full_df['sender_id'].astype(str).values
    receivers = full_df['receiver_id'].astype(str).values
    amounts = full_df['amount'].fillna(0.0).values.astype(np.float64)

    # Convert timestamps to numeric seconds
    ts = pd.to_datetime(full_df['timestamp'])
    time_sec = (ts.astype("int64") // 10**9).values

    order = np.argsort(time_sec, kind="mergesort")
    rev_order = order[::-1]

    # Pre-allocate output feature arrays (float32)
    s_time_since_last_in = np.full(n, 999999.0, dtype=np.float32)
    s_time_since_last_out = np.full(n, 999999.0, dtype=np.float32)
    s_pass_through_ratio = np.full(n, -1.0, dtype=np.float32)
    s_running_balance = np.zeros(n, dtype=np.float32)
    s_in_out_count_ratio = np.zeros(n, dtype=np.float32)

    r_time_since_last_in = np.full(n, 999999.0, dtype=np.float32)
    r_time_since_last_out = np.full(n, 999999.0, dtype=np.float32)
    r_running_balance = np.zeros(n, dtype=np.float32)

    pair_prev_cnt = np.zeros(n, dtype=np.float32)
    pair_time_since_last = np.full(n, 999999.0, dtype=np.float32)
    rev_pair_prev_cnt = np.zeros(n, dtype=np.float32)
    rev_pair_time_since = np.full(n, 999999.0, dtype=np.float32)

    # Forward tracking state
    last_in_time = {}
    last_in_amt = {}
    last_out_time = {}
    last_out_amt = {}
    cum_in_amt = defaultdict(float)
    cum_out_amt = defaultdict(float)
    cum_in_cnt = defaultdict(int)
    cum_out_cnt = defaultdict(int)

    pair_cnt = defaultdict(int)
    pair_last_t = {}

    print(f"[LEDGER] Running forward chronological state stream over {n:,} transactions...")
    for idx in order:
        u = senders[idx]
        v = receivers[idx]
        a = amounts[idx]
        t = time_sec[idx]

        # Sender state
        if u in last_in_time:
            s_time_since_last_in[idx] = max(0.0, t - last_in_time[u])
            s_pass_through_ratio[idx] = a / (last_in_amt[u] + 1.0)
        if u in last_out_time:
            s_time_since_last_out[idx] = max(0.0, t - last_out_time[u])
        s_running_balance[idx] = cum_in_amt[u] - cum_out_amt[u]
        s_in_out_count_ratio[idx] = (cum_in_cnt[u] + 1.0) / (cum_out_cnt[u] + 1.0)

        # Receiver state
        if v in last_in_time:
            r_time_since_last_in[idx] = max(0.0, t - last_in_time[v])
        if v in last_out_time:
            r_time_since_last_out[idx] = max(0.0, t - last_out_time[v])
        r_running_balance[idx] = cum_in_amt[v] - cum_out_amt[v]

        # Pairwise and reverse pair (ping-pong)
        uv = (u, v)
        vu = (v, u)
        pair_prev_cnt[idx] = pair_cnt[uv]
        if uv in pair_last_t:
            pair_time_since_last[idx] = max(0.0, t - pair_last_t[uv])
        rev_pair_prev_cnt[idx] = pair_cnt[vu]
        if vu in pair_last_t:
            rev_pair_time_since[idx] = max(0.0, t - pair_last_t[vu])

        # State updates
        last_out_time[u] = t
        last_out_amt[u] = a
        cum_out_amt[u] += a
        cum_out_cnt[u] += 1

        last_in_time[v] = t
        last_in_amt[v] = a
        cum_in_amt[v] += a
        cum_in_cnt[v] += 1

        pair_cnt[uv] += 1
        pair_last_t[uv] = t

    print("[LEDGER] Running backward pass for receiver next outflow pass-through...")
    r_time_until_next_out = np.full(n, 999999.0, dtype=np.float32)
    r_next_out_pass_ratio = np.full(n, -1.0, dtype=np.float32)
    next_out_time = {}
    next_out_amt = {}

    for idx in rev_order:
        u = senders[idx]
        v = receivers[idx]
        a = amounts[idx]
        t = time_sec[idx]

        if v in next_out_time:
            r_time_until_next_out[idx] = max(0.0, next_out_time[v] - t)
            r_next_out_pass_ratio[idx] = next_out_amt[v] / (a + 1.0)

        next_out_time[u] = t
        next_out_amt[u] = a

    # Summary conduit metrics per account
    all_accounts = set(cum_in_cnt.keys()) | set(cum_out_cnt.keys())
    conduit_index = {}
    turnover_ratio = {}
    total_tx = {}
    for acc in all_accounts:
        tin = cum_in_amt[acc]
        tout = cum_out_amt[acc]
        cin = cum_in_cnt[acc]
        cout = cum_out_cnt[acc]
        conduit_index[acc] = abs(tin - tout) / (tin + tout + 1.0)
        turnover_ratio[acc] = (tout + 1.0) / (tin + 1.0)
        total_tx[acc] = cin + cout

    # Assemble feature dataframe
    feats = pd.DataFrame({
        "s_time_since_last_in_log": np.log1p(s_time_since_last_in).astype(np.float32),
        "s_time_since_last_out_log": np.log1p(s_time_since_last_out).astype(np.float32),
        "s_pass_through_ratio": s_pass_through_ratio,
        "s_is_exact_passthrough": ((s_pass_through_ratio >= 0.88) & (s_pass_through_ratio <= 1.05)).astype(np.float32),
        "s_is_fast_mule_conduit": ((s_time_since_last_in < 900) & (s_pass_through_ratio >= 0.80) & (s_pass_through_ratio <= 1.10)).astype(np.float32),
        "s_running_balance": s_running_balance,
        "s_in_out_count_ratio": s_in_out_count_ratio,

        "r_time_since_last_in_log": np.log1p(r_time_since_last_in).astype(np.float32),
        "r_time_since_last_out_log": np.log1p(r_time_since_last_out).astype(np.float32),
        "r_running_balance": r_running_balance,
        "r_time_until_next_out_log": np.log1p(r_time_until_next_out).astype(np.float32),
        "r_next_out_pass_ratio": r_next_out_pass_ratio,
        "r_is_rapid_relay_mule": ((r_time_until_next_out < 900) & (r_next_out_pass_ratio >= 0.80) & (r_next_out_pass_ratio <= 1.10)).astype(np.float32),

        "pair_prev_cnt": pair_prev_cnt,
        "pair_time_since_last_log": np.log1p(pair_time_since_last).astype(np.float32),
        "rev_pair_prev_cnt": rev_pair_prev_cnt,
        "rev_pair_time_since_log": np.log1p(rev_pair_time_since).astype(np.float32),

        "s_global_conduit_index": pd.Series(senders).map(conduit_index).fillna(0.0).values.astype(np.float32),
        "r_global_conduit_index": pd.Series(receivers).map(conduit_index).fillna(0.0).values.astype(np.float32),
        "s_global_turnover_ratio": pd.Series(senders).map(turnover_ratio).fillna(1.0).values.astype(np.float32),
        "r_global_turnover_ratio": pd.Series(receivers).map(turnover_ratio).fillna(1.0).values.astype(np.float32),
        "s_global_total_tx": pd.Series(senders).map(total_tx).fillna(0.0).values.astype(np.float32),
        "r_global_total_tx": pd.Series(receivers).map(total_tx).fillna(0.0).values.astype(np.float32),
    })

    print(f"[LEDGER] Generated {feats.shape[1]} bidirectional ledger features in {time.time() - t0:.1f}s.")
    return feats

def get_or_create_ledger_features(train_df: pd.DataFrame, test_df: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(LEDGER_FEATS_PARQUET):
        print(f"[LEDGER] Loading precomputed ledger features from {LEDGER_FEATS_PARQUET}")
        return pd.read_parquet(LEDGER_FEATS_PARQUET)

    cols = ['sender_id', 'receiver_id', 'amount', 'timestamp']
    full_df = pd.concat([train_df[cols], test_df[cols]], axis=0, ignore_index=True)
    feats = compute_bidirectional_ledger(full_df)
    feats.to_parquet(LEDGER_FEATS_PARQUET, compression="zstd", index=False)
    print(f"[LEDGER] Saved ledger features to {LEDGER_FEATS_PARQUET}")
    return feats

if __name__ == "__main__":
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    feats = get_or_create_ledger_features(train_df, test_df, force=True)
    print("Ledger feats sample:")
    print(feats.head(3))
