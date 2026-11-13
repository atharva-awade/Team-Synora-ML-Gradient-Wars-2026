import os
import gc
import time
import numpy as np
import pandas as pd
from scipy import sparse
from src.config import TRAIN_PARQUET, TEST_PARQUET, CACHE_DIR

GRAPH_FEATS_PARQUET = os.path.join(CACHE_DIR, "graph_features.parquet")

def compute_graph_features(full_df: pd.DataFrame) -> pd.DataFrame:
    print("[GRAPH] Building Transductive Flow and Multiplex Graph...", flush=True)
    t0 = time.time()
    n = len(full_df)

    senders = full_df['sender_id'].astype(str).values
    receivers = full_df['receiver_id'].astype(str).values
    amounts = full_df['amount'].fillna(1.0).values.astype(np.float64)

    # Map all unique accounts to integer node indices [0 .. N_nodes - 1]
    unique_nodes = pd.Index(np.unique(np.concatenate([senders, receivers])))
    n_nodes = len(unique_nodes)
    node_to_idx = pd.Series(np.arange(n_nodes, dtype=np.int32), index=unique_nodes)

    u_idx = node_to_idx.loc[senders].values
    v_idx = node_to_idx.loc[receivers].values
    print(f"  -> Total unique graph nodes: {n_nodes:,} across {n:,} edges", flush=True)

    # 1. Sparse Adjacency Matrices
    ones = np.ones(n, dtype=np.float32)
    A_dir = sparse.csr_matrix((ones, (u_idx, v_idx)), shape=(n_nodes, n_nodes))
    W_dir = sparse.csr_matrix((amounts.astype(np.float32), (u_idx, v_idx)), shape=(n_nodes, n_nodes))
    A_bin = (A_dir > 0).astype(np.float32)

    # 2. Degree and Volume
    out_tx_cnt = np.asarray(A_dir.sum(axis=1)).ravel()
    in_tx_cnt = np.asarray(A_dir.sum(axis=0)).ravel()
    out_unique_deg = np.asarray(A_bin.sum(axis=1)).ravel()
    in_unique_deg = np.asarray(A_bin.sum(axis=0)).ravel()
    out_vol = np.asarray(W_dir.sum(axis=1)).ravel()
    in_vol = np.asarray(W_dir.sum(axis=0)).ravel()

    # 3. Reciprocity (A <-> B)
    recip_edges = A_bin.multiply(A_bin.T)
    node_recip_deg = np.asarray(recip_edges.sum(axis=1)).ravel()

    # 4. Filtered 3-Cycles (Excluding massive commercial hubs > 300 degree to avoid O(D^2) combinatorial explosion)
    print("  -> Computing directed 3-cycles on ring-scale subgraphs...", flush=True)
    ring_mask = (out_unique_deg <= 300) & (in_unique_deg <= 300)
    diag_filter = sparse.diags(ring_mask.astype(np.float32))
    A_ring = diag_filter @ A_bin @ diag_filter
    A2_ring = A_ring @ A_ring
    cycles_3_dir = np.asarray(A2_ring.multiply(A_ring.T).sum(axis=1)).ravel()

    # 5. Sparse Power-Iteration PageRank (alpha = 0.85)
    print("  -> Running power-iteration PageRank (20 rounds)...", flush=True)
    out_deg_safe = np.maximum(out_unique_deg, 1.0)
    inv_deg = sparse.diags(1.0 / out_deg_safe)
    P_trans = (inv_deg @ A_bin).T

    pr = np.full(n_nodes, 1.0 / n_nodes, dtype=np.float64)
    alpha = 0.85
    for _ in range(20):
        pr = alpha * (P_trans @ pr) + (1.0 - alpha) / n_nodes
        pr /= pr.sum()

    # 6. Connected Components (WCC and SCC)
    print("  -> Computing Strongly and Weakly Connected Components...", flush=True)
    _, wcc_labels = sparse.csgraph.connected_components(A_dir, directed=False)
    wcc_counts = pd.Series(wcc_labels).value_counts()
    wcc_sizes = wcc_counts.loc[wcc_labels].values

    _, scc_labels = sparse.csgraph.connected_components(A_dir, directed=True, connection="strong")
    scc_counts = pd.Series(scc_labels).value_counts()
    scc_sizes = scc_counts.loc[scc_labels].values

    # Assemble node-level mapped features
    feats = pd.DataFrame(index=full_df.index)

    def map_uv(arr, col_name):
        feats[f"s_{col_name}"] = arr[u_idx].astype(np.float32)
        feats[f"r_{col_name}"] = arr[v_idx].astype(np.float32)

    map_uv(out_tx_cnt, "out_tx_cnt")
    map_uv(in_tx_cnt, "in_tx_cnt")
    map_uv(out_unique_deg, "out_unique_deg")
    map_uv(in_unique_deg, "in_unique_deg")
    map_uv(np.log1p(out_vol), "out_vol_log")
    map_uv(np.log1p(in_vol), "in_vol_log")
    map_uv(node_recip_deg, "recip_deg")
    map_uv(cycles_3_dir, "cycles_3_dir")
    map_uv(pr * n_nodes, "pagerank_scaled")
    map_uv(wcc_sizes, "wcc_size")
    map_uv(scc_sizes, "scc_size")

    feats["s_fan_in_out_ratio"] = (feats["s_in_unique_deg"] + 1.0) / (feats["s_out_unique_deg"] + 1.0)
    feats["r_fan_in_out_ratio"] = (feats["r_in_unique_deg"] + 1.0) / (feats["r_out_unique_deg"] + 1.0)
    feats["pair_in_same_scc"] = ((scc_labels[u_idx] == scc_labels[v_idx]) & (scc_sizes[u_idx] >= 2)).astype(np.float32)

    # 7. Device Hardware Multiplex Features
    print("  -> Computing Device Multiplex & Impossible Travel Features...", flush=True)
    dev_str = full_df['device_id'].fillna("MISSING").astype(str)
    snd_str = full_df['sender_id'].astype(str)

    expected_dev = 'DV' + snd_str.str[2:].str.zfill(7)
    feats["dev_is_own"] = (dev_str == expected_dev).astype(np.float32)

    dev_n_senders = full_df.groupby('device_id')['sender_id'].nunique()
    dev_n_receivers = full_df.groupby('device_id')['receiver_id'].nunique()
    dev_n_cities = full_df.groupby('device_id')['device_city'].nunique()

    feats["dev_uniq_senders"] = dev_str.map(dev_n_senders).fillna(1).astype(np.float32)
    feats["dev_uniq_receivers"] = dev_str.map(dev_n_receivers).fillna(1).astype(np.float32)
    feats["dev_uniq_cities"] = dev_str.map(dev_n_cities).fillna(1).astype(np.float32)
    feats["dev_is_shared_farm"] = (feats["dev_uniq_senders"] >= 3).astype(np.float32)
    feats["dev_impossible_travel"] = (feats["dev_uniq_cities"] >= 3).astype(np.float32)

    snd_n_dev = full_df.groupby('sender_id')['device_id'].nunique()
    feats["snd_uniq_devices"] = snd_str.map(snd_n_dev).fillna(1).astype(np.float32)

    print(f"[GRAPH] Generated {feats.shape[1]} graph features in {time.time() - t0:.1f}s.", flush=True)
    return feats

def get_or_create_graph_features(train_df: pd.DataFrame, test_df: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(GRAPH_FEATS_PARQUET):
        print(f"[GRAPH] Loading precomputed graph features from {GRAPH_FEATS_PARQUET}")
        return pd.read_parquet(GRAPH_FEATS_PARQUET)

    cols = ['sender_id', 'receiver_id', 'amount', 'device_id', 'device_city']
    full_df = pd.concat([train_df[cols], test_df[cols]], axis=0, ignore_index=True)
    feats = compute_graph_features(full_df)
    feats.to_parquet(GRAPH_FEATS_PARQUET, compression="zstd", index=False)
    print(f"[GRAPH] Saved graph features to {GRAPH_FEATS_PARQUET}")
    return feats

if __name__ == "__main__":
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    feats = get_or_create_graph_features(train_df, test_df, force=True)
    print("Graph feats sample:")
    print(feats.head(3))
