import os
import gc
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from src.config import TRAIN_PARQUET, TEST_PARQUET, CACHE_DIR

SPECTRAL_FEATS_PARQUET = os.path.join(CACHE_DIR, "spectral_features.parquet")

def compute_spectral_embeddings(full_df: pd.DataFrame, n_components: int = 32) -> pd.DataFrame:
    print(f"[SPECTRAL] Computing {n_components}-dim Transductive SVD Bipartite Embeddings...", flush=True)
    t0 = time.time()
    n = len(full_df)

    # Filter to sender-merchant or sender-receiver edges
    senders = full_df['sender_id'].astype(str).values
    receivers = full_df['receiver_id'].astype(str).values

    unique_senders = pd.Index(np.unique(senders))
    unique_receivers = pd.Index(np.unique(receivers))
    n_s = len(unique_senders)
    n_r = len(unique_receivers)

    s_map = pd.Series(np.arange(n_s, dtype=np.int32), index=unique_senders)
    r_map = pd.Series(np.arange(n_r, dtype=np.int32), index=unique_receivers)

    u_idx = s_map.loc[senders].values
    v_idx = r_map.loc[receivers].values

    # Construct sparse bipartite matrix
    ones = np.ones(n, dtype=np.float32)
    B = sparse.csr_matrix((ones, (u_idx, v_idx)), shape=(n_s, n_r))

    print(f"  -> Sparse Bipartite Matrix shape: {B.shape}, nonzeros: {B.nnz:,}", flush=True)
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", n_iter=5, random_state=42)
    U_s = svd.fit_transform(B)  # shape: (n_s, n_components)
    V_r = svd.components_.T     # shape: (n_r, n_components)

    # Normalize rows to unit L2 norm
    U_norm = np.linalg.norm(U_s, axis=1, keepdims=True) + 1e-12
    V_norm = np.linalg.norm(V_r, axis=1, keepdims=True) + 1e-12
    U_unit = U_s / U_norm
    V_unit = V_r / V_norm

    # For each transaction, compute cosine similarity between sender and receiver in latent space
    u_vecs = U_unit[u_idx]
    v_vecs = V_unit[v_idx]
    cos_sim = np.sum(u_vecs * v_vecs, axis=1).astype(np.float32)
    l2_dist = np.linalg.norm(u_vecs - v_vecs, axis=1).astype(np.float32)

    feats = pd.DataFrame(index=full_df.index)
    feats['spectral_cos_sim'] = cos_sim
    feats['spectral_l2_dist'] = l2_dist
    # First 4 principal singular dimensions for sender and receiver
    for k in range(4):
        feats[f's_svd_{k}'] = U_unit[u_idx, k].astype(np.float32)
        feats[f'r_svd_{k}'] = V_unit[v_idx, k].astype(np.float32)

    print(f"[SPECTRAL] Generated {feats.shape[1]} spectral features in {time.time() - t0:.1f}s.", flush=True)
    return feats

def get_or_create_spectral_features(train_df: pd.DataFrame, test_df: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(SPECTRAL_FEATS_PARQUET):
        print(f"[SPECTRAL] Loading precomputed spectral features from {SPECTRAL_FEATS_PARQUET}")
        return pd.read_parquet(SPECTRAL_FEATS_PARQUET)

    cols = ['sender_id', 'receiver_id']
    full_df = pd.concat([train_df[cols], test_df[cols]], axis=0, ignore_index=True)
    feats = compute_spectral_embeddings(full_df)
    feats.to_parquet(SPECTRAL_FEATS_PARQUET, compression="zstd", index=False)
    print(f"[SPECTRAL] Saved spectral features to {SPECTRAL_FEATS_PARQUET}")
    return feats

if __name__ == "__main__":
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    feats = get_or_create_spectral_features(train_df, test_df, force=True)
    print("Spectral feats sample:")
    print(feats.head(3))

