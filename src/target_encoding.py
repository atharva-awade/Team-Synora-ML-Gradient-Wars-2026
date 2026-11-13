import os
import gc
import time
import numpy as np
import pandas as pd
from src.config import TRAIN_PARQUET, TEST_PARQUET, CACHE_DIR

TE_FEATS_PARQUET = os.path.join(CACHE_DIR, "target_encoding_features.parquet")

def compute_historical_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes strict point-in-time historical fraud rates for sender, receiver, and device.
    Jul -> prior
    Aug -> Jul stats
    Sep -> Jul + Aug stats
    Test (Oct) -> Jul + Aug + Sep stats (Full 3-month history!)
    """
    print("[TARGET_ENCODING] Computing point-in-time entity risk features...", flush=True)
    t0 = time.time()
    n_train = len(train_df)
    n_test = len(test_df)
    n_total = n_train + n_test

    # Extract month flags for train
    ts_train = pd.to_datetime(train_df['timestamp'])
    month_train = ts_train.dt.month.values  # 7 = Jul, 8 = Aug, 9 = Sep

    y_train = (train_df['label_id'].values > 0).astype(np.float32)
    prior_rate = float(np.mean(y_train))
    print(f"  -> Global Train Fraud Prior Rate: {prior_rate:.4f} ({y_train.sum():,.0f} fraud rows)", flush=True)

    entities = ['sender_id', 'receiver_id', 'device_id']
    
    # Output matrices
    train_feats = np.zeros((n_train, len(entities) * 3), dtype=np.float32)
    test_feats = np.zeros((n_test, len(entities) * 3), dtype=np.float32)

    k_smooth = 15.0

    for col_idx, col in enumerate(entities):
        print(f"  -> Target encoding entity: {col}...", flush=True)
        col_vals_train = train_df[col].astype(str).values
        col_vals_test = test_df[col].astype(str).values

        # 1. Point-in-time calculation for Train
        # Month 7 (Jul): uses prior
        m7_mask = (month_train == 7)
        train_feats[m7_mask, col_idx*3] = prior_rate
        train_feats[m7_mask, col_idx*3 + 1] = 0.0
        train_feats[m7_mask, col_idx*3 + 2] = 0.0

        # Month 8 (Aug): uses Month 7 history
        s7_cnt = pd.Series(y_train[m7_mask]).groupby(col_vals_train[m7_mask]).count()
        s7_pos = pd.Series(y_train[m7_mask]).groupby(col_vals_train[m7_mask]).sum()
        s7_rate = (s7_pos + k_smooth * prior_rate) / (s7_cnt + k_smooth)

        m8_mask = (month_train == 8)
        m8_vals = pd.Series(col_vals_train[m8_mask])
        train_feats[m8_mask, col_idx*3] = m8_vals.map(s7_rate).fillna(prior_rate).values
        train_feats[m8_mask, col_idx*3 + 1] = m8_vals.map(s7_cnt).fillna(0.0).values
        train_feats[m8_mask, col_idx*3 + 2] = m8_vals.map(s7_pos).fillna(0.0).values

        # Month 9 (Sep): uses Month 7 + Month 8 history
        m78_mask = (month_train == 7) | (month_train == 8)
        s78_cnt = pd.Series(y_train[m78_mask]).groupby(col_vals_train[m78_mask]).count()
        s78_pos = pd.Series(y_train[m78_mask]).groupby(col_vals_train[m78_mask]).sum()
        s78_rate = (s78_pos + k_smooth * prior_rate) / (s78_cnt + k_smooth)

        m9_mask = (month_train == 9)
        m9_vals = pd.Series(col_vals_train[m9_mask])
        train_feats[m9_mask, col_idx*3] = m9_vals.map(s78_rate).fillna(prior_rate).values
        train_feats[m9_mask, col_idx*3 + 1] = m9_vals.map(s78_cnt).fillna(0.0).values
        train_feats[m9_mask, col_idx*3 + 2] = m9_vals.map(s78_pos).fillna(0.0).values

        # 2. Test (Oct): uses ALL of Jul + Aug + Sep history!
        s_all_cnt = pd.Series(y_train).groupby(col_vals_train).count()
        s_all_pos = pd.Series(y_train).groupby(col_vals_train).sum()
        s_all_rate = (s_all_pos + k_smooth * prior_rate) / (s_all_cnt + k_smooth)

        te_vals = pd.Series(col_vals_test)
        test_feats[:, col_idx*3] = te_vals.map(s_all_rate).fillna(prior_rate).values
        test_feats[:, col_idx*3 + 1] = te_vals.map(s_all_cnt).fillna(0.0).values
        test_feats[:, col_idx*3 + 2] = te_vals.map(s_all_pos).fillna(0.0).values

    col_names = []
    for ent in entities:
        col_names.extend([f"hist_{ent}_fraud_rate", f"hist_{ent}_tx_cnt", f"hist_{ent}_fraud_cnt"])

    all_feats = np.vstack([train_feats, test_feats])
    df_out = pd.DataFrame(all_feats, columns=col_names)
    
    # Log transforms for count columns
    for ent in entities:
        df_out[f"hist_{ent}_tx_cnt_log"] = np.log1p(df_out[f"hist_{ent}_tx_cnt"])
        df_out[f"hist_{ent}_fraud_cnt_log"] = np.log1p(df_out[f"hist_{ent}_fraud_cnt"])
        df_out[f"hist_{ent}_is_known_fraud"] = (df_out[f"hist_{ent}_fraud_cnt"] > 0).astype(np.float32)

    print(f"[TARGET_ENCODING] Generated {df_out.shape[1]} risk encoding features in {time.time() - t0:.1f}s.", flush=True)
    return df_out

def get_or_create_target_encoding_features(train_df: pd.DataFrame, test_df: pd.DataFrame, force=False) -> pd.DataFrame:
    if not force and os.path.exists(TE_FEATS_PARQUET):
        print(f"[TARGET_ENCODING] Loading precomputed risk features from {TE_FEATS_PARQUET}")
        return pd.read_parquet(TE_FEATS_PARQUET)

    feats = compute_historical_target_encoding(train_df, test_df)
    feats.to_parquet(TE_FEATS_PARQUET, compression="zstd", index=False)
    print(f"[TARGET_ENCODING] Saved risk features to {TE_FEATS_PARQUET}")
    return feats

if __name__ == "__main__":
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)
    feats = get_or_create_target_encoding_features(train_df, test_df, force=True)
    print("Risk feats sample:")
    print(feats.head(3))

