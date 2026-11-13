import os
import gc
import pandas as pd
import numpy as np
from src.config import (
    TRAIN_CSV, TEST_CSV, ACCOUNTS_CSV, MERCHANTS_CSV,
    TRAIN_PARQUET, TEST_PARQUET, ACCOUNTS_PARQUET, MERCHANTS_PARQUET,
    LABEL2ID
)

def prepare_data_cache(force=False):
    """Loads CSVs, removes data traps, downcasts datatypes, and saves to fast Parquet."""
    if not force and os.path.exists(TRAIN_PARQUET) and os.path.exists(TEST_PARQUET):
        print("[CACHE] Clean parquet files already exist. Skipping raw conversion.")
        return

    print("[CACHE] Reading accounts.csv and merchants.csv...")
    accounts = pd.read_csv(ACCOUNTS_CSV)
    merchants = pd.read_csv(MERCHANTS_CSV)

    accounts.to_parquet(ACCOUNTS_PARQUET, compression="zstd", index=False)
    merchants.to_parquet(MERCHANTS_PARQUET, compression="zstd", index=False)
    print(f"  -> Saved {ACCOUNTS_PARQUET} ({len(accounts)} rows)")
    print(f"  -> Saved {MERCHANTS_PARQUET} ({len(merchants)} rows)")

    del accounts, merchants
    gc.collect()

    print("[CACHE] Processing train.csv...")
    train = pd.read_csv(TRAIN_CSV)
    orig_train_len = len(train)
    
    # 1. Purge the 147 October contaminated rows in train
    train = train[train['timestamp'] < '2026-10-01'].reset_index(drop=True)
    print(f"  -> Purged {orig_train_len - len(train)} contaminated October rows (Kept {len(train):,} Jul-Sep rows)")

    # 2. Drop the fatal 'app_version' column (causes tree split overfitting)
    if 'app_version' in train.columns:
        train.drop(columns=['app_version'], inplace=True)
        print("  -> Dropped 'app_version' column")

    # 3. Encode label to integer ID
    train['label_id'] = train['label'].map(LABEL2ID).astype(np.int8)

    # 4. Downcast floats and ints
    for col in train.select_dtypes(include=['float64']).columns:
        train[col] = train[col].astype(np.float32)
    for col in train.select_dtypes(include=['int64']).columns:
        if col != 'label_id':
            train[col] = train[col].astype(np.int32)

    train.to_parquet(TRAIN_PARQUET, compression="zstd", index=False)
    print(f"  -> Saved {TRAIN_PARQUET} ({train.shape})")
    del train
    gc.collect()

    print("[CACHE] Processing test.csv...")
    test = pd.read_csv(TEST_CSV)
    if 'app_version' in test.columns:
        test.drop(columns=['app_version'], inplace=True)
        print("  -> Dropped 'app_version' column from test")

    for col in test.select_dtypes(include=['float64']).columns:
        test[col] = test[col].astype(np.float32)
    for col in test.select_dtypes(include=['int64']).columns:
        test[col] = test[col].astype(np.int32)

    test.to_parquet(TEST_PARQUET, compression="zstd", index=False)
    print(f"  -> Saved {TEST_PARQUET} ({test.shape})")
    del test
    gc.collect()

    print("[CACHE] Phase 1 Data Foundation complete!")

if __name__ == "__main__":
    prepare_data_cache(force=True)

