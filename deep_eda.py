import pandas as pd
import numpy as np

# Load full train (read needed columns to be fast and memory efficient)
cols = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'device_id', 'amount', 'txn_type', 'merchant_category', 'device_city', 'sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee', 'pin_retries', 'login_to_txn_sec', 'balance_before', 'ip_risk_score', 'qr_vpa_mismatch', 'label']
print("Loading train...")
df = pd.read_csv(r'C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings\train.csv', usecols=cols)
print("Loaded:", len(df))

# Check distribution of amount by label
print("\n=== AMOUNT QUANTILES BY LABEL ===")
for lbl, grp in df.groupby('label'):
    amt = grp['amount']
    print(f"{lbl:<20}: count={len(amt):>6}, p10={amt.quantile(0.1):>8.1f}, p50={amt.median():>8.1f}, p90={amt.quantile(0.9):>8.1f}, max={amt.max():>9.1f}, is_int%={(amt % 1 == 0).mean()*100:>5.1f}%")

# Check txn_type breakdown by label
print("\n=== TXN_TYPE BREAKDOWN BY LABEL (%) ===")
ct = pd.crosstab(df['label'], df['txn_type'], normalize='index') * 100
print(ct.round(1))

# Check receiver_type breakdown by label
print("\n=== RECEIVER_TYPE BREAKDOWN BY LABEL (%) ===")
ct2 = pd.crosstab(df['label'], df['receiver_type'], normalize='index') * 100
print(ct2.round(1))

# Check qr_vpa_mismatch
print("\n=== QR_VPA_MISMATCH BY LABEL (%) ===")
print((df.groupby('label')['qr_vpa_mismatch'].mean() * 100).round(2))

