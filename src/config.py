import os

# Base paths
PROJECT_DIR = r"d:\ML Gradient Gemini"
RAW_DATA_DIR = r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings"
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
SUBMISSIONS_DIR = os.path.join(PROJECT_DIR, "submissions")

# Raw data files
TRAIN_CSV = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_CSV = os.path.join(RAW_DATA_DIR, "test.csv")
ACCOUNTS_CSV = os.path.join(RAW_DATA_DIR, "accounts.csv")
MERCHANTS_CSV = os.path.join(RAW_DATA_DIR, "merchants.csv")
SAMPLE_SUB_CSV = os.path.join(RAW_DATA_DIR, "sample_submission.csv")

# Parquet cache files
TRAIN_PARQUET = os.path.join(CACHE_DIR, "train_clean.parquet")
TEST_PARQUET = os.path.join(CACHE_DIR, "test_clean.parquet")
ACCOUNTS_PARQUET = os.path.join(CACHE_DIR, "accounts.parquet")
MERCHANTS_PARQUET = os.path.join(CACHE_DIR, "merchants.parquet")
FEATURES_PARQUET = os.path.join(CACHE_DIR, "features_master.parquet")

# Competition Classes (12 Classes)
LABELS = [
    'legit',
    'mule_account',
    'velocity_bot',
    'layering_chain',
    'sim_swap',
    'account_takeover',
    'phishing_collect',
    'qr_swap',
    'refund_abuse',
    'fake_cashback',
    'loan_app_extortion',
    'merchant_collusion'
]

LABEL2ID = {lbl: idx for idx, lbl in enumerate(LABELS)}
ID2LABEL = {idx: lbl for idx, lbl in enumerate(LABELS)}

# Prior class counts expected in test.csv (Total 260,582 rows)
EXPECTED_TEST_COUNTS = {
    'legit': 250998,
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

SEED = 42
NUM_THREADS = 16

