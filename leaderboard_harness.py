#!/usr/bin/env python3
"""
Leaderboard & Submission Harness for UPI Fraud Rings (ML Gradient Wars 2026)
Platform: hack4hack.com
"""

import sys
import json
import urllib.request
from datetime import datetime
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

LEADERBOARD_API = "https://hack4hack.com/api/hackathons/ml-gradient-wars-2026-5tns/leaderboard"

CANONICAL_LABELS = {
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
}

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

def fetch_leaderboard():
    """Fetches the live leaderboard and prints current status."""
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print("  UPI FRAUD RINGS — LIVE LEADERBOARD TRACKER")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 70)

    req = urllib.request.Request(LEADERBOARD_API, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'})
    try:
        res = urllib.request.urlopen(req, timeout=10)
        data = json.loads(res.read().decode('utf-8'))
    except Exception as e:
        print(f"[ERROR] Could not fetch leaderboard: {e}")
        return

    print(f"Metric: {data.get('metric')} | Refresh: Every {int(data.get('refreshHours', 0.25)*60)} mins")
    print(f"Last platform update: {data.get('updatedAt')} | Next update: {data.get('nextUpdate')}")
    print(f"Total Participating Teams: {len(data.get('entries', []))}\n")

    print(f"{'Rank':<6}{'Team':<30}{'Macro-F1':<14}{'Entries'}")
    print("-" * 58)
    for entry in data.get('entries', [])[:15]:
        medal = "🥇 " if entry['rank'] == 1 else ("🥈 " if entry['rank'] == 2 else ("🥉 " if entry['rank'] == 3 else "   "))
        print(f"{entry['rank']:<2} {medal}{entry['team']:<26}{entry['score']:<14.4f}{entry.get('entries', '-')}")
    print("=" * 70)


def validate_submission(submission_csv_path: str, test_csv_path: str = r"C:\Users\acer\Downloads\upi-fraud-rings-gradient-wars\upi-fraud-rings\test.csv"):
    """Thoroughly checks a submission file against all competition constraints."""
    print(f"\n[VALIDATING SUBMISSION]: {submission_csv_path}")
    df = pd.read_csv(submission_csv_path)

    # 1. Header check
    if list(df.columns) != ['id', 'label']:
        print(f"❌ FAIL: Columns must be exactly ['id', 'label'], found {list(df.columns)}")
        return False
    print("✓ PASS: Correct header ('id', 'label')")

    # 2. Row count check
    if len(df) != 260582:
        print(f"❌ FAIL: Row count is {len(df)}, must be exactly 260,582")
        return False
    print("✓ PASS: Row count is exactly 260,582")

    # 3. ID matching check
    test_df = pd.read_csv(test_csv_path, usecols=['txn_id'])
    test_ids = set(test_df['txn_id'])
    sub_ids = set(df['id'])
    if test_ids != sub_ids:
        print(f"❌ FAIL: IDs do not match test.csv txn_ids!")
        return False
    print("✓ PASS: All 260,582 IDs match test.csv txn_ids perfectly")

    # 4. Canonical labels check
    labels_found = set(df['label'].dropna())
    invalid_labels = labels_found - CANONICAL_LABELS
    if invalid_labels:
        print(f"❌ FAIL: Found invalid labels: {invalid_labels}")
        return False
    print("✓ PASS: All labels belong to the 12 canonical competition classes")

    # 5. Class distribution sanity check
    counts = df['label'].value_counts()
    print("\n--- Predicted Class Counts vs. Expected Priors ---")
    all_ok = True
    for label in sorted(CANONICAL_LABELS):
        cnt = counts.get(label, 0)
        expected = EXPECTED_TEST_COUNTS[label]
        ratio = cnt / expected if expected else 0
        status = "✓ OK"
        if cnt == 0:
            status = "❌ ZERO PREDICTIONS (CRITICAL MACRO-F1 KILLER)"
            all_ok = False
        elif ratio < 0.2 or ratio > 3.0:
            status = "⚠️ SKEWED RATIO (>3x or <0.2x of expected)"
        print(f"  {label:<22}: {cnt:>6} rows (Expected ~{expected:>6}, Ratio: {ratio:.2f}) -> {status}")

    if all_ok:
        print("\n✅ SUBMISSION IS FULLY VALIDATED AND SAFE TO UPLOAD!")
    else:
        print("\n❌ SUBMISSION HAS SERIOUS ISSUES THAT WILL HURT MACRO-F1!")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate_submission(sys.argv[1])
    else:
        fetch_leaderboard()

