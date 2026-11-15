import pandas as pd

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

sub = pd.read_csv(r'D:\ML Gradient Gemini\submissions\submission_sprint3_crown_063.csv')
actual = sub['label'].value_counts()

print(f"{'Label':<22} | {'Sprint 3 (0.5054)':<18} | {'True Expected':<14} | {'Difference':<12} | {'Coverage Ratio':<15}")
print('-' * 88)
for l, exp in EXPECTED_TEST_COUNTS.items():
    act = actual.get(l, 0)
    diff = act - exp
    ratio = act / exp
    print(f"{l:<22} | {act:<18} | {exp:<14} | {diff:+12} | {ratio:<15.1%}")

