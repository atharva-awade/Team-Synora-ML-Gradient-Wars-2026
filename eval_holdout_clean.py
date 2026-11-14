import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

print("Loading clean data...")
tr = pd.read_parquet(r'd:\ML Gradient Gemini\cache\train_clean.parquet')
te = pd.read_parquet(r'd:\ML Gradient Gemini\cache\test_clean.parquet')
tr['is_train'] = 1
te['is_train'] = 0
te['label'] = 'legit'
te['label_id'] = 0

df = pd.concat([tr, te], ignore_index=True)
df['t'] = pd.to_datetime(df['timestamp'])
df['ts'] = (df['t'] - df['t'].min()).dt.total_seconds().astype('int64')
df = df.sort_values('ts', kind='stable').reset_index(drop=True)

is_tr = df['is_train'].values == 1
tr_idx = np.where(is_tr)[0]
month = df['t'].dt.month.values
last = (month[tr_idx] == 9)
y_all = np.array([L2I[l] for l in df['label'].values[tr_idx]])
y_sept = y_all[last]

print(f"Total train: {len(tr_idx)}, September holdout: {len(y_sept)}")

# Load OOF arrays
print("Loading OOF arrays...")
s1_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_1_oof.npy')
s2_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')
s3_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')

def eval_raw(P, name):
    preds = P[last].argmax(axis=1)
    macro = f1_score(y_sept, preds, average='macro')
    per_cls = f1_score(y_sept, preds, average=None)
    print(f"\n--- {name} RAW ARGMAX Macro-F1: {macro:.4f} ---")
    for l, f in zip(LABELS, per_cls):
        print(f"  {l:20s}: {f:.4f}")
    return macro

def tune_mult(P_sept, y_val, n_cycles=4):
    best_w = np.ones(K, dtype=np.float64)
    best_score = f1_score(y_val, (P_sept * best_w).argmax(axis=1), average='macro')
    print(f"  Baseline Macro-F1: {best_score:.4f}")
    for cycle in range(n_cycles):
        for c in range(1, K):
            best_wc = best_w[c]
            candidates = np.unique(np.concatenate([
                np.linspace(0.2, 3.0, 30),
                np.linspace(max(0.1, best_wc * 0.6), min(12.0, best_wc * 1.6), 25)
            ]))
            for cand in candidates:
                w_try = best_w.copy()
                w_try[c] = cand
                preds = (P_sept * w_try).argmax(axis=1)
                score = f1_score(y_val, preds, average='macro') - 1e-4 * ((cand - 1.0) ** 2)
                if score > best_score:
                    best_score = score
                    best_wc = cand
            best_w[c] = best_wc
        print(f"    Cycle {cycle + 1}/{n_cycles} -> Tuned OOF Macro-F1: {best_score:.4f}")
    return best_w, best_score

eval_raw(s1_oof, "Stage 1")
eval_raw(s2_oof, "Stage 2")
eval_raw(s3_oof, "Stage 3")

blend = 0.5 * s2_oof + 0.5 * s3_oof
eval_raw(blend, "Stage 2 + 3 Blend")

print("\nTuning multipliers for Stage 2 + 3 Blend on September Holdout...")
w_blend, sc_blend = tune_mult(blend[last], y_sept)

preds_tuned = (blend[last] * w_blend).argmax(axis=1)
per_cls = f1_score(y_sept, preds_tuned, average=None)
print(f"\n=== TUNED BLEND September Macro-F1: {sc_blend:.4f} ===")
for l, f, w in zip(LABELS, per_cls, w_blend):
    print(f"  {l:20s}: F1={f:.4f}  (weight={w:.3f})")
