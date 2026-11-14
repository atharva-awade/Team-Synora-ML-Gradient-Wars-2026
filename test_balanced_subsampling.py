import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

print("Loading cached base features...")
X_all = pd.read_parquet(r'd:\ML Gradient Gemini\checkpoints_resilient\base_features.parquet')

tr = pd.read_parquet(r'd:\ML Gradient Gemini\cache\train_clean.parquet')
te = pd.read_parquet(r'd:\ML Gradient Gemini\cache\test_clean.parquet')
tr['is_train'] = 1
te['is_train'] = 0
df = pd.concat([tr, te], ignore_index=True)
df['t'] = pd.to_datetime(df['timestamp'])
df['ts'] = (df['t'] - df['t'].min()).dt.total_seconds().astype('int64')
df = df.sort_values('ts', kind='stable').reset_index(drop=True)

is_tr = df['is_train'].values == 1
tr_idx = np.where(is_tr)[0]
month = df['t'].dt.month.values
y_all = np.array([L2I[l] for l in df['label'].values[tr_idx]])

# Train split: July + August (months 7, 8)
# Val split: September (month 9)
train_mask = (month[tr_idx] < 9)
val_mask = (month[tr_idx] == 9)

X_tr = X_all.iloc[tr_idx[train_mask]].reset_index(drop=True)
y_tr = y_all[train_mask]

X_va = X_all.iloc[tr_idx[val_mask]].reset_index(drop=True)
y_va = y_all[val_mask]

print(f"Train samples (Jul-Aug): {len(y_tr)}, Val samples (Sep): {len(y_va)}")
print(f"Train fraud: {(y_tr != 0).sum()} ({(y_tr != 0).mean():.2%})")

# Test 3 different legit subsampling fractions:
# 1) Standard 0.60 (our previous run)
# 2) Balanced 0.10 (fraud becomes ~30%)
# 3) Aggressive 0.05 (fraud becomes ~50%)
for legit_frac in [0.60, 0.10, 0.05]:
    rng = np.random.RandomState(42)
    keep = (y_tr != 0) | (rng.random(len(y_tr)) < legit_frac)
    X_sub = X_tr.iloc[keep].values
    y_sub = y_tr[keep]
    print(f"\n=======================================================")
    print(f"Evaluating legit_frac = {legit_frac}: Train size = {len(y_sub)} (Fraud: {(y_sub != 0).mean():.1%})")
    
    # Class weights for remaining samples
    cnt = np.bincount(y_sub, minlength=K).astype(float)
    w = (cnt.max() / np.maximum(cnt, 1.0)) ** 0.5
    w_sample = (w / w.min())[y_sub]
    
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=8,
        verbose=-1
    )
    clf.fit(X_sub, y_sub, sample_weight=w_sample)
    
    val_probs = clf.predict_proba(X_va)
    raw_preds = val_probs.argmax(axis=1)
    raw_f1 = f1_score(y_va, raw_preds, average='macro')
    print(f"Raw Argmax September Macro-F1: {raw_f1:.4f}")
    
    # Tune multipliers on validation
    w_best = np.ones(K)
    best_score = raw_f1
    for c in range(1, K):
        for mult in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
            w_test = w_best.copy()
            w_test[c] = mult
            s = f1_score(y_va, (val_probs * w_test).argmax(axis=1), average='macro')
            if s > best_score:
                best_score = s
                w_best[c] = mult
    print(f"Tuned September Macro-F1: {best_score:.4f}")
