import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

LABELS = [
    'legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap',
    'account_takeover', 'phishing_collect', 'qr_swap', 'refund_abuse',
    'fake_cashback', 'loan_app_extortion', 'merchant_collusion'
]
L2I = {l: i for i, l in enumerate(LABELS)}

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
sept_mask = (month[tr_idx] == 9)
y_all = np.array([L2I[l] for l in df['label'].values[tr_idx]])
y_sept = y_all[sept_mask]

s2_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_2_oof.npy')
s3_oof = np.load(r'd:\ML Gradient Gemini\checkpoints_resilient\Stage_3_oof.npy')
blend = 0.5 * s2_oof + 0.5 * s3_oof
w_blend = np.array([1.0, 1.10, 0.393, 1.225, 0.933, 1.0, 0.264, 0.145, 0.203, 0.998, 1.017, 1.058])
preds = (blend[sept_mask] * w_blend).argmax(axis=1)

cm = confusion_matrix(y_sept, preds)
cm_df = pd.DataFrame(cm, index=['T_'+l for l in LABELS], columns=['P_'+l for l in LABELS])
print("=== Confusion Matrix (True vs Pred) ===")
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)
print(cm_df)

print("\n=== Classification Report ===")
print(classification_report(y_sept, preds, target_names=LABELS, digits=4))
