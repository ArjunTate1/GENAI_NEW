"""
Run from genai_day folder:  python verify_features.py
Edit CSV_PATH / TEXT_COL below to point at your TRAINING data.
Compares my feature extraction against the scaler's saved mean/std.
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from app_features import compute_linguistic_dict  # same code as app.py

CSV_PATH = "data.csv"      # <-- your training csv
TEXT_COL = "text"          # <-- column with the text

sc = joblib.load("artifacts/linguistic_scaler.joblib")
df = pd.read_csv(CSV_PATH)
feats = pd.DataFrame([compute_linguistic_dict(str(t)) for t in df[TEXT_COL]])

print("Scaler mean/std per slot:")
for i, (m, s) in enumerate(zip(sc.mean_, sc.scale_)):
    print(f"slot {i}: mean={m:.4f} std={s:.4f}")

print("\nMy candidates (mean / std) - find the closest match for each slot:")
print(feats.agg(["mean", "std"]).T.round(4).to_string())
