"""
Preprocessing pipeline for Diabetes dataset (Classification).
Replaces physiologically impossible zeros with NaN, imputes, scales,
performs stratified 80/20 split, and saves all artifacts to models/.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ── 2. Replace physiologically impossible zeros with NaN ──────────────────────
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    n_zeros = (df[col] == 0).sum()
    df = df.assign(**{col: df[col].replace(0, np.nan)})
    print(f"  {col}: replaced {n_zeros} zero(s) with NaN")

# ── 3. Separate features and target ───────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns if c != "Outcome"]
X = df[FEATURE_COLS].copy()
y_raw = df["Outcome"].copy()

# ── 4. Encode target ──────────────────────────────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(y_raw)
joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
print(f"\nTarget classes: {list(le.classes_)}  →  {list(range(len(le.classes_)))}")

# ── 5. Stratified 80/20 split ─────────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"\nSplit shapes:")
print(f"  X_train_raw : {X_train_raw.shape}")
print(f"  X_test_raw  : {X_test_raw.shape}")
print(f"  y_train     : {y_train.shape}")
print(f"  y_test      : {y_test.shape}")

# Class distribution
train_counts = np.bincount(y_train)
test_counts  = np.bincount(y_test)
print(f"\nClass distribution in y_train:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} (class {i}): {train_counts[i]}  ({train_counts[i]/len(y_train)*100:.1f}%)")
print(f"Class distribution in y_test:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} (class {i}): {test_counts[i]}  ({test_counts[i]/len(y_test)*100:.1f}%)")

# ── 6. Build & fit ColumnTransformer ──────────────────────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, FEATURE_COLS),
    ],
    remainder="drop",
)

preprocessor.fit(X_train_raw)
print(f"\nPreprocessor fitted on X_train_raw ({len(FEATURE_COLS)} numeric features).")

# ── 7. Transform for reference arrays ─────────────────────────────────────────
X_train_proc = preprocessor.transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

# ── 8. Save all artifacts ─────────────────────────────────────────────────────
artifacts = {
    "X_train_raw.pkl"   : (joblib.dump, X_train_raw,   os.path.join(MODELS_DIR, "X_train_raw.pkl")),
    "X_test_raw.pkl"    : (joblib.dump, X_test_raw,    os.path.join(MODELS_DIR, "X_test_raw.pkl")),
    "preprocessor.pkl"  : (joblib.dump, preprocessor,  os.path.join(MODELS_DIR, "preprocessor.pkl")),
    "label_encoder.pkl" : (joblib.dump, le,             os.path.join(MODELS_DIR, "label_encoder.pkl")),
}

for name, (fn, obj, path) in artifacts.items():
    fn(obj, path)

np.save(os.path.join(MODELS_DIR, "y_train.npy"),  y_train)
np.save(os.path.join(MODELS_DIR, "y_test.npy"),   y_test)
np.save(os.path.join(MODELS_DIR, "X_train.npy"),  X_train_proc)
np.save(os.path.join(MODELS_DIR, "X_test.npy"),   X_test_proc)

print("\nArtifacts saved:")
saved_files = [
    "X_train_raw.pkl", "X_test_raw.pkl",
    "y_train.npy",     "y_test.npy",
    "label_encoder.pkl",
    "preprocessor.pkl",
    "X_train.npy",     "X_test.npy",
]
for f in saved_files:
    full = os.path.join(MODELS_DIR, f)
    size = os.path.getsize(full)
    print(f"  models/{f}  ({size:,} bytes)")

print("\nPreprocessing complete.")
