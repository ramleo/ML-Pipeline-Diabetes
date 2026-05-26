"""
Optimization Agent — Steps 4–6: Model Training, Hyperparameter Tuning & Evaluation
Task type: Classification
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

BASE = "/Users/wrks/Downloads/Claude-documentation/Projects/ML-Diabetes/ML-Pipeline-Diabetes_20260526_103441"

# ── Load artifacts ────────────────────────────────────────────────────────────
X_train = joblib.load(f"{BASE}/models/X_train_raw.pkl")
X_test  = joblib.load(f"{BASE}/models/X_test_raw.pkl")
y_train = np.load(f"{BASE}/models/y_train.npy")
y_test  = np.load(f"{BASE}/models/y_test.npy")
preprocessor = joblib.load(f"{BASE}/models/preprocessor.pkl")

print(f"X_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_train dist  : {dict(zip(*np.unique(y_train, return_counts=True)))}")

# ── Candidate models & grids ─────────────────────────────────────────────────
candidates = {
    "LogisticRegression": (
        LogisticRegression(solver="saga", max_iter=1000, random_state=42),
        {"C": [0.1, 1, 10]},
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
    ),
    "SVC": (
        SVC(probability=True, random_state=42),
        {"C": [1, 10], "kernel": ["rbf", "linear"]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
}

# ── Grid search over all candidates ──────────────────────────────────────────
results = []

for name, (model, grid) in candidates.items():
    print(f"\n[{name}] Running GridSearchCV …")
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    param_grid = {f"model__{k}": v for k, v in grid.items()}
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
    cv_acc = gs.best_score_
    print(f"  Best params : {best_params}")
    print(f"  CV accuracy : {cv_acc:.4f}")
    results.append({"model": name, "best_params": best_params, "cv_accuracy": cv_acc, "gs": gs})

# ── Select winner ─────────────────────────────────────────────────────────────
best = max(results, key=lambda r: r["cv_accuracy"])
print(f"\n{'='*60}")
print(f"WINNER: {best['model']}  CV accuracy = {best['cv_accuracy']:.4f}")
print(f"Best params: {best['best_params']}")

# ── Refit winner on full training set ────────────────────────────────────────
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

model_map = {
    "LogisticRegression": LogisticRegression(solver="saga", max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

winner_model = model_map[best["model"]]
winner_model.set_params(**best["best_params"])

final_pipeline = Pipeline([("preprocessor", preprocessor), ("model", winner_model)])
final_pipeline.fit(X_train, y_train)

# ── Evaluate on test set ──────────────────────────────────────────────────────
y_pred = final_pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nTest accuracy : {test_acc:.4f}")
print("\nClassification Report:")
print(report)

# ── Save final pipeline ───────────────────────────────────────────────────────
out_path = f"{BASE}/models/final_pipeline.pkl"
joblib.dump(final_pipeline, out_path)
print(f"\nfinal_pipeline.pkl saved → {out_path}")

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS TABLE")
print("="*60)
df_results = pd.DataFrame([
    {"Model": r["model"], "Best Params": str(r["best_params"]), "CV Accuracy": f"{r['cv_accuracy']:.4f}"}
    for r in results
]).sort_values("CV Accuracy", ascending=False)
print(df_results.to_string(index=False))
print("="*60)
