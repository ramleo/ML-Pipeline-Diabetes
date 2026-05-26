"""
EDA Agent — Step 2: Data Inspection & Exploratory Data Analysis
Dataset: data/diabetes.csv | Target: Outcome | Task: Classification
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE, 'data', 'diabetes.csv')
PLOTS_DIR = os.path.join(BASE, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

TARGET = 'Outcome'
ZERO_INVALID_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# ── 1. Load data ───────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Columns: {list(df.columns)}")

# ── 2. Basic profile ───────────────────────────────────────────────────────
print(f"\n=== DTYPES & MISSING ===")
missing = df.isnull().sum()
print(pd.DataFrame({'dtype': df.dtypes, 'null_count': missing, 'null_pct': (missing/len(df)*100).round(2)}))

# ── 3. Zero-as-missing (physiologically impossible zeros) ──────────────────
print(f"\n=== ZERO-AS-MISSING (physiologically invalid) ===")
zero_counts = {}
for col in ZERO_INVALID_COLS:
    n = (df[col] == 0).sum()
    zero_counts[col] = n
    pct = n / len(df) * 100
    print(f"  {col}: {n} zeros ({pct:.1f}%)")

# ── 4. Class balance ───────────────────────────────────────────────────────
print(f"\n=== CLASS BALANCE (Outcome) ===")
vc = df[TARGET].value_counts().sort_index()
for label, count in vc.items():
    print(f"  Class {label}: {count} ({count/len(df)*100:.1f}%)")

# ── 5. Correlation with target ─────────────────────────────────────────────
print(f"\n=== CORRELATION WITH {TARGET} ===")
corr_series = df.corr()[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print(corr_series.round(3))

# ── 6. Outlier detection (IQR method) ─────────────────────────────────────
print(f"\n=== OUTLIERS (IQR method) ===")
feature_cols = [c for c in df.columns if c != TARGET]
outlier_counts = {}
for col in feature_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    outlier_counts[col] = n_out
    print(f"  {col}: {n_out} outliers")

# ── 7. PLOT 1 — Class distribution ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#4C9BE8', '#E8694C']
bars = ax.bar(['No Diabetes (0)', 'Diabetes (1)'], vc.values, color=colors, edgecolor='white', width=0.5)
for bar, val in zip(bars, vc.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val}\n({val/len(df)*100:.1f}%)', ha='center', va='bottom', fontsize=11)
ax.set_title('Class Distribution — Outcome', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Count')
ax.set_ylim(0, vc.max() * 1.18)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, 'class_distribution.png')
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {p1}")

# ── 8. PLOT 2 — Feature distributions (histograms 2×4 grid) ───────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].hist(df[col], bins=30, color='#4C9BE8', edgecolor='white', alpha=0.85)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].spines[['top','right']].set_visible(False)
plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, 'feature_distributions.png')
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p2}")

# ── 9. PLOT 3 — Correlation heatmap ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
p3 = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
plt.savefig(p3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p3}")

# ── 10. PLOT 4 — Boxplots vs Outcome (2×4 grid) ───────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
palette = {'0': '#4C9BE8', '1': '#E8694C'}
df_str = df.copy()
df_str[TARGET] = df_str[TARGET].astype(str)
for i, col in enumerate(feature_cols):
    sns.boxplot(data=df_str, x=TARGET, y=col, ax=axes[i], palette=palette, width=0.5)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Outcome (0=No, 1=Yes)')
    axes[i].spines[['top','right']].set_visible(False)
plt.suptitle('Feature Distributions by Outcome', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
p4 = os.path.join(PLOTS_DIR, 'boxplots.png')
plt.savefig(p4, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p4}")

print("\n=== EDA COMPLETE ===")
