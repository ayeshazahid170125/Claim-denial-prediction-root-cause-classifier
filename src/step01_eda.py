"""
STEP 01 — Data Load & Exploration (Day 1)
WellMind Data Solutions — Claim Denial Prediction System
Run: python step01_eda.py
"""
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
REPORT_DIR = BASE_DIR / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = (
    BASE_DIR
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "2024"
    / "PHY_R26_P05_V10_D24_Prov_Svc.csv"
)

NUMERIC_SUMMARY_PATH = REPORT_DIR / "eda_numeric_summary.csv"
CATEGORICAL_SUMMARY_PATH = REPORT_DIR / "eda_categorical_summary.csv"
COLUMN_PROFILE_PATH = REPORT_DIR / "eda_column_profile.csv"

# ============================================================
# DATA LOAD
# ============================================================
print("=" * 50)
print("DATA LOAD")
print("=" * 50)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"File nahi mili: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"✅ Data successfully load ho gaya!")

# ============================================================
# 1. SHAPE
# ============================================================
print("\n" + "=" * 50)
print("1️⃣  SHAPE (Rows x Columns)")
print("=" * 50)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

# ============================================================
# 2. COLUMNS LIST
# ============================================================
print("\n" + "=" * 50)
print("2️⃣  COLUMN NAMES")
print("=" * 50)
for i, col in enumerate(df.columns, 1):
    print(f"  {i:>3}. {col}")

# ============================================================
# 3. HEAD (First 5 rows)
# ============================================================
print("\n" + "=" * 50)
print("3️⃣  HEAD — Pehli 5 Rows")
print("=" * 50)
print(df.head().to_string())

# ============================================================
# 4. TAIL (Last 5 rows)
# ============================================================
print("\n" + "=" * 50)
print("4️⃣  TAIL — Aakhri 5 Rows")
print("=" * 50)
print(df.tail().to_string())

# ============================================================
# 5. INFO
# ============================================================
print("\n" + "=" * 50)
print("5️⃣  INFO — Column Types & Non-Null Counts")
print("=" * 50)
df.info()

# ============================================================
# 6. DESCRIBE — Numeric Columns
# ============================================================
print("\n" + "=" * 50)
print("6️⃣  DESCRIBE — Numeric Columns (Statistics)")
print("=" * 50)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Total Numeric Columns: {len(numeric_cols)}")
numeric_summary = df[numeric_cols].describe().T
numeric_summary["missing_count"] = df[numeric_cols].isnull().sum()
numeric_summary["missing_pct"] = (numeric_summary["missing_count"] / len(df) * 100).round(2)
print(numeric_summary.to_string())

# ============================================================
# 7. DESCRIBE — Categorical Columns
# ============================================================
print("\n" + "=" * 50)
print("7️⃣  DESCRIBE — Categorical Columns (Object/String)")
print("=" * 50)
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Total Categorical Columns: {len(cat_cols)}")
if cat_cols:
    categorical_summary = df[cat_cols].describe().T
    categorical_summary["missing_count"] = df[cat_cols].isnull().sum()
    categorical_summary["missing_pct"] = (categorical_summary["missing_count"] / len(df) * 100).round(2)
    print(categorical_summary.to_string())
else:
    categorical_summary = pd.DataFrame()

# ============================================================
# 8. NUMERIC vs CATEGORICAL — Summary Table
# ============================================================
print("\n" + "=" * 50)
print("8️⃣  COLUMN TYPE BREAKDOWN")
print("=" * 50)
dtype_summary = df.dtypes.value_counts()
print(dtype_summary.to_string())
print(f"\n  Numeric  Columns : {len(numeric_cols)}")
print(f"  Category Columns : {len(cat_cols)}")
other_cols = [c for c in df.columns if c not in numeric_cols + cat_cols]
print(f"  Other    Columns : {len(other_cols)} {other_cols if other_cols else ''}")

# ============================================================
# 9. CATEGORICAL VALUE COUNTS
# ============================================================
print("\n" + "=" * 50)
print("9️⃣  CATEGORICAL COLUMNS — Unique Values & Frequencies")
print("=" * 50)

MAX_UNIQUE_TO_SHOW = 20  # agar zyada unique hain toh sirf top dikhayenge

for col in cat_cols:
    n_unique = df[col].nunique()
    print(f"\n📌 Column: '{col}'")
    print(f"   Unique Values: {n_unique}")
    vc = df[col].value_counts(dropna=False)
    if n_unique <= MAX_UNIQUE_TO_SHOW:
        print(vc.to_string())
    else:
        print(f"   (Top 20 dikhaye ja rahe hain out of {n_unique})")
        print(vc.head(20).to_string())

# ============================================================
# 10. SAVE EDA DELIVERABLES
# ============================================================
print("\n" + "=" * 50)
print("🔖 SAVING EDA SUMMARY FILES")
print("=" * 50)

numeric_summary.to_csv(NUMERIC_SUMMARY_PATH)
print("  ✅ Saved: eda_numeric_summary.csv")

if not categorical_summary.empty:
    categorical_summary.to_csv(CATEGORICAL_SUMMARY_PATH)
    print("  ✅ Saved: eda_categorical_summary.csv")

column_profile = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "non_null_count": [df[c].notna().sum() for c in df.columns],
    "missing_count": [df[c].isna().sum() for c in df.columns],
    "missing_pct": [(df[c].isna().sum() / len(df) * 100) for c in df.columns],
    "unique_count": [df[c].nunique(dropna=True) for c in df.columns],
})
column_profile["missing_pct"] = column_profile["missing_pct"].round(2)
column_profile.to_csv(COLUMN_PROFILE_PATH, index=False)
print("  ✅ Saved: eda_column_profile.csv")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 50)
print("✅ DAY 1 EDA COMPLETE!")
print("=" * 50)
