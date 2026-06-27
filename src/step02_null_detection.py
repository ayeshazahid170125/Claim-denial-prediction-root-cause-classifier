"""
STEP 02 — Null Values Detection (Day 2)
WellMind Data Solutions — Claim Denial Prediction System
Run: python step02_null_detection.py
"""
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
REPORT_DIR = BASE_DIR / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = (
    BASE_DIR
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "2024"
    / "PHY_R26_P05_V10_D24_Prov_Svc.csv"
)
SUMMARY_PATH = REPORT_DIR / "null_detection_summary.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File nahi mili: {DATA_PATH}")

# Extension check karke load karo
if DATA_PATH.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_PATH, low_memory=False)
elif DATA_PATH.suffix.lower() in {".xlsx", ".xls"}:
    df = pd.read_excel(DATA_PATH)
else:
    raise ValueError("Sirf .csv ya .xlsx files supported hain")

print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   File: {DATA_PATH}\n")

# ============================================================
# NULL PATTERNS — Har type ki null values detect karni hain
# ============================================================
# Real-world data mein sirf NaN nahi hoti, ye bhi hoti hain:
NULL_PATTERNS = [
    None,       # Python None
    np.nan,     # NumPy NaN
    'nan',      # String "nan"
    'NaN',      # String "NaN"
    'NAN',      # String "NAN"
    'none',     # String "none"
    'None',     # String "None"
    'NONE',     # String "NONE"
    'null',     # String "null"
    'NULL',     # String "NULL"
    'NA',       # String "NA"
    'na',       # String "na"
    'N/A',      # String "N/A"
    'n/a',      # String "n/a"
    '',         # Empty string
    ' ',        # Single space
    '  ',       # Double space
    '-',        # Dash
    '--',       # Double dash
    '.',        # Dot
    '?',        # Question mark
    'unknown',  # unknown
    'Unknown',  # Unknown
    'UNKNOWN',  # UNKNOWN
    'missing',  # missing
    'Missing',  # Missing
    'MISSING',  # MISSING
    '#N/A',     # Excel style
    '#NA',      # Excel style
    # "0" intentionally excluded: Medicare data mein zero valid value bhi ho sakti hai.
    # Numeric zero values neeche sirf review-only signal ke tor par flag honge.
]

# ============================================================
# HELPER FUNCTION
# ============================================================
def detect_all_nulls(series, col_name, dtype):
    """Har column mein sari tarah ki nulls detect karta hai"""
    results = {}
    total = len(series)

    # 1. Standard NaN / None (pandas built-in)
    standard_null = series.isnull().sum()
    results['Standard NaN/None'] = standard_null

    # 2. String-based null patterns (sirf object columns mein)
    if dtype == 'object' or str(dtype) == 'str':
        for pattern in NULL_PATTERNS:
            if pattern is None or (isinstance(pattern, float) and np.isnan(pattern)):
                continue  # ye upar count ho gaya
            count = (series.astype(str).str.strip() == str(pattern).strip()).sum()
            if count > 0:
                results[f'String "{pattern}"'] = count

        # 3. Whitespace-only strings
        whitespace_count = series.dropna().astype(str).str.strip().eq('').sum()
        if whitespace_count > 0:
            results['Whitespace Only'] = whitespace_count

        # 4. Single character placeholders ('.', '-', '?')
        placeholder_chars = ['.', '-', '?', '*', 'x', 'X']
        for ch in placeholder_chars:
            count = series.dropna().astype(str).str.strip().eq(ch).sum()
            if count > 0:
                results[f'Placeholder "{ch}"'] = count

    # 5. Numeric columns: 0 values
    # NOTE: Zero values suspicious hain, lekin direct missing/null nahi hoti.
    # Cleaning phase mein context dekh ke decide karna.
    if dtype in ['int64', 'float64']:
        zero_count = (series == 0).sum()
        if zero_count > 0:
            results['Zero (0) values - review only'] = zero_count

        # 6. Negative values (invalid in some contexts)
        neg_count = (series < 0).sum()
        if neg_count > 0:
            results['Negative values'] = neg_count

        # 7. Infinite values
        if dtype == 'float64':
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                results['Inf / -Inf'] = inf_count

    return results, total


# ============================================================
# MAIN NULL DETECTION — 28 Columns
# ============================================================
print("\n" + "=" * 70)
print("🔍 NULL DETECTION — SARI 28 COLUMNS")
print("=" * 70)
print(f"{'Column':<35} {'Dtype':<10} {'Null Type':<30} {'Count':>10} {'%':>8}")
print("-" * 97)

# Final summary ke liye
summary_rows = []
cols_with_no_nulls = []

for col in df.columns:
    dtype = df[col].dtype
    null_results, total = detect_all_nulls(df[col], col, dtype)

    # True null/missing aur review-only flags ko separate rakho.
    has_true_null = any(
        v > 0 and "review only" not in k
        for k, v in null_results.items()
    )
    has_any_flag = any(v > 0 for v in null_results.values())

    if not has_true_null:
        cols_with_no_nulls.append(col)

    if not has_any_flag:
        continue

    first = True
    for null_type, count in null_results.items():
        if count > 0:
            pct = count / total * 100
            col_display = col if first else ''
            dtype_display = str(dtype) if first else ''
            print(f"{col_display:<35} {dtype_display:<10} {null_type:<30} {count:>10,} {pct:>7.2f}%")
            summary_rows.append({
                'Column': col,
                'Dtype': str(dtype),
                'Null Type': null_type,
                'Flag Category': 'Review Only' if 'review only' in null_type else 'Null / Missing',
                'Count': count,
                'Percentage': round(pct, 2)
            })
            first = False

    print("-" * 97)

# ============================================================
# COLUMNS WITH ZERO NULLS
# ============================================================
print("\n" + "=" * 70)
print("✅ COLUMNS WITH ZERO NULLS (koi bhi null nahi)")
print("=" * 70)
if cols_with_no_nulls:
    for col in cols_with_no_nulls:
        print(f"  ✅ {col}")
else:
    print("  (Har column mein kuch na kuch null hai)")

# ============================================================
# SUMMARY TABLE — Sorted by Count
# ============================================================
print("\n" + "=" * 70)
print("📊 SUMMARY — NULL COUNT BY COLUMN (Sorted)")
print("=" * 70)

summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty:
    true_null_df = summary_df[summary_df['Flag Category'] == 'Null / Missing']
    col_summary = true_null_df.groupby('Column')['Count'].sum().reset_index()
    col_summary['Percentage'] = (col_summary['Count'] / len(df) * 100).round(2)
    col_summary = col_summary.sort_values('Count', ascending=False)
    col_summary.index = range(1, len(col_summary) + 1)
    if col_summary.empty:
        print("No true null/missing values found. Review-only flags may still exist.")
    else:
        print(col_summary.to_string())

# ============================================================
# OVERALL NULL STATS
# ============================================================
print("\n" + "=" * 70)
print("📈 OVERALL NULL STATISTICS")
print("=" * 70)
total_cells  = df.shape[0] * df.shape[1]
standard_nulls = df.isnull().sum().sum()
print(f"Total Rows           : {df.shape[0]:,}")
print(f"Total Columns        : {df.shape[1]}")
print(f"Total Cells          : {total_cells:,}")
print(f"Standard NaN/None    : {standard_nulls:,}  ({standard_nulls/total_cells*100:.2f}%)")
print(f"Columns WITH nulls   : {df.shape[1] - len(cols_with_no_nulls)}")
print(f"Columns WITHOUT nulls: {len(cols_with_no_nulls)}")

if not summary_df.empty:
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print(f"Summary saved        : {SUMMARY_PATH}")

print("\nNOTE:")
print("  Zero values direct missing/null nahi hain.")
print("  Unhein review-only signal maana gaya hai kyun ke healthcare data mein 0 valid bhi ho sakta hai.")

print("\n" + "=" * 70)
print("✅ NULL DETECTION COMPLETE!")
print("=" * 70)
