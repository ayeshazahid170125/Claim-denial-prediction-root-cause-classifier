"""
STEP 04 - Data Cleaning & Null Handling (Day 3)
WellMind Data Solutions - Claim Denial Prediction System
Run: python step04_cleaning.py

Purpose:
1. Convert documented string placeholders into real NaN values.
2. Treat zero payment/service values as review flags, not automatic nulls.
3. Apply conservative null handling for low-risk columns.
4. Drop columns that are too sparse, too specific, duplicated, or low-value for modeling.
5. Save a cleaned Week 1 dataset plus audit summaries for reporting.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "2024"
    / "PHY_R26_P05_V10_D24_Prov_Svc.csv"
)

OUTPUT_DATA_PATH = BASE_DIR / "Medicare_Cleaned_Week1.csv"
NULL_COMPARISON_PATH = BASE_DIR / "cleaning_null_before_after.csv"
STRING_FIXES_PATH = BASE_DIR / "cleaning_string_null_fixes.csv"
ZERO_REVIEW_PATH = BASE_DIR / "cleaning_zero_review_flags.csv"
COLUMN_ACTIONS_PATH = BASE_DIR / "cleaning_column_actions.csv"
SUMMARY_PATH = BASE_DIR / "cleaning_summary.csv"


def load_dataset(path):
    """Load CSV or Excel input with a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Only .csv, .xlsx, and .xls files are supported.")


def get_null_snapshot(dataframe):
    """Return column-level null count and percentage."""
    rows = []
    total_rows = len(dataframe)

    for col in dataframe.columns:
        null_count = int(dataframe[col].isna().sum())
        null_pct = (null_count / total_rows * 100) if total_rows else 0
        rows.append(
            {
                "Column": col,
                "Dtype": str(dataframe[col].dtype),
                "Null_Count": null_count,
                "Null_Pct": round(null_pct, 4),
            }
        )

    return pd.DataFrame(rows)


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def print_nonzero_nulls(snapshot, empty_message):
    result = snapshot[snapshot["Null_Count"] > 0]
    if result.empty:
        print(empty_message)
    else:
        print(result.to_string(index=False))


# ============================================================
# LOAD DATA
# ============================================================
df = load_dataset(DATA_PATH)
original_rows = len(df)
original_cols = df.shape[1]

print(f"\nData loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"File: {DATA_PATH}")

print_section("SNAPSHOT BEFORE CLEANING - NULL COUNTS")
before_snap = get_null_snapshot(df)
print_nonzero_nulls(before_snap, "No standard null values found before cleaning.")


# ============================================================
# STEP 1 - STRING PLACEHOLDERS TO NaN
# ============================================================
print_section("STEP 1 - STRING PLACEHOLDERS TO PROPER NaN")

# These replacements are column-specific. Numeric zero is intentionally not
# included because Medicare data can contain valid zero-like values in context.
string_fixes = {
    "Rndrng_Prvdr_First_Name": ["Unknown", "UNKNOWN", "unknown"],
    "Rndrng_Prvdr_MI": ["-", "0", "(", "X", "x"],
    "Rndrng_Prvdr_St1": ["Unknown", "UNKNOWN", "unknown"],
    "Rndrng_Prvdr_St2": ["0"],
    "Rndrng_Prvdr_Zip5": ["0"],
    "Rndrng_Prvdr_RUCA_Desc": ["Unknown", "UNKNOWN", "unknown"],
}

string_fix_rows = []

for col, bad_values in string_fixes.items():
    if col not in df.columns:
        print(f"  SKIP - {col}: column not found")
        string_fix_rows.append(
            {
                "Column": col,
                "Placeholder": None,
                "Rows_Fixed": 0,
                "Action": "Skipped - column not found",
            }
        )
        continue

    total_fixed = 0
    for bad_value in bad_values:
        mask = df[col].notna() & df[col].astype(str).str.strip().eq(str(bad_value).strip())
        fixed_count = int(mask.sum())

        if fixed_count > 0:
            df.loc[mask, col] = np.nan
            total_fixed += fixed_count

        string_fix_rows.append(
            {
                "Column": col,
                "Placeholder": bad_value,
                "Rows_Fixed": fixed_count,
                "Action": "Converted to NaN" if fixed_count else "No match",
            }
        )

    print(f"  {col}: {total_fixed:,} placeholder values converted to NaN")

string_fix_df = pd.DataFrame(string_fix_rows)


# ============================================================
# STEP 2 - ZERO VALUES REVIEW FLAGS
# ============================================================
print_section("STEP 2 - ZERO VALUES REVIEW FLAGS")

zero_review_cols = [
    "Avg_Mdcr_Alowd_Amt",
    "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]

zero_review_rows = []
for col in zero_review_cols:
    if col not in df.columns:
        print(f"  SKIP - {col}: column not found")
        zero_review_rows.append(
            {
                "Column": col,
                "Zero_Count": 0,
                "Zero_Pct": 0,
                "Action": "Skipped - column not found",
                "Reason": "Column unavailable in dataset",
            }
        )
        continue

    zero_count = int((df[col] == 0).sum())
    zero_pct = zero_count / len(df) * 100 if len(df) else 0
    zero_review_rows.append(
        {
            "Column": col,
            "Zero_Count": zero_count,
            "Zero_Pct": round(zero_pct, 4),
            "Action": "Kept in dataset and documented as a review flag",
            "Reason": "Zero values are extremely rare and may be valid CMS payment outcomes; removing them could create bias",
        }
    )
    print(f"  {col}: {zero_count:,} zero values flagged for review ({zero_pct:.4f}%)")

print("\nDecision: zero values were kept and documented because removal could bias payment analysis.")

zero_review_df = pd.DataFrame(zero_review_rows)


# ============================================================
# STEP 3 - FILL LOW-RISK NULLS
# ============================================================
print_section("STEP 3 - LOW-RISK NULL HANDLING")

column_action_rows = []

ruca_col = "Rndrng_Prvdr_RUCA"
if ruca_col in df.columns:
    null_before = int(df[ruca_col].isna().sum())
    mode_values = df[ruca_col].dropna().mode()

    if null_before > 0 and not mode_values.empty:
        fill_value = mode_values.iloc[0]
        df[ruca_col] = df[ruca_col].fillna(fill_value)
        null_after = int(df[ruca_col].isna().sum())
        action = f"Filled nulls with mode value {fill_value}"
    elif null_before > 0:
        fill_value = None
        null_after = null_before
        action = "No fill applied - no non-null mode available"
    else:
        fill_value = None
        null_after = 0
        action = "No nulls found"

    column_action_rows.append(
        {
            "Column": ruca_col,
            "Action_Type": "Imputation",
            "Rows_Affected": null_before - null_after,
            "Null_Before": null_before,
            "Null_After": null_after,
            "Decision": action,
            "Reason": "RUCA is a structured geographic code; mode fill is conservative for limited missingness",
        }
    )
    print(f"  {ruca_col}: {action}")
else:
    column_action_rows.append(
        {
            "Column": ruca_col,
            "Action_Type": "Imputation",
            "Rows_Affected": 0,
            "Null_Before": None,
            "Null_After": None,
            "Decision": "Skipped - column not found",
            "Reason": "Column unavailable in dataset",
        }
    )
    print(f"  SKIP - {ruca_col}: column not found")


# ============================================================
# STEP 4 - DROP LOW-VALUE COLUMNS
# ============================================================
print_section("STEP 4 - DROP LOW-VALUE OR REDUNDANT COLUMNS")

cols_to_drop = {
    "Rndrng_Prvdr_MI": "High missingness and limited predictive value",
    "Rndrng_Prvdr_St1": "Street address is too specific and noisy for modeling",
    "Rndrng_Prvdr_St2": "Optional address field with high missingness",
    "Rndrng_Prvdr_State_FIPS": "Redundant with state abbreviation",
    "Rndrng_Prvdr_Zip5": "Very high cardinality; better handled later if needed",
    "Rndrng_Prvdr_RUCA_Desc": "Text duplicate of numeric RUCA code",
}

for col, reason in cols_to_drop.items():
    if col in df.columns:
        null_before = int(df[col].isna().sum())
        df = df.drop(columns=[col])
        column_action_rows.append(
            {
                "Column": col,
                "Action_Type": "Dropped",
                "Rows_Affected": len(df),
                "Null_Before": null_before,
                "Null_After": 0,
                "Decision": "Dropped from cleaned dataset",
                "Reason": reason,
            }
        )
        print(f"  DROPPED - {col}: {reason}")
    else:
        column_action_rows.append(
            {
                "Column": col,
                "Action_Type": "Dropped",
                "Rows_Affected": 0,
                "Null_Before": None,
                "Null_After": None,
                "Decision": "Skipped - column not found",
                "Reason": reason,
            }
        )
        print(f"  SKIP - {col}: column not found")

column_actions_df = pd.DataFrame(column_action_rows)


# ============================================================
# STEP 5 - AFTER CLEANING SNAPSHOT
# ============================================================
print_section("SNAPSHOT AFTER CLEANING - NULL COUNTS")

after_snap = get_null_snapshot(df)
print_nonzero_nulls(after_snap, "No standard null values remain after cleaning.")


# ============================================================
# STEP 6 - BEFORE VS AFTER COMPARISON
# ============================================================
print_section("BEFORE VS AFTER - CLEANING COMPARISON")

null_comparison = before_snap[["Column", "Null_Count", "Null_Pct"]].rename(
    columns={"Null_Count": "Null_Count_Before", "Null_Pct": "Null_Pct_Before"}
)
after_for_merge = after_snap[["Column", "Null_Count", "Null_Pct"]].rename(
    columns={"Null_Count": "Null_Count_After", "Null_Pct": "Null_Pct_After"}
)

null_comparison = null_comparison.merge(after_for_merge, on="Column", how="left")
null_comparison["Null_Count_After"] = null_comparison["Null_Count_After"].fillna(0).astype(int)
null_comparison["Null_Pct_After"] = null_comparison["Null_Pct_After"].fillna(0)
null_comparison["Status"] = np.where(
    ~null_comparison["Column"].isin(df.columns),
    "Dropped",
    np.where(
        null_comparison["Null_Count_After"] == 0,
        "Clean",
        np.where(
            null_comparison["Null_Count_After"] < null_comparison["Null_Count_Before"],
            "Reduced",
            np.where(
                null_comparison["Null_Count_After"] > null_comparison["Null_Count_Before"],
                "Increased - placeholders converted",
                "Unchanged",
            ),
        ),
    ),
)

print(f"{'Metric':<35} {'Before':>15} {'After':>15}")
print("-" * 67)
print(f"{'Total Rows':<35} {original_rows:>15,} {len(df):>15,}")
print(f"{'Total Columns':<35} {original_cols:>15,} {df.shape[1]:>15,}")
print(f"{'Total Null Cells':<35} {before_snap['Null_Count'].sum():>15,} {after_snap['Null_Count'].sum():>15,}")
print(
    f"{'Columns with Nulls':<35} "
    f"{(before_snap['Null_Count'] > 0).sum():>15} "
    f"{(after_snap['Null_Count'] > 0).sum():>15}"
)
print(f"{'Rows Dropped During Cleaning':<35} {0:>15,} {original_rows - len(df):>15,}")

changed_nulls = null_comparison[
    (null_comparison["Null_Count_Before"] > 0)
    | (null_comparison["Status"].isin(["Dropped", "Reduced"]))
]
print("\nColumn-level null comparison:")
print(changed_nulls.to_string(index=False))


# ============================================================
# STEP 7 - FINAL DATAFRAME INFO
# ============================================================
print_section("FINAL DATAFRAME - CLEANED WEEK 1 OUTPUT")

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nRemaining columns:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    null_count = int(df[col].isna().sum())
    status = f"{null_count:,} nulls" if null_count else "clean"
    print(f"  {i:>2}. {col:<35} {str(dtype):<12} {status}")


# ============================================================
# SAVE OUTPUTS
# ============================================================
print_section("SAVE CLEANED DATA AND AUDIT FILES")
print("Saving large CSV files now. With 9.6M rows this can take a few minutes...")

summary_df = pd.DataFrame(
    [
        {"Metric": "Original rows", "Value": original_rows},
        {"Metric": "Final rows", "Value": len(df)},
        {"Metric": "Rows dropped", "Value": original_rows - len(df)},
        {"Metric": "Original columns", "Value": original_cols},
        {"Metric": "Final columns", "Value": df.shape[1]},
        {"Metric": "Columns dropped", "Value": original_cols - df.shape[1]},
        {"Metric": "Null cells before", "Value": int(before_snap["Null_Count"].sum())},
        {"Metric": "Null cells after", "Value": int(after_snap["Null_Count"].sum())},
        {
            "Metric": "Zero values removed automatically",
            "Value": "No - review only",
        },
    ]
)

df.to_csv(OUTPUT_DATA_PATH, index=False)
null_comparison.to_csv(NULL_COMPARISON_PATH, index=False)
string_fix_df.to_csv(STRING_FIXES_PATH, index=False)
zero_review_df.to_csv(ZERO_REVIEW_PATH, index=False)
column_actions_df.to_csv(COLUMN_ACTIONS_PATH, index=False)
summary_df.to_csv(SUMMARY_PATH, index=False)

print(f"Cleaned data saved              : {OUTPUT_DATA_PATH}")
print(f"Null before/after report saved  : {NULL_COMPARISON_PATH}")
print(f"String placeholder report saved : {STRING_FIXES_PATH}")
print(f"Zero review report saved        : {ZERO_REVIEW_PATH}")
print(f"Column actions report saved     : {COLUMN_ACTIONS_PATH}")
print(f"Cleaning summary saved          : {SUMMARY_PATH}")

print("\nNOTE:")
print("  This cleaning step is conservative.")
print("  Outliers and zero values are documented for review instead of being blindly removed.")
print("  This keeps the Medicare data defensible for later modeling and reporting.")

print("\n" + "=" * 75)
print("STEP 04 COMPLETE - CLEANED WEEK 1 DATA READY")
print("=" * 75)
