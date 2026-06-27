"""
STEP 06 - Feature Transformation & Encoding (Day 5)
WellMind Data Solutions - Claim Denial Prediction System
Run: python step06_encoding.py

This file creates two outputs:
1. Medicare_Cleaned_Outliers.csv
   Readable dataset with review flags and log-transformed numeric fields.
2. Medicare_Cleaned_Encoded.csv
   Numeric baseline dataset for quick checks.

Important learning point:
High-cardinality columns are NOT encoded here. Even non-target frequency
encoding should be fit on train folds only, then applied to validation/test
folds. Step 08 will handle that inside the modeling workflow to avoid
train/test contamination.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
REPORT_DIR = BASE_DIR / "reports" / "modeling"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PATH = BASE_DIR / "Medicare_Cleaned_Week1.csv"
OUTLIER_OUTPUT_PATH = BASE_DIR / "Medicare_Cleaned_Outliers.csv"
ENCODED_OUTPUT_PATH = BASE_DIR / "Medicare_Cleaned_Encoded.csv"
RULE_AUDIT_PATH = REPORT_DIR / "feature_rule_review_summary.csv"
ENCODING_SUMMARY_PATH = REPORT_DIR / "encoding_summary.csv"
FREQUENCY_MAP_PATH = REPORT_DIR / "frequency_encoding_maps.csv"
DEFERRED_ENCODING_PATH = REPORT_DIR / "encoding_deferred_high_cardinality.csv"


NUMERIC_COLS = [
    "Tot_Benes",
    "Tot_Srvcs",
    "Tot_Bene_Day_Srvcs",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Alowd_Amt",
    "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]

VALID_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "GU", "VI", "MP", "AS",
}

BINARY_MAPS = {
    "Rndrng_Prvdr_Ent_Cd": {"I": 1, "O": 0},
    "Rndrng_Prvdr_Mdcr_Prtcptg_Ind": {"Y": 1, "N": 0},
    "HCPCS_Drug_Ind": {"Y": 1, "N": 0},
    "Place_Of_Srvc": {"F": 1, "O": 0},
}

DROP_FOR_MODELING = [
    "Rndrng_NPI",
    "Rndrng_Prvdr_Last_Org_Name",
    "Rndrng_Prvdr_First_Name",
    "Rndrng_Prvdr_City",
    "Rndrng_Prvdr_Cntry",
    "HCPCS_Desc",
]

HIGH_CARDINALITY_COLS = [
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
    "HCPCS_Cd",
]


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def require_columns(dataframe, columns):
    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def add_rule_flag(dataframe, flag_col, condition, explanation, audit_rows):
    dataframe[flag_col] = condition.astype(int)
    count = int(dataframe[flag_col].sum())
    pct = count / len(dataframe) * 100 if len(dataframe) else 0
    audit_rows.append(
        {
            "Flag": flag_col,
            "Count": count,
            "Pct": round(pct, 4),
            "Action": "Flag retained; rows not removed automatically",
            "Explanation": explanation,
        }
    )
    print(f"  {flag_col:<35} {count:>12,} ({pct:>7.4f}%)")


print_section("DATA LOAD")
df = pd.read_csv(INPUT_PATH, low_memory=False)
require_columns(df, NUMERIC_COLS)

rows_before = len(df)
cols_before = df.shape[1]
print(f"Loaded: {rows_before:,} rows x {cols_before} columns")
print(f"Input : {INPUT_PATH}")

print_section("STEP 1 - BUSINESS RULE REVIEW FLAGS")
rule_audit_rows = []

add_rule_flag(
    df,
    "flag_charge_lt_payment",
    df["Avg_Sbmtd_Chrg"] < df["Avg_Mdcr_Pymt_Amt"],
    "Submitted charge lower than Medicare payment. Review before removal.",
    rule_audit_rows,
)
add_rule_flag(
    df,
    "flag_services_lt_benes",
    df["Tot_Srvcs"] < df["Tot_Benes"],
    "Service count lower than beneficiary count. Review before removal.",
    rule_audit_rows,
)
add_rule_flag(
    df,
    "flag_zero_payment",
    df["Avg_Mdcr_Pymt_Amt"] <= 0,
    "Zero/non-positive payment. Kept for review because zero may be meaningful.",
    rule_audit_rows,
)
add_rule_flag(
    df,
    "flag_zero_allowed",
    df["Avg_Mdcr_Alowd_Amt"] <= 0,
    "Zero/non-positive allowed amount. Kept for review.",
    rule_audit_rows,
)

if "Rndrng_Prvdr_State_Abrvtn" in df.columns:
    add_rule_flag(
        df,
        "flag_invalid_state",
        ~df["Rndrng_Prvdr_State_Abrvtn"].isin(VALID_STATES),
        "State code outside US state/territory list.",
        rule_audit_rows,
    )

if "Rndrng_Prvdr_Cntry" in df.columns:
    add_rule_flag(
        df,
        "flag_non_us_country",
        df["Rndrng_Prvdr_Cntry"].fillna("Unknown") != "US",
        "Non-US country kept as review signal, not automatic error.",
        rule_audit_rows,
    )

print("\nDecision: rows are flagged, not blindly dropped.")

print_section("STEP 2 - RUCA UNKNOWN HANDLING")
if "Rndrng_Prvdr_RUCA" in df.columns:
    df["RUCA_Unknown_Flag"] = (df["Rndrng_Prvdr_RUCA"] == 99).astype(int)
    ruca_unknown_count = int(df["RUCA_Unknown_Flag"].sum())
    valid_ruca = df.loc[df["Rndrng_Prvdr_RUCA"].notna() & (df["Rndrng_Prvdr_RUCA"] != 99), "Rndrng_Prvdr_RUCA"]
    ruca_fill = valid_ruca.median()

    if pd.notna(ruca_fill):
        df.loc[df["Rndrng_Prvdr_RUCA"] == 99, "Rndrng_Prvdr_RUCA"] = ruca_fill

    print(f"RUCA=99 flagged       : {ruca_unknown_count:,}")
    print(f"RUCA=99 replaced with : {ruca_fill}")
else:
    print("Rndrng_Prvdr_RUCA not found; skip.")

print_section("STEP 3 - LOG TRANSFORM SKEWED NUMERIC COLUMNS")
log_summary_rows = []
for col in NUMERIC_COLS:
    new_col = f"{col}_log"
    clipped = df[col].clip(lower=0)
    skew_before = float(df[col].skew())
    df[new_col] = np.log1p(clipped)
    skew_after = float(df[new_col].skew())
    log_summary_rows.append(
        {
            "Column": col,
            "Log_Column": new_col,
            "Skew_Before": round(skew_before, 4),
            "Skew_After": round(skew_after, 4),
            "Note": "Original column kept in readable output; log version used for modeling",
        }
    )
    print(f"  {col:<25} skew {skew_before:>9.2f} -> {skew_after:>9.2f}")

print_section("STEP 4 - DROP CORRELATED LOG FEATURES")
drop_corr = [
    "Avg_Mdcr_Stdzd_Amt_log",
    "Tot_Bene_Day_Srvcs_log",
]
drop_corr = [col for col in drop_corr if col in df.columns]
if drop_corr:
    print(f"Marked for modeling drop: {drop_corr}")
    print("Reason: highly correlated with related log features from Step 03 review.")
else:
    print("No correlated log columns found to drop.")

print_section("SAVE READABLE TRANSFORMED DATA")
df.to_csv(OUTLIER_OUTPUT_PATH, index=False)
print(f"Saved: {OUTLIER_OUTPUT_PATH}")
print(f"Rows : {len(df):,}")
print(f"Cols : {df.shape[1]}")

print_section("STEP 5 - BUILD NUMERIC MODELING DATASET")
model_df = df.copy()

drop_existing = [col for col in DROP_FOR_MODELING if col in model_df.columns]
model_df = model_df.drop(columns=drop_existing)
print(f"Dropped non-model columns: {drop_existing}")

original_numeric_drop = [col for col in NUMERIC_COLS if col in model_df.columns]
model_df = model_df.drop(columns=original_numeric_drop)
print(f"Dropped raw numeric columns after log transform: {original_numeric_drop}")

drop_corr_existing = [col for col in drop_corr if col in model_df.columns]
model_df = model_df.drop(columns=drop_corr_existing)
print(f"Dropped correlated log columns: {drop_corr_existing}")

print_section("STEP 6 - BINARY ENCODING")
encoding_summary_rows = []
for col, mapping in BINARY_MAPS.items():
    if col not in model_df.columns:
        encoding_summary_rows.append(
            {
                "Column": col,
                "Encoding": "Binary",
                "Action": "Skipped - column not found",
                "Detail": "",
            }
        )
        continue

    before_values = model_df[col].value_counts(dropna=False).to_dict()
    encoded_col = f"{col}_bin"
    model_df[encoded_col] = model_df[col].map(mapping)
    unknown_count = int(model_df[encoded_col].isna().sum())
    model_df[encoded_col] = model_df[encoded_col].fillna(-1).astype(int)
    model_df = model_df.drop(columns=[col])

    encoding_summary_rows.append(
        {
            "Column": col,
            "Encoding": "Binary",
            "Action": f"Mapped to {encoded_col}",
            "Detail": f"Unknown/unmapped rows filled with -1: {unknown_count}; before={before_values}",
        }
    )
    print(f"  {col:<35} -> {encoded_col}; unknown={unknown_count:,}")

print_section("STEP 7 - ONE-HOT ENCODING LOW-CARDINALITY COLUMNS")
if "Crdntls_Group" in model_df.columns:
    before_cols = model_df.shape[1]
    dummies = pd.get_dummies(model_df["Crdntls_Group"], prefix="Crdntls", drop_first=True)
    model_df = pd.concat([model_df.drop(columns=["Crdntls_Group"]), dummies], axis=1)
    added = model_df.shape[1] - before_cols
    encoding_summary_rows.append(
        {
            "Column": "Crdntls_Group",
            "Encoding": "One-Hot",
            "Action": "pd.get_dummies(drop_first=True)",
            "Detail": f"Added columns: {added}",
        }
    )
    print(f"Crdntls_Group one-hot encoded; added {added} columns")
else:
    print("Crdntls_Group not found; skip.")

print_section("STEP 8 - DEFER HIGH-CARDINALITY ENCODING")
deferred_rows = []
for col in HIGH_CARDINALITY_COLS:
    if col not in model_df.columns:
        encoding_summary_rows.append(
            {
                "Column": col,
                "Encoding": "Frequency",
                "Action": "Skipped - column not found",
                "Detail": "",
            }
        )
        continue

    unique_count = int(model_df[col].nunique(dropna=False))
    top_values = model_df[col].value_counts(dropna=False).head(10)
    for value, count in top_values.items():
        deferred_rows.append(
            {
                "Column": col,
                "Example_Category": value,
                "Example_Count": int(count),
                "Unique_Categories": unique_count,
                "Decision": "Deferred to Step 08 train-fold encoder",
                "Leakage_Control": "Fit category frequencies on training fold only; apply learned map to validation/test",
            }
        )

    model_df = model_df.drop(columns=[col])
    encoding_summary_rows.append(
        {
            "Column": col,
            "Encoding": "Deferred",
            "Action": "Dropped from Step 06 baseline encoded file",
            "Detail": f"Unique categories: {unique_count}; encode inside Step 08 cross-validation/training only",
        }
    )
    print(f"  {col:<35} deferred; categories={unique_count:,}")

print_section("STEP 9 - FINAL NUMERIC CHECK")
bool_cols = model_df.select_dtypes(include=["bool"]).columns
if len(bool_cols) > 0:
    model_df[bool_cols] = model_df[bool_cols].astype(int)

object_cols = model_df.select_dtypes(include=["object"]).columns.tolist()
if object_cols:
    print(f"Remaining object columns will be dropped: {object_cols}")
    model_df = model_df.drop(columns=object_cols)
else:
    print("All remaining columns are numeric-compatible.")

null_counts = model_df.isna().sum()
null_counts = null_counts[null_counts > 0]
if not null_counts.empty:
    print("Nulls found; filling numeric nulls with median:")
    print(null_counts.to_string())
    model_df = model_df.fillna(model_df.median(numeric_only=True))
else:
    print("No nulls found.")

print(f"Final encoded shape: {model_df.shape[0]:,} rows x {model_df.shape[1]} columns")

print_section("SAVE ENCODED DATA AND AUDIT REPORTS")
rule_audit_df = pd.DataFrame(rule_audit_rows)
encoding_summary_df = pd.DataFrame(encoding_summary_rows + log_summary_rows)
deferred_df = pd.DataFrame(deferred_rows)

model_df.to_csv(ENCODED_OUTPUT_PATH, index=False)
rule_audit_df.to_csv(RULE_AUDIT_PATH, index=False)
encoding_summary_df.to_csv(ENCODING_SUMMARY_PATH, index=False)
deferred_df.to_csv(DEFERRED_ENCODING_PATH, index=False)

# Keep the old artifact name as a compatibility note for notebooks/reports that
# may already reference it.
pd.DataFrame(
    [
        {
            "Status": "Deferred",
            "Reason": "Full-dataset frequency encoding was removed to avoid train/test contamination.",
            "Next_Step": "Fit high-cardinality encoders inside Step 08 train folds only.",
        }
    ]
).to_csv(FREQUENCY_MAP_PATH, index=False)

print(f"Encoded data saved       : {ENCODED_OUTPUT_PATH}")
print(f"Rule audit saved         : {RULE_AUDIT_PATH}")
print(f"Encoding summary saved   : {ENCODING_SUMMARY_PATH}")
print(f"Deferred encoding saved  : {DEFERRED_ENCODING_PATH}")
print(f"Frequency map note saved : {FREQUENCY_MAP_PATH}")

print("\nSTEP 06 COMPLETE")
print("Next: Step 07 creates the denial-risk proxy target.")
