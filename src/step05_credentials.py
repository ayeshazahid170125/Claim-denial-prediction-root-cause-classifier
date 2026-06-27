"""
STEP 05 - Credentials Standardization (Day 4)
WellMind Data Solutions - Claim Denial Prediction System
Run: python step05_credentials.py

Why this step matters:
Provider credentials have thousands of spelling variants (M.D., MD, FNP-C,
PT, DPT, etc.). Models learn better when those variants are grouped into a
small number of clinically meaningful categories.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
REPORT_DIR = BASE_DIR / "reports" / "cleaning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FILE_PATH = BASE_DIR / "Medicare_Cleaned_Week1.csv"
SAVE_PATH = BASE_DIR / "Medicare_Cleaned_Credentials.csv"
GROUP_SUMMARY_PATH = REPORT_DIR / "credentials_group_summary.csv"
MAPPING_AUDIT_PATH = REPORT_DIR / "credentials_mapping_audit.csv"
OTHER_REVIEW_PATH = REPORT_DIR / "credentials_other_review.csv"

CREDENTIAL_COL = "Rndrng_Prvdr_Crdntls"
PAYMENT_COL = "Avg_Mdcr_Pymt_Amt"


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def clean_credential(value):
    """Normalize punctuation/casing while keeping the credential meaning."""
    if pd.isna(value):
        return np.nan

    cleaned = str(value).strip().upper()
    for token in [".", ",", "-", "/", "\\", "(", ")", " "]:
        cleaned = cleaned.replace(token, "")
    return cleaned or np.nan


def build_credential_map():
    """Return group definitions and a reverse lookup map."""
    groups = {
        "Physician": [
            "MD", "DO", "MBBS", "MB", "MBCHB", "MBB", "MDPHD", "DOMED",
            "MBBSMD", "MDMS", "MDMPH", "MDMBA",
        ],
        "CRNA": [
            "CRNA", "CRNABC", "APRNCRNA", "CAA", "CAAC",
        ],
        "PA": [
            "PA", "PAC", "RPAC", "RPA", "MPAS", "PACER", "MSPAC",
            "PHYSICIANASSISTANT",
        ],
        "NP": [
            "NP", "FNP", "FNPC", "CRNP", "APRN", "CNP", "WHNP", "AGACNP",
            "AGNP", "ACNP", "PMHNP", "CPNP", "ANP", "ANPC", "GNP", "NPC",
            "NPBC", "FNPBC", "AGACNPBC", "APNP", "CRNPBC", "ARNP",
            "ACNPBC", "CNPBC", "WHCNP", "APN", "DNP", "NURSEPRACTITIONER",
            "MSNFNPC", "MSNFNPBC", "CFNP", "AGPCNPBC", "DNPFNPC",
            "DNPFNPBC", "MSNAPRNFNPC", "MSNAPRNFNPBC", "DNPAPRNFNPC",
        ],
        "Therapist": [
            "PT", "DPT", "PTDPT", "OT", "OTR", "OTRL", "SLP", "ATP",
            "CHT", "MPT", "MSPT", "RPT", "LPT", "PTMPT",
            "PHYSICALTHERAPIST",
        ],
        "Other_Doctor": [
            "DPM", "OD", "PHD", "PSYD", "PHARMD", "AUD", "AUDC", "DDS", "DMD",
            "OPTOMETRIST",
        ],
        "DC_Chiro": ["DC"],
        "Social_Worker": ["LCSW", "LICSW", "LMSW", "CSW", "LISW"],
        "Nursing_Other": ["RN", "MSN", "CNS", "CNM"],
        "Dietitian": ["RD"],
        "Anesthesia_Assistant": ["AA", "AAC"],
    }

    reverse = {}
    duplicates = []
    for group, credentials in groups.items():
        for credential in credentials:
            key = credential.upper()
            if key in reverse and reverse[key] != group:
                duplicates.append((key, reverse[key], group))
            reverse[key] = group

    if duplicates:
        duplicate_text = ", ".join(f"{k}: {a}/{b}" for k, a, b in duplicates)
        raise ValueError(f"Credential appears in multiple groups: {duplicate_text}")

    return groups, reverse


def assign_group(cleaned_value, reverse_map):
    """Map cleaned credential into a modeling group."""
    if pd.isna(cleaned_value):
        return "Unknown"

    value = str(cleaned_value).strip().upper()

    if value in reverse_map:
        return reverse_map[value]

    # Common combined credentials often appear as MDFACC, DOMPH, ODMS, PTATC.
    # These suffixes are fellowships/degrees, so the clinical base credential is
    # the most useful modeling signal.
    prefix_rules = [
        ("MD", "Physician"),
        ("DO", "Physician"),
        ("MBBS", "Physician"),
        ("OD", "Other_Doctor"),
        ("DPM", "Other_Doctor"),
        ("PA", "PA"),
        ("NP", "NP"),
        ("PT", "Therapist"),
        ("OT", "Therapist"),
        ("RN", "Nursing_Other"),
    ]
    for prefix, group in prefix_rules:
        if value.startswith(prefix):
            return group

    # Conservative fallback for combined credentials such as MDMPH or APRNFNP.
    # Longer keys are checked first to avoid short keys stealing the match.
    for key in sorted(reverse_map, key=len, reverse=True):
        if len(key) >= 3 and (value.startswith(key) or key in value):
            return reverse_map[key]

    return "Other"


def require_columns(dataframe, columns):
    missing = [col for col in columns if col not in dataframe.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


print_section("DATA LOAD")
df = pd.read_csv(FILE_PATH, low_memory=False)
require_columns(df, [CREDENTIAL_COL, PAYMENT_COL])
rows_before = len(df)
cols_before = df.shape[1]

print(f"Loaded: {rows_before:,} rows x {cols_before} columns")
print(f"Input : {FILE_PATH}")

print_section("STEP 1 - SNAPSHOT BEFORE STANDARDIZATION")
unique_before = df[CREDENTIAL_COL].nunique(dropna=True)
missing_before = int(df[CREDENTIAL_COL].isna().sum())
print(f"Unique credential strings : {unique_before:,}")
print(f"Missing credential rows   : {missing_before:,}")
print("\nTop 10 raw credential values:")
print(df[CREDENTIAL_COL].value_counts(dropna=False).head(10).to_string())

print_section("STEP 2 - SPELLING STANDARDIZATION")
df["Crdntls_Clean"] = df[CREDENTIAL_COL].apply(clean_credential)
unique_clean = df["Crdntls_Clean"].nunique(dropna=True)

print(f"Unique before spelling fix : {unique_before:,}")
print(f"Unique after spelling fix  : {unique_clean:,}")
print("Lesson: punctuation variants are now one consistent spelling.")

examples = ["M.D.", "D.O.", "PT, DPT", "PA-C", "FNP-C"]
for raw in examples:
    cleaned = clean_credential(raw)
    print(f"  {raw:<10} -> {cleaned}")

print_section("STEP 3 - GROUP CREDENTIALS INTO MODELING CATEGORIES")
cred_groups, cred_map = build_credential_map()
df["Crdntls_Group"] = df["Crdntls_Clean"].apply(lambda value: assign_group(value, cred_map))

group_summary = (
    df.groupby("Crdntls_Group", dropna=False)
    .agg(
        Count=(CREDENTIAL_COL, "size"),
        Avg_Payment=(PAYMENT_COL, "mean"),
        Median_Payment=(PAYMENT_COL, "median"),
        Unique_Clean_Credentials=("Crdntls_Clean", "nunique"),
    )
    .reset_index()
)
group_summary["Pct"] = (group_summary["Count"] / len(df) * 100).round(4)
group_summary["Avg_Payment"] = group_summary["Avg_Payment"].round(2)
group_summary["Median_Payment"] = group_summary["Median_Payment"].round(2)
group_summary = group_summary.sort_values("Count", ascending=False)

print(group_summary.to_string(index=False))

print_section("STEP 4 - REVIEW REMAINING OTHER CREDENTIALS")
other_review = (
    df.loc[df["Crdntls_Group"] == "Other", "Crdntls_Clean"]
    .value_counts(dropna=False)
    .head(50)
    .rename_axis("Crdntls_Clean")
    .reset_index(name="Count")
)
other_rows = int((df["Crdntls_Group"] == "Other").sum())
other_pct = other_rows / len(df) * 100 if len(df) else 0

print(f"Other rows: {other_rows:,} ({other_pct:.4f}%)")
print("Top 50 Other credentials saved for manual review.")
if not other_review.empty:
    print(other_review.head(20).to_string(index=False))

print_section("STEP 5 - DROP INTERMEDIATE COLUMNS")
columns_to_drop = [CREDENTIAL_COL, "Crdntls_Clean"]
existing_drop_cols = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=existing_drop_cols)

print(f"Dropped columns: {existing_drop_cols}")
print("Kept feature   : Crdntls_Group")
print(f"Final shape    : {df.shape[0]:,} rows x {df.shape[1]} columns")

print_section("SAVE OUTPUTS")
mapping_audit = pd.DataFrame(
    [
        {"Clean_Credential": credential, "Group": group}
        for credential, group in sorted(cred_map.items())
    ]
)

df.to_csv(SAVE_PATH, index=False)
group_summary.to_csv(GROUP_SUMMARY_PATH, index=False)
mapping_audit.to_csv(MAPPING_AUDIT_PATH, index=False)
other_review.to_csv(OTHER_REVIEW_PATH, index=False)

print(f"Cleaned credential data saved : {SAVE_PATH}")
print(f"Group summary saved           : {GROUP_SUMMARY_PATH}")
print(f"Mapping audit saved           : {MAPPING_AUDIT_PATH}")
print(f"Other review saved            : {OTHER_REVIEW_PATH}")

print("\nSTEP 05 COMPLETE")
print("Next: Step 06 will create flags, log features, and encoded modeling data.")
