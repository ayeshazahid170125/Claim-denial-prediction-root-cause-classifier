"""
STEP 10 - Synthetic RARC-Style Denial Text Dataset
WellMind Data Solutions - Claim Denial Prediction System

Run:
    py step10_synthetic_rarc_data.py

Purpose:
Generate a balanced, reproducible synthetic text dataset for root-cause NLP
classification using the Step 09 taxonomy.

Important:
This data is synthetic RARC-style training text. It is not real remittance
data and should be replaced or augmented with real CARC/RARC denial text in a
client deployment.
"""

from pathlib import Path
import random
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMY_PATH = OUTPUT_DIR / "rarc_root_cause_taxonomy.csv"
REAL_RARC_PATH = OUTPUT_DIR / "real_rarc_codes.csv"
DATASET_PATH = OUTPUT_DIR / "rarc_denial_dataset.csv"
TRAIN_PATH = OUTPUT_DIR / "rarc_train.csv"
VAL_PATH = OUTPUT_DIR / "rarc_val.csv"
TEST_PATH = OUTPUT_DIR / "rarc_test.csv"
QUALITY_REPORT_PATH = OUTPUT_DIR / "rarc_dataset_quality_report.csv"
CHART_PATH = OUTPUT_DIR / "rarc_data_stats.png"

RANDOM_STATE = 42
# Rows per category based on available real RARC codes
# More codes = more unique rows possible
ROWS_PER_CATEGORY = {
    "eligibility":                 150,   # 7 real codes
    "authorization":               150,   # 6 real codes
    "duplicate_claim":             150,   # 7 real codes
    "medical_necessity":           150,   # 7 real codes
    "timely_filing":               100,   # 2 real codes
    "coding_error":                200,   # 12 real codes
    "coordination_of_benefits":    200,   # 10 real codes
    "not_covered":                 250,   # 17 real codes
    "other":                       250,   # 17 real codes
    "documentation":               400,   # 42 real codes
}
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
DISTRACTOR_CONTEXT_RATE = 0.35
TYPO_NOISE_RATE = 0.08

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

RARC_TO_CATEGORY = {
    "eligibility": ["N30", "N103", "N216", "N425", "N619", "N819", "N912"],
    "coding_error": ["M20", "M51", "M76", "M81", "M84", "N56", "N122", "N519", "N784", "N822", "N823", "N846"],
    "authorization": ["M62", "N54", "N175", "N241", "N758", "N915"],
    "duplicate_claim": ["M80", "M86", "N111", "N522", "N537", "N832", "N913"],
    "not_covered": ["M2", "M28", "M37", "M82", "M83", "M89", "M90", "N30", "N383", "N425", "N428", "N429", "N567", "N569", "N789", "N808", "N906"],
    "timely_filing": ["N892", "N921"],
    "medical_necessity": ["M25", "M26", "M42", "N115", "N386", "N661", "N825"],
    "coordination_of_benefits": ["MA04", "N8", "N23", "N32", "N155", "N177", "N360", "N420", "N854", "N891"],
    "documentation": [
        "M19", "M23", "M29", "M30", "M31", "M60", "M69", "M127", "N26", "N227", "N232", "N233",
        "N234", "N236", "N237", "N240", "N393", "N394", "N395", "N396", "N705", "N706", "N707",
        "N708", "N709", "N710", "N711", "N712", "N713", "N714", "N785", "N791", "N792", "N850",
        "N893", "N896", "N897", "N899", "N900", "N901", "N902", "N903",
    ],
    "other": ["N104", "N130", "N210", "N211", "N381", "N387", "N433", "N438", "N790", "N802", "N803", "N831", "N836", "N852", "N880", "N887", "N920"],
}


BASE_TEMPLATES = {
    "eligibility": [
        "Patient coverage was inactive on the date of service.",
        "Member eligibility could not be verified for this claim.",
        "Subscriber ID does not match active payer records.",
        "Patient was not enrolled in the plan on the service date.",
        "Coverage terminated before the submitted service was performed.",
        "No active benefit record was found for this member.",
        "The service date falls outside the patient's coverage period.",
        "Plan records show the member was not eligible for benefits.",
        "Coverage was suspended when the service was rendered.",
        "Policy number was not valid for the submitted date of service.",
    ],
    "coding_error": [
        "Procedure code is invalid for the submitted claim format.",
        "Modifier is missing or inconsistent with the billed service.",
        "Diagnosis code does not support the procedure code submitted.",
        "The HCPCS code requires a more specific billing modifier.",
        "Claim contains a code combination that requires correction.",
        "Submitted CPT code is not valid for the provider specialty.",
        "Procedure and diagnosis coding are inconsistent.",
        "The code billed is not payable with the submitted modifier.",
        "Claim line requires corrected coding before processing.",
        "Coding edit indicates the billed service conflicts with payer rules.",
    ],
    "authorization": [
        "Prior authorization was required but not found.",
        "Authorization number is invalid for this service.",
        "Referral was required before the service could be covered.",
        "Pre-certification was not obtained before treatment.",
        "Authorization expired before the date of service.",
        "The approved authorization does not match the billed procedure.",
        "Payer records do not show authorization for this claim.",
        "Referral information is missing or incomplete.",
        "The service exceeded the authorized number of visits.",
        "Authorization was denied before claim submission.",
    ],
    "duplicate_claim": [
        "This claim appears to duplicate a previously submitted claim.",
        "Service was already billed for the same patient and date.",
        "Duplicate claim line detected by payer processing rules.",
        "A matching claim has already been paid.",
        "The submitted service duplicates another claim in payer records.",
        "Claim was previously processed with the same service details.",
        "Duplicate submission identified for provider, patient, and date.",
        "The claim line matches a prior paid or pending line.",
        "Repeat billing detected for the same procedure.",
        "Payer records show this service was already adjudicated.",
    ],
    "not_covered": [
        "Service is not covered under the patient's benefit plan.",
        "Procedure is excluded by payer coverage policy.",
        "The billed service is considered non-covered for this plan.",
        "Benefit limitations do not allow payment for this service.",
        "Service is bundled into another payable procedure.",
        "The plan does not cover this item in the submitted setting.",
        "Coverage policy excludes the billed procedure.",
        "The service is not a payable benefit for this member.",
        "Payer policy identifies the service as investigational or excluded.",
        "Claim denied because the benefit category does not cover this service.",
    ],
    "timely_filing": [
        "Claim was received after the timely filing deadline.",
        "Submission exceeded the payer's filing limit.",
        "Corrected claim was submitted outside the allowed window.",
        "Appeal or reconsideration request was not filed timely.",
        "Proof of timely filing is required for review.",
        "Claim filing period expired before payer receipt.",
        "Payer received the claim after the contractual deadline.",
        "Late submission prevents payment under plan rules.",
        "The claim was not submitted within the required timeframe.",
        "Timely filing limit was exceeded for this service.",
    ],
    "medical_necessity": [
        "Documentation does not support medical necessity for the service.",
        "Diagnosis submitted does not meet payer medical necessity criteria.",
        "Service is not considered medically necessary for the reported condition.",
        "Clinical criteria were not met for the billed procedure.",
        "Payer policy does not support necessity based on submitted diagnosis.",
        "The level of service is not supported by clinical information.",
        "Medical necessity review determined the service was not justified.",
        "Submitted information does not meet coverage determination criteria.",
        "Procedure lacks clinical indication under payer guidelines.",
        "Diagnosis and documentation do not justify the service.",
    ],
    "coordination_of_benefits": [
        "Another payer is primary and must process the claim first.",
        "Coordination of benefits information is missing or outdated.",
        "Medicare secondary payer information is required.",
        "Primary insurance payment details are needed before processing.",
        "Payer order could not be determined from submitted information.",
        "Claim requires updated other-insurance information.",
        "Benefits must be coordinated with the patient's primary carrier.",
        "Secondary payer cannot process without primary payer adjudication.",
        "COB records indicate another plan has payment responsibility.",
        "Other coverage information conflicts with payer records.",
    ],
    "documentation": [
        "Medical records are required to support this claim.",
        "Submitted documentation is incomplete for the billed service.",
        "Operative note or clinical report is missing.",
        "Additional documentation is needed before claim review can continue.",
        "Required attachment was not included with the claim.",
        "Documentation does not support the billed procedure details.",
        "Payer requested records were not received.",
        "Clinical notes are insufficient to validate the service.",
        "Order, report, or supporting record is missing.",
        "Claim lacks documentation required by payer policy.",
    ],
    "other": [
        "Claim requires manual review due to payer processing rules.",
        "Administrative issue prevented claim adjudication.",
        "Provider enrollment or billing status requires review.",
        "Claim format issue requires correction before payment.",
        "Payer system returned a general processing denial.",
        "Billing information could not be validated by payer.",
        "Claim contains inconsistent administrative information.",
        "Payer requested additional review before final adjudication.",
        "Submission did not meet payer processing requirements.",
        "Unclassified denial reason requires analyst review.",
    ],
}

PATIENT_TERMS = ["patient", "member", "beneficiary", "subscriber", "claimant"]
PAYER_TERMS = ["payer", "health plan", "carrier", "insurer", "Medicare contractor"]
SERVICE_TERMS = ["service", "procedure", "claim line", "encounter", "treatment"]
PREFIXES = [
    "",
    "Denial reason: ",
    "Remark: ",
    "Claim review note: ",
    "Payer response: ",
    "Adjudication message: ",
]
SUFFIXES = [
    "",
    " Please review before resubmission.",
    " Correct and resubmit if appropriate.",
    " Additional follow-up is required.",
    " Route to the responsible RCM workqueue.",
    " Validate supporting information before appeal.",
]
CLAIM_CONTEXTS = [
    "DOS {dos}",
    "claim line {line}",
    "claim ID {claim_id}",
    "encounter {encounter_id}",
    "review batch {batch_id}",
    "payer edit {edit_id}",
]
ACTION_NOTES = [
    "workqueue: pre-bill review",
    "action: verify before resubmission",
    "action: route to denial prevention team",
    "action: confirm payer rule",
    "action: attach support if available",
    "status: analyst review recommended",
]
REFERENCE_TERMS = [
    "claim reference",
    "review reference",
    "case reference",
    "prebill reference",
    "audit reference",
]
DISTRACTOR_CONTEXTS = [
    "Eligibility was checked separately, but the line still needs review.",
    "Coverage appears active, but another payer rule may still apply.",
    "Authorization details may be present, but the claim still needs validation.",
    "Coding was reviewed, but payer edits may still require follow-up.",
    "Documentation may be available, but the record should be confirmed.",
    "Filing date appears within the normal workflow, but the payer response remains unresolved.",
    "The service may be covered, but another denial condition should be reviewed.",
    "No duplicate was confirmed during intake, but adjudication notes still require review.",
    "Other insurance information may be current, but payer sequencing should be checked.",
    "Medical necessity support may exist, but the claim needs final analyst review.",
]
TYPO_REPLACEMENTS = {
    "claim": "clm",
    "service": "svc",
    "authorization": "auth",
    "documentation": "docs",
    "procedure": "proc",
    "diagnosis": "dx",
    "medical": "med",
    "benefit": "bnft",
}


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def normalize_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def add_typo_noise(text):
    """Add light billing-note shorthand without encoding the target label."""
    if random.random() >= TYPO_NOISE_RATE:
        return text

    candidates = [word for word in TYPO_REPLACEMENTS if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE)]
    if not candidates:
        return text
    word = random.choice(candidates)
    return re.sub(rf"\b{word}\b", TYPO_REPLACEMENTS[word], text, count=1, flags=re.IGNORECASE)


def load_taxonomy():
    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TAXONOMY_PATH}. Run step09_rarc_taxonomy.py first."
        )
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    required = {"label_id", "label", "display_name"}
    missing = required - set(taxonomy.columns)
    if missing:
        raise ValueError(f"Taxonomy missing required columns: {sorted(missing)}")
    return taxonomy.sort_values("label_id").reset_index(drop=True)


def load_real_rarc_template_lookup():
    if not REAL_RARC_PATH.exists():
        return {}, 0

    real_rarc = pd.read_csv(REAL_RARC_PATH)
    required = {"code", "description"}
    missing = required - set(real_rarc.columns)
    if missing:
        raise ValueError(f"{REAL_RARC_PATH} missing required columns: {sorted(missing)}")

    real_rarc = real_rarc.copy()
    real_rarc["code"] = real_rarc["code"].astype(str).str.strip().str.upper()
    real_rarc["description"] = real_rarc["description"].astype(str).map(normalize_text)
    real_rarc = real_rarc[(real_rarc["code"] != "") & (real_rarc["description"] != "")]
    code_to_description = dict(zip(real_rarc["code"], real_rarc["description"]))

    lookup = {}
    for label, codes in RARC_TO_CATEGORY.items():
        templates = []
        for code in codes:
            description = code_to_description.get(code.upper())
            if description:
                # Store as-is — no prefix, no augmentation
                templates.append(description)
        if templates:
            lookup[label] = templates
    return lookup, len(real_rarc)


def mutate_template(template, label, sequence_id):
    text = template
    replacements = {
        r"\bPatient\b": random.choice(PATIENT_TERMS).capitalize(),
        r"\bpatient\b": random.choice(PATIENT_TERMS),
        r"\bMember\b": random.choice(PATIENT_TERMS).capitalize(),
        r"\bmember\b": random.choice(PATIENT_TERMS),
        r"\bPayer\b": random.choice(PAYER_TERMS).capitalize(),
        r"\bpayer\b": random.choice(PAYER_TERMS),
        r"\bservice\b": random.choice(SERVICE_TERMS),
        r"\bService\b": random.choice(SERVICE_TERMS).capitalize(),
    }
    for pattern, replacement in replacements.items():
        if random.random() < 0.45:
            text = re.sub(pattern, replacement, text)

    text = f"{random.choice(PREFIXES)}{text}{random.choice(SUFFIXES)}"
    if random.random() < 0.18:
        text = text.replace(".", ";")
    if random.random() < 0.12:
        text = text.upper() if random.random() < 0.35 else text.lower()
    if random.random() < 0.75:
        context = random.choice(CLAIM_CONTEXTS).format(
            dos=f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            line=random.randint(1, 12),
            claim_id=f"CLM{sequence_id:06d}",
            encounter_id=f"ENC{random.randint(10000, 99999)}",
            batch_id=f"B{random.randint(100, 999)}",
            edit_id=f"EDT-{random.randint(100, 999)}",
        )
        text = f"{text} ({context})"
    if random.random() < 0.45:
        text = f"{text} {random.choice(ACTION_NOTES)}."
    if random.random() < DISTRACTOR_CONTEXT_RATE:
        text = f"{text} {random.choice(DISTRACTOR_CONTEXTS)}"
    text = add_typo_noise(text)
    # Guaranteed neutral uniqueness. The reference number is not category-coded,
    # so it prevents duplicate synthetic text without leaking the label.
    text = f"{text} [{random.choice(REFERENCE_TERMS)} REF{sequence_id:06d}]"
    return normalize_text(text)



def mutate_real_description(template, sequence_id):
    """
    Light variation on real X12 descriptions.
    Only patient/payer/service synonyms are swapped.
    No fake suffixes, no claim context, no distractor sentences.
    This keeps the meaning accurate while generating enough unique rows.
    """
    text = template
    replacements = {
        r"\bPatient\b": random.choice(PATIENT_TERMS).capitalize(),
        r"\bpatient\b": random.choice(PATIENT_TERMS),
        r"\bMember\b": random.choice(PATIENT_TERMS).capitalize(),
        r"\bmember\b": random.choice(PATIENT_TERMS),
        r"\bPayer\b": random.choice(PAYER_TERMS).capitalize(),
        r"\bpayer\b": random.choice(PAYER_TERMS),
        r"\bservice\b": random.choice(SERVICE_TERMS),
        r"\bService\b": random.choice(SERVICE_TERMS).capitalize(),
    }
    for pattern, replacement in replacements.items():
        if random.random() < 0.45:
            text = re.sub(pattern, replacement, text)
    return normalize_text(text)


def generate_category_rows(label_id, label, display_name, n_rows, template_lookup=None):
    template_lookup = template_lookup or {}
    if label not in BASE_TEMPLATES and label not in template_lookup:
        raise KeyError(f"No templates configured for label: {label}")

    real_templates = template_lookup.get(label, [])
    synthetic_templates = BASE_TEMPLATES.get(label, [])

    # Combine both pools — real descriptions as base, synthetic for variety
    # Weight: 60% real (if available), 40% synthetic
    has_real = len(real_templates) > 0
    all_templates = real_templates + synthetic_templates

    if not all_templates:
        raise KeyError(f"No templates configured for label: {label}")

    rows = []
    seen = set()
    attempts = 0
    max_attempts = n_rows * 50

    while len(rows) < n_rows and attempts < max_attempts:
        attempts += 1

        # Pick from real or synthetic pool
        if has_real and random.random() < 0.60:
            template = random.choice(real_templates)
            text = mutate_real_description(template, attempts)
            source_type = "official_rarc"
        else:
            template = random.choice(synthetic_templates)
            text = mutate_template(template, label, attempts)
            source_type = "synthetic_rarc_style"

        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "text": text,
                "label_id": int(label_id),
                "label": label,
                "display_name": display_name,
                "source_type": source_type,
            }
        )

    actual = len(rows)
    if actual < n_rows:
        print(f"  WARNING: Only generated {actual}/{n_rows} rows for {label}")

    return rows


def make_quality_report(dataset, train_df, val_df, test_df, real_rarc_count, real_rarc_labels):
    split_counts = pd.DataFrame(
        [
            {"split": "full", "rows": len(dataset)},
            {"split": "train", "rows": len(train_df)},
            {"split": "validation", "rows": len(val_df)},
            {"split": "test", "rows": len(test_df)},
        ]
    )

    label_balance = (
        dataset.groupby(["label_id", "label"])
        .agg(rows=("text", "size"), unique_text=("text", "nunique"))
        .reset_index()
    )
    label_balance["row_pct"] = (label_balance["rows"] / len(dataset) * 100).round(4)
    duplicate_text_rows = int(dataset["text"].duplicated().sum())

    quality_rows = [
        {"metric": "total_rows", "value": len(dataset)},
        {"metric": "unique_text_rows", "value": dataset["text"].nunique()},
        {"metric": "duplicate_text_rows", "value": duplicate_text_rows},
        {"metric": "labels", "value": dataset["label"].nunique()},
        {"metric": "rows_per_category_target", "value": str(ROWS_PER_CATEGORY)},
        {"metric": "random_state", "value": RANDOM_STATE},
        {"metric": "distractor_context_rate", "value": DISTRACTOR_CONTEXT_RATE},
        {"metric": "typo_noise_rate", "value": TYPO_NOISE_RATE},
        {"metric": "real_rarc_csv_path", "value": str(REAL_RARC_PATH)},
        {"metric": "real_rarc_codes_loaded", "value": real_rarc_count},
        {"metric": "labels_using_real_rarc", "value": len(real_rarc_labels)},
        {"metric": "train_size", "value": len(train_df)},
        {"metric": "validation_size", "value": len(val_df)},
        {"metric": "test_size", "value": len(test_df)},
        {
            "metric": "real_denial_text_used",
            "value": "Official RARC descriptions augmented with synthetic context" if real_rarc_labels else "No - synthetic RARC-style text",
        },
    ]

    try:
        with pd.ExcelWriter(OUTPUT_DIR / "rarc_dataset_quality_report.xlsx") as writer:
            pd.DataFrame(quality_rows).to_excel(writer, sheet_name="summary", index=False)
            split_counts.to_excel(writer, sheet_name="splits", index=False)
            label_balance.to_excel(writer, sheet_name="label_balance", index=False)
    except Exception as exc:
        print(f"Excel quality report skipped: {exc}")

    report = pd.DataFrame(quality_rows)
    report.to_csv(QUALITY_REPORT_PATH, index=False)
    return report, label_balance


def save_chart(label_balance, train_df, val_df, test_df):
    split_frames = []
    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        temp = split_df["label"].value_counts().rename_axis("label").reset_index(name="rows")
        temp["split"] = split_name
        split_frames.append(temp)
    split_balance = pd.concat(split_frames, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(label_balance["label"], label_balance["rows"], color="#4E79A7")
    axes[0].set_title("Synthetic RARC Rows by Root Cause")
    axes[0].set_xlabel("Rows")

    labels = sorted(split_balance["label"].unique())
    x = np.arange(len(labels))
    width = 0.25
    colors = {"train": "#59A14F", "validation": "#F28E2B", "test": "#E15759"}
    for i, split_name in enumerate(["train", "validation", "test"]):
        values = [
            int(split_balance.loc[
                (split_balance["split"] == split_name) & (split_balance["label"] == label),
                "rows",
            ].sum())
            for label in labels
        ]
        axes[1].bar(x + (i - 1) * width, values, width, label=split_name, color=colors[split_name])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_title("Train / Validation / Test Label Balance")
    axes[1].set_ylabel("Rows")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print_section("STEP 10 - SYNTHETIC RARC-STYLE DATASET")
    taxonomy = load_taxonomy()
    real_rarc_lookup, real_rarc_count = load_real_rarc_template_lookup()
    real_rarc_labels = sorted(real_rarc_lookup)
    print(f"Loaded taxonomy: {len(taxonomy)} labels")
    if real_rarc_lookup:
        print(f"Loaded real RARC CSV: {real_rarc_count:,} codes")
        print(f"Labels using real RARC descriptions: {real_rarc_labels}")
    else:
        print(f"Real RARC CSV not found at {REAL_RARC_PATH}; using synthetic templates.")

    rows = []
    for item in taxonomy.itertuples(index=False):
        n_rows = ROWS_PER_CATEGORY.get(item.label, 200)
        print(f"Generating {n_rows:,} rows for {item.label}...")
        rows.extend(
            generate_category_rows(
                label_id=item.label_id,
                label=item.label,
                display_name=item.display_name,
                n_rows=n_rows,
                template_lookup=real_rarc_lookup,
            )
        )

    dataset = pd.DataFrame(rows)
    dataset["text"] = dataset["text"].apply(normalize_text)
    dataset = dataset.drop_duplicates(subset=["text"]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    dataset["text_length"] = dataset["text"].str.len()
    dataset["word_count"] = dataset["text"].str.split().str.len()

    print_section("DATASET QUALITY")
    print(f"Rows: {len(dataset):,}")
    print(f"Unique text: {dataset['text'].nunique():,}")
    print("\nLabel counts:")
    print(dataset["label"].value_counts().sort_index().to_string())

    # Ensure minimum rows per class for stratified split
    min_rows = dataset["label_id"].value_counts().min()
    if min_rows < 3:
        print(f"  WARNING: Some classes have fewer than 3 rows — disabling stratify for split.")
        stratify_col = None
        stratify_temp = None
    else:
        stratify_col = dataset["label_id"]
        stratify_temp = None  # set after first split

    train_df, temp_df = train_test_split(
        dataset,
        train_size=TRAIN_SIZE,
        stratify=stratify_col,
        random_state=RANDOM_STATE,
    )
    relative_val_size = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    # Check temp_df too
    min_temp = temp_df["label_id"].value_counts().min()
    stratify_temp = temp_df["label_id"] if min_temp >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=stratify_temp,
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    dataset.to_csv(DATASET_PATH, index=False)
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    quality_report, label_balance = make_quality_report(
        dataset,
        train_df,
        val_df,
        test_df,
        real_rarc_count=real_rarc_count,
        real_rarc_labels=real_rarc_labels,
    )
    save_chart(label_balance, train_df, val_df, test_df)

    print_section("SAVED OUTPUTS")
    print(f"Full dataset     : {DATASET_PATH}")
    print(f"Train split      : {TRAIN_PATH}")
    print(f"Validation split : {VAL_PATH}")
    print(f"Test split       : {TEST_PATH}")
    print(f"Quality report   : {QUALITY_REPORT_PATH}")
    print(f"Chart            : {CHART_PATH}")

    print("\nQuality summary:")
    print(quality_report.to_string(index=False))
    print("\nSTEP 10 COMPLETE")


if __name__ == "__main__":
    main()
