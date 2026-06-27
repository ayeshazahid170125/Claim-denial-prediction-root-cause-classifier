"""
STEP 09 - RARC Root-Cause Taxonomy
WellMind Data Solutions - Claim Denial Prediction System

Run:
    py step09_rarc_taxonomy.py

Purpose:
Create a professional, reproducible root-cause taxonomy for denial remark
classification before synthetic NLP data is generated in Step 10.

Important:
CMS public claims PUF data does not include real denial labels, CARC codes, or
RARC codes. This taxonomy is a defensible RCM label framework for a portfolio
demo. Step 10 uses it to generate RARC-style synthetic training text.
"""

from pathlib import Path
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TAXONOMY_PATH = OUTPUT_DIR / "rarc_root_cause_taxonomy.csv"
MAPPING_PATH = OUTPUT_DIR / "rarc_category_mapping.csv"
SUMMARY_PATH = OUTPUT_DIR / "rarc_reference_summary.csv"
MODEL_CARD_PATH = OUTPUT_DIR / "rarc_taxonomy_model_card.md"
TAXONOMY_JSON_PATH = OUTPUT_DIR / "rarc_root_cause_taxonomy.json"


ROOT_CAUSES = [
    {
        "label_id": 0,
        "label": "eligibility",
        "display_name": "Eligibility / Coverage",
        "business_definition": "Patient coverage, enrollment, member ID, or benefit eligibility issue.",
        "buyer_explanation": "Coverage was inactive, missing, expired, or could not be verified for the service date.",
        "example_signals": "inactive coverage; member not eligible; subscriber ID mismatch; benefit period issue",
        "operational_owner": "Front desk / eligibility verification team",
        "first_pass_fix": "Verify active coverage before claim submission and correct subscriber/member identifiers.",
    },
    {
        "label_id": 1,
        "label": "coding_error",
        "display_name": "Coding Error",
        "business_definition": "Procedure, diagnosis, modifier, or code-combination issue.",
        "buyer_explanation": "The submitted coding does not match payer rules, documentation, or claim format requirements.",
        "example_signals": "invalid CPT; modifier missing; diagnosis inconsistent; code not billable",
        "operational_owner": "Coding team",
        "first_pass_fix": "Review CPT/HCPCS, ICD, modifier, and NCCI-style code-pair edits before submission.",
    },
    {
        "label_id": 2,
        "label": "authorization",
        "display_name": "Authorization / Referral",
        "business_definition": "Prior authorization, referral, pre-certification, or approval number issue.",
        "buyer_explanation": "The payer required approval before service, but authorization was missing, invalid, or expired.",
        "example_signals": "prior auth missing; referral required; authorization expired; auth number invalid",
        "operational_owner": "Authorization team",
        "first_pass_fix": "Confirm authorization requirement and attach valid approval details before submission.",
    },
    {
        "label_id": 3,
        "label": "duplicate_claim",
        "display_name": "Duplicate Claim",
        "business_definition": "Claim appears to duplicate a previous submission or paid claim.",
        "buyer_explanation": "The payer identified the claim as already processed or matching another submitted claim.",
        "example_signals": "duplicate service; already paid; repeat claim; same patient/date/service/provider",
        "operational_owner": "Billing operations",
        "first_pass_fix": "Check prior submissions and resubmit only corrected or appeal-ready claims.",
    },
    {
        "label_id": 4,
        "label": "not_covered",
        "display_name": "Coverage / Benefit Exclusion",
        "business_definition": "Service is excluded, non-covered, investigational, bundled, or not payable under plan rules.",
        "buyer_explanation": "The payer policy does not cover the submitted service for this patient or setting.",
        "example_signals": "non-covered service; benefit exclusion; bundled service; policy limitation",
        "operational_owner": "Revenue integrity / billing policy",
        "first_pass_fix": "Validate benefit coverage and payer medical/payment policy before billing.",
    },
    {
        "label_id": 5,
        "label": "timely_filing",
        "display_name": "Timely Filing",
        "business_definition": "Claim, corrected claim, or appeal submitted after payer filing deadline.",
        "buyer_explanation": "The claim was received outside the allowed filing window.",
        "example_signals": "filing limit exceeded; late submission; appeal window expired; proof of timely filing needed",
        "operational_owner": "Billing operations",
        "first_pass_fix": "Track payer filing limits and submit corrected claims before deadline.",
    },
    {
        "label_id": 6,
        "label": "medical_necessity",
        "display_name": "Medical Necessity",
        "business_definition": "Payer does not consider the service medically necessary based on diagnosis or documentation.",
        "buyer_explanation": "The clinical reason submitted does not support the billed service under payer criteria.",
        "example_signals": "diagnosis does not support service; LCD/NCD criteria not met; necessity not established",
        "operational_owner": "Clinical documentation / coding team",
        "first_pass_fix": "Validate diagnosis support, payer policy criteria, and clinical documentation before submission.",
    },
    {
        "label_id": 7,
        "label": "coordination_of_benefits",
        "display_name": "Coordination of Benefits",
        "business_definition": "Primary/secondary payer order or other insurance responsibility issue.",
        "buyer_explanation": "The payer requires another carrier to process first or needs updated COB information.",
        "example_signals": "other insurance primary; COB required; Medicare secondary; payer order mismatch",
        "operational_owner": "Eligibility / billing team",
        "first_pass_fix": "Verify primary payer order and collect updated COB information before submission.",
    },
    {
        "label_id": 8,
        "label": "documentation",
        "display_name": "Documentation",
        "business_definition": "Missing, incomplete, illegible, or insufficient supporting documentation.",
        "buyer_explanation": "The claim needs additional records, notes, orders, or documentation support.",
        "example_signals": "medical records missing; operative note required; documentation incomplete; attachment missing",
        "operational_owner": "Clinical documentation / HIM",
        "first_pass_fix": "Attach required notes, orders, records, and payer-specific documentation before submission.",
    },
    {
        "label_id": 9,
        "label": "other",
        "display_name": "Other Administrative",
        "business_definition": "Administrative or payer-processing reason not captured by the other root-cause buckets.",
        "buyer_explanation": "The denial reason requires manual review because it is not one of the common root causes.",
        "example_signals": "payer processing issue; claim format issue; provider enrollment review; manual review needed",
        "operational_owner": "RCM analyst / billing supervisor",
        "first_pass_fix": "Route to analyst review and update category mapping if a repeatable pattern emerges.",
    },
]


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def validate_taxonomy(df):
    required_cols = {
        "label_id",
        "label",
        "display_name",
        "business_definition",
        "buyer_explanation",
        "example_signals",
        "operational_owner",
        "first_pass_fix",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Taxonomy missing required columns: {sorted(missing)}")
    if df["label_id"].duplicated().any():
        raise ValueError("Duplicate label_id values found.")
    if df["label"].duplicated().any():
        raise ValueError("Duplicate label names found.")
    if sorted(df["label_id"].tolist()) != list(range(len(df))):
        raise ValueError("label_id values must be contiguous from 0 to n-1.")


def write_model_card(df):
    labels = "\n".join(
        f"- {row.label_id}: {row.label} - {row.display_name}"
        for row in df.itertuples(index=False)
    )
    card = f"""# RARC Root-Cause Taxonomy Card

Project: WellMind Data Solutions - Claim Denial Prediction System

## Purpose
This taxonomy defines the 10 denial root-cause labels used by the NLP
classifier. It is designed for a public-data portfolio demo where CMS PUF
claim data does not include real denial labels or remittance remark codes.

## Label Set
{labels}

## Design Principle
The labels are operational RCM buckets: each category maps to an owner and a
first-pass fix workflow. This makes model output useful for prevention, not
only classification.

## Important Limitation
The taxonomy is not claiming that CMS PUF contains actual denial outcomes.
Synthetic Step 10 text is RARC-style training data for demonstration. A real
client deployment should replace or augment it with payer remittance data,
CARC/RARC codes, appeal notes, and adjudication outcomes.

## Leakage Control
This taxonomy contains labels and definitions only. It does not use target
values from the Medicare payment model and does not train any classifier.
"""
    MODEL_CARD_PATH.write_text(card, encoding="utf-8")


def main():
    print_section("STEP 09 - RARC ROOT-CAUSE TAXONOMY")
    taxonomy_df = pd.DataFrame(ROOT_CAUSES).sort_values("label_id").reset_index(drop=True)
    validate_taxonomy(taxonomy_df)

    mapping_df = taxonomy_df[["label_id", "label", "display_name"]].copy()
    summary_df = pd.DataFrame(
        [
            {"metric": "total_root_cause_labels", "value": len(taxonomy_df)},
            {"metric": "label_ids", "value": "0-9"},
            {"metric": "source_type", "value": "RCM root-cause taxonomy for RARC-style synthetic NLP data"},
            {"metric": "real_denial_labels_used", "value": "No"},
            {"metric": "next_step", "value": "Step 10 generates synthetic denial remark text from this taxonomy"},
        ]
    )

    taxonomy_df.to_csv(TAXONOMY_PATH, index=False)
    mapping_df.to_csv(MAPPING_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    TAXONOMY_JSON_PATH.write_text(
        json.dumps(ROOT_CAUSES, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_model_card(taxonomy_df)

    print(f"Saved taxonomy        : {TAXONOMY_PATH}")
    print(f"Saved mapping         : {MAPPING_PATH}")
    print(f"Saved summary         : {SUMMARY_PATH}")
    print(f"Saved JSON taxonomy   : {TAXONOMY_JSON_PATH}")
    print(f"Saved taxonomy card   : {MODEL_CARD_PATH}")

    print("\nRoot-cause labels:")
    print(mapping_df.to_string(index=False))
    print("\nSTEP 09 COMPLETE")


if __name__ == "__main__":
    main()
