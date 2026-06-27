# RARC Root-Cause Taxonomy Card

Project: WellMind Data Solutions - Claim Denial Prediction System

## Purpose
This taxonomy defines the 10 denial root-cause labels used by the NLP
classifier. It is designed for a public-data portfolio demo where CMS PUF
claim data does not include real denial labels or remittance remark codes.

## Label Set
- 0: eligibility - Eligibility / Coverage
- 1: coding_error - Coding Error
- 2: authorization - Authorization / Referral
- 3: duplicate_claim - Duplicate Claim
- 4: not_covered - Coverage / Benefit Exclusion
- 5: timely_filing - Timely Filing
- 6: medical_necessity - Medical Necessity
- 7: coordination_of_benefits - Coordination of Benefits
- 8: documentation - Documentation
- 9: other - Other Administrative

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
