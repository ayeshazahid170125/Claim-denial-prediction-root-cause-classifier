# Model Card: Claim Denial Prediction & Root Cause Classifier

**Project:** WellMind Data Solutions — Claim Denial Prediction System
**Version:** 1.0
**Last Updated:** June 2026
**Maintainer:** WellMind Data Solutions

---

## Overview

This system contains two machine learning models that work together to help healthcare billing teams catch claim denials before submission:

1. **Denial Risk Model** — predicts the probability that a Medicare claim will be denied
2. **Root Cause Classifier** — reads a denial remark and sorts it into one of 10 standard categories

Both models are part of a portfolio demonstration built on public data. They are not deployed in a live clinical or billing environment.

---

## Model 1: Denial Risk Model

### Purpose
Estimates the probability that a given Medicare claim will be denied, based on provider, billing, and utilization characteristics, before the claim is submitted.

### Model Type
LightGBM ensemble classifier (gradient-boosted decision trees), with feature importance explainability.

### Training Data
- **Source:** CMS Medicare Fee-for-Service Provider Utilization & Payment Data (PUF), 2023 release
- **Size:** 9.6 million provider-service rows
- **Access:** Public domain, no approval required
- **Target variable:** A denial-risk proxy constructed from claim-level business rules (e.g., payment-to-charge ratios, zero-payment flags), since the public CMS PUF does not contain real denial outcomes. This is explicitly documented and is not a substitute for real payer remittance data.

### Features Used
- Provider specialty, credential group, state, rural/urban classification (RUCA)
- Billing code (HCPCS/CPT) frequency and historical denial-rate encoding
- Log-transformed utilization metrics (total services, charges, payments, beneficiaries)
- Business-rule flags (e.g., charge less than payment, zero payment, services less than beneficiaries)
- Entity type, Medicare participation status, place of service

### Performance
| Metric | Value |
|---|---|
| Test set accuracy | ~80% (80,058 / 100,000 correct) |
| Predicted high-risk rows | 22,817 |
| Actual high-risk rows | 19,942 |
| Primary evaluation metric | PR-AUC (chosen for class imbalance) |

### Explainability
Feature-importance-based driver attribution is shown alongside each prediction, highlighting which inputs pushed the risk score up or down.

### Intended Use
Portfolio demonstration of a pre-submission denial risk screening tool for a Revenue Cycle Management (RCM) audience. Designed to illustrate how a real system in this space might be structured.

### Limitations
- The target label is a constructed proxy, not a real denial outcome from a payer. Performance on the public PUF data does not guarantee equivalent performance on real payer remittance data.
- Trained only on Medicare Fee-for-Service data; does not reflect commercial payer behavior, Medicaid, or Medicare Advantage denial patterns.
- High-cardinality encodings (billing code, provider type, state) are derived from the training set; codes not seen during training fall back to dataset-average values, which may understate or overstate risk for rare or new codes.

### Ethical Considerations
- No patient-level or individually identifiable data is used. CMS PUF data is provider- and service-level aggregate data.
- The model should not be used to make real coverage, payment, or clinical decisions. It is a demonstration system only.

---

## Model 2: Root Cause Classifier (NLP)

### Purpose
Reads the free-text reason for a claim denial and classifies it into one of 10 standard root-cause categories, with a suggested first-pass fix and the team responsible for resolving it.

### Model Type
TF-IDF vectorization + Logistic Regression (selected as best model). DistilBERT was also trained and evaluated as a comparison.

### Training Data
- **Source:** Official X12 Remittance Advice Remark Code (RARC) descriptions (932 codes), augmented with structured synthetic claim-review context for categories with limited real-code coverage.
- **Size:** 1,994 labeled examples (1,395 train / 299 validation / 300 test)
- **Important disclosure:** The underlying RARC code-to-category mapping is based on real, official X12 codes. Where a category had very few official codes (e.g., timely filing had only 2), additional synthetic training examples were generated to reach a workable training set size. This mix is documented per-row in the dataset (`source_type` column: `official_rarc` or `synthetic_rarc_style`) and is not presented as real-world payer remittance text.

### Label Categories
1. Eligibility / Coverage
2. Coding Error
3. Authorization / Referral
4. Duplicate Claim
5. Coverage / Benefit Exclusion
6. Timely Filing
7. Medical Necessity
8. Coordination of Benefits
9. Documentation
10. Other Administrative

### Performance
| Model | Test Accuracy | Test F1 (weighted) | Training Time |
|---|---|---|---|
| **TF-IDF + Logistic Regression (selected)** | 98.3% | 0.9834 | 1.9 sec |
| DistilBERT (comparison) | 86.0% | 0.8452 | 80 sec |

TF-IDF outperformed DistilBERT on this dataset primarily due to the limited training set size and the presence of distinctive keyword patterns in RARC-style text — conditions that favor a lexical model over a deep transformer.

### Intended Use
Portfolio demonstration of automated denial-reason categorization for an RCM audience, showing how a remark could be triaged to the correct internal team with a suggested fix.

### Limitations
- Training data is a mix of official code descriptions and synthetic text, not real-world free-text remarks written by payer adjudicators, which tend to be noisier and less consistent than the training distribution.
- Some categories (timely filing, authorization, duplicate claim) have very few official RARC codes, so the model has seen limited real-world phrasing for these categories.
- Performance on genuinely messy, real payer remittance text is expected to be lower than the reported test metrics, which were measured on data drawn from the same construction process as the training set.

### Ethical Considerations
- No real patient or claim records were used to build the text dataset.
- The model's category labels and suggested fixes are illustrative; a production deployment should be validated against real remittance data and reviewed by RCM subject matter experts before use in an operational workflow.

---

## System-Level Notes

- Both models are served through a FastAPI inference endpoint and a Streamlit dashboard for demonstration purposes.
- End-to-end latency for a combined risk + root-cause prediction is under 100ms in local testing.
- The system was manually verified against four hand-constructed test scenarios (two high-risk, two low-risk) covering different provider types, billing codes, and remark texts; all four produced risk scores and root-cause labels consistent with expert expectation.

## Out-of-Scope Uses

This system should **not** be used to:
- Make real coverage, payment, or denial decisions for any patient or claim
- Replace a certified medical billing or coding professional's judgment
- Process or store real patient health information (PHI)

## Data Sources & Attribution

- CMS Medicare Physician & Other Practitioners — by Provider and Service, 2023 (Centers for Medicare & Medicaid Services, public domain)
- X12 Remittance Advice Remark Codes (X12.org, official external code list)
