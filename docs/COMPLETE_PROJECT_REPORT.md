# Claim Denial Prediction & Root-Cause Classifier

## Project Completion Report

**Project:** WellMind Data Solutions - Claim Denial Prediction & Root-Cause Classifier  
**Business Hook:** Before a claim is submitted, the system estimates denial/underpayment risk and explains the likely reason, so the billing team can correct the claim before the normal rejection cycle.  
**Domain:** Healthcare Revenue Cycle Management (RCM)  
**Core Methods:** Predictive Machine Learning, Explainability, NLP Root-Cause Classification  
**Current Status:** Data pipeline, ML model, NLP classifier, FastAPI endpoint, and Streamlit dashboard are complete as a portfolio/demo system. Remaining production work would require real payer denial labels and real remittance validation.

---

## 1. Original Requirement

The project requirement was to build an end-to-end claim denial prediction and root-cause classification system using public CMS Medicare claims/utilization data and RARC-based denial reason text.

Required build items:

- Feature engineering on CPT/HCPCS codes, modifiers, provider specialty, service location, claim age/proxy features.
- ML denial-risk classifier with explainability.
- NLP model to classify denial remark text into 10 root-cause buckets.
- FastAPI inference endpoint that accepts claim data and returns risk score plus top denial drivers.
- Dashboard showing denial/risk trends by payer, CPT/HCPCS, provider specialty, and time/trend views.
- Public GitHub repo with reproducible pipeline, model card, and methodology.

RCM problems covered:

- Denial prediction
- Root-cause analysis
- Denial categorization NLP
- First-pass resolution
- Denial trend forecasting foundation

---

## 2. Data Sources Used

### 2.1 CMS Medicare Claims / Utilization Data

The main ML pipeline uses the public CMS Medicare Physician & Other Practitioners dataset.

Source used:

- **CMS Medicare Physician & Other Practitioners - by Provider and Service**
- Public CMS dataset covering Original Medicare Fee-for-Service Part B services, procedures, submitted charges, utilization, and payment information.
- CMS page: https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider
- CMS API documentation: https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider/api-docs

Why this source was selected:

- It is public and does not require approval.
- It contains healthcare billing/utilization fields relevant to denial-risk proxy modeling.
- It includes provider, specialty, HCPCS/CPT-like procedure code, service volume, charges, and payment information.

Important note:

- CMS public utilization data does not contain real claim denial labels.
- Therefore, the ML target was created as a **denial/underpayment risk proxy** using the lowest 20% payment-to-charge ratio.
- This is suitable for a portfolio/demo project, but it should not be presented as a real production denial label model without actual payer denial outcomes.

### 2.2 Official RARC Code Source for NLP

For the NLP root-cause classifier, official Remittance Advice Remark Codes were used.

Source used:

- **X12 Remittance Advice Remark Codes**
- X12 page: https://x12.org/codes/remittance-advice-remark-codes

What RARC codes are:

- RARC codes provide additional explanation for an adjustment already described by a Claim Adjustment Reason Code (CARC), or they convey information about remittance processing.
- These are published industry-standard healthcare remittance codes, not invented labels.

How they were used:

- The official RARC page text was copied and parsed into `outputs/nlp/real_rarc_codes.csv`.
- A mapping was created from real RARC codes to the 10 project root-cause categories.
- Step 10 now uses official RARC descriptions when available and augments them with synthetic claim-review context.

Important note:

- The project uses official RARC descriptions, but not real payer-specific remittance records.
- This improves credibility compared with purely invented templates, while still requiring real-world validation for production use.

---

## 3. Work Completed by Pipeline Step

### Step 01 - Exploratory Data Analysis

File:

- `step01_eda.py`

What was done:

- Loaded the CMS Medicare provider/service data.
- Created initial dataset profile.
- Reviewed numeric and categorical distributions.
- Generated summary outputs for EDA.

Why it matters:

- EDA confirms the shape, field types, missingness, and modeling feasibility before building the pipeline.
- It helps identify useful features such as provider specialty, HCPCS code, service volume, charges, and payment fields.

Outputs:

- Numeric summaries
- Categorical summaries
- Column profile reports
- EDA charts

### Step 02 - Null Detection

File:

- `step02_null_detection.py`

What was done:

- Detected missing values and null-like string values.
- Identified columns requiring cleaning or review.
- Created before/after null tracking outputs.

Why it matters:

- Missing values can create model bias or runtime errors.
- Proper null handling improves data quality and reproducibility.

### Step 03 - Duplicate and Outlier Detection

File:

- `step03_duplicate_outlier.py`

What was done:

- Checked duplicated rows and suspicious duplicates.
- Reviewed numeric outliers.
- Created outlier summary and charts.

Why it matters:

- Duplicate records can inflate model confidence.
- Outliers can distort numeric features and model decisions.

### Step 04 - Data Cleaning

File:

- `step04_cleaning.py`

What was done:

- Cleaned null-like values.
- Standardized fields.
- Applied cleaning rules for numeric and categorical columns.
- Produced cleaning audit reports.

Why it matters:

- This creates a stable modeling dataset and keeps cleaning decisions auditable.

### Step 05 - Credentials / Provider Field Review

File:

- `step05_credentials.py`

What was done:

- Reviewed provider credential-related fields.
- Grouped and audited provider credential/category information.

Why it matters:

- Provider type and specialty are important denial-risk drivers.
- Clean provider categories improve model interpretability.

### Step 06 - Encoding

File:

- `step06_encoding.py`

What was done:

- Encoded categorical variables.
- Deferred high-cardinality columns for safer train-only encoding in the modeling stage.
- Created encoding summaries and mapping files.

Why it matters:

- ML models need numeric inputs.
- High-cardinality features such as HCPCS code, provider specialty, and state must be handled carefully to avoid overfitting and leakage.

### Step 07 - Premodel EDA

File:

- `step07_premodel_eda.py`

What was done:

- Analyzed target proxy behavior before modeling.
- Reviewed risk patterns by place of service, specialty, and HCPCS.
- Created premodel statistical tests and charts.

Why it matters:

- This validates whether the target proxy has meaningful business patterns.
- It helps explain why certain fields are useful for denial-risk prediction.

### Step 08 - Denial / Underpayment Risk ML Model

File:

- `step08_model.py`

What was done:

- Created a denial/underpayment risk target using payment-to-charge ratio.
- Used a train-derived threshold to avoid target leakage.
- Built feature engineering for claim/service risk:
  - service volume features
  - charge-per-service proxy
  - high-volume indicator
  - urban/service-location proxy
  - train-only frequency encoding
  - train-only target encoding for high-cardinality variables
- Dropped payment/charge leakage columns before modeling.
- Compared multiple models:
  - Logistic Regression
  - HistGradientBoosting
  - Random Forest
  - Extra Trees
  - XGBoost
  - LightGBM
- Performed hyperparameter tuning.
- Selected the best model using validation performance.
- Produced explainability outputs and model reports.

Model result setup:

- Six baseline models were trained and evaluated.
- The final model run used a stratified 2.5M-row modeling sample:
  - 1.5M train rows
  - 500K validation rows
  - 500K test rows
- Model selection used validation PR-AUC because the target is imbalanced.
- Final performance below is reported on the untouched holdout test set.

Baseline model comparison:

| Model | Test PR-AUC | Test F1 | ROC-AUC | Status |
|---|---:|---:|---:|---|
| **HistGB Tuned (BEST)** | **0.9457** | **0.8669** | **0.9820** | **Winner - Deployed** |
| HistGradientBoosting | 0.9186 | 0.8313 | 0.9723 | Baseline |
| LightGBM | 0.9091 | 0.8207 | 0.9703 | Baseline |
| XGBoost | 0.9004 | 0.8112 | 0.9670 | Baseline |
| Random Forest | 0.8531 | 0.7608 | 0.9509 | Baseline |
| Extra Trees | 0.8117 | 0.7200 | 0.9346 | Baseline |
| Logistic Regression | 0.7757 | 0.6927 | 0.9210 | Baseline |

Best model result:

| Metric | Value |
|---|---:|
| Best model | HistGB Tuned |
| Test threshold | 0.6922 |
| Test accuracy | 0.9468 |
| Test balanced accuracy | 0.9166 |
| Test precision | 0.8677 |
| Test recall | 0.8662 |
| Test F1 | 0.8669 |
| Test ROC-AUC | 0.9820 |
| Test PR-AUC | 0.9457 |
| Test Brier Score | 0.0512 |

Hyperparameter tuning:

- HistGradientBoosting was tuned using Optuna Bayesian optimization.
- The tuning process used 40 trials and 3-fold cross-validation on a 400K-row tuning sample.
- Best cross-validation PR-AUC: **0.9228**.

Best parameters:

| Parameter | Value |
|---|---:|
| max_iter | 557 |
| max_depth | 10 |
| learning_rate | 0.1456 |
| min_samples_leaf | 77 |
| max_leaf_nodes | 106 |
| l2_regularization | 1.7093 |

Business interpretation:

- The model is strong at ranking high-risk claims.
- Precision and recall are balanced, so the model is not only catching risk but also avoiding too many false alerts.
- Decile lift shows the model can concentrate risky claims in the top review groups, which is useful for RCM workqueue prioritization.

Important controls:

- Payment/charge leakage columns were removed from features.
- Target threshold was derived from train data.
- High-cardinality encoding was fit on training data and then applied to validation/test.
- Model results are strong but should be described as a denial-risk proxy, not real denial outcome prediction.

### Step 09 - RARC Root-Cause Taxonomy

File:

- `step09_rarc_taxonomy.py`

What was done:

- Created a 10-category denial root-cause taxonomy:
  - eligibility
  - coding_error
  - authorization
  - duplicate_claim
  - not_covered
  - timely_filing
  - medical_necessity
  - coordination_of_benefits
  - documentation
  - other

Why it matters:

- A model prediction is more useful when the team knows why the claim is risky.
- These buckets match common RCM workqueue categories and can route claims to the correct team.

### Step 10A - Official RARC Parser

File:

- `step10a_prepare_real_rarc_codes.py`

What was done:

- Added a parser that converts copied X12 RARC page text into a clean CSV.
- Output file:
  - `outputs/nlp/real_rarc_codes.csv`

How it works:

- Reads copied text from the X12 RARC page.
- Extracts:
  - RARC code
  - description
  - start date
  - last modified date
  - notes
- Saves the structured CSV for Step 10.

Why it matters:

- This makes the NLP dataset based on published RARC wording instead of purely invented text.
- It improves project credibility for technical reviewers.

### Step 10 - RARC-Style NLP Dataset Generation

File:

- `step10_synthetic_rarc_data.py`

What was done:

- Generates a balanced 50,000-row NLP dataset.
- Uses official RARC descriptions when `real_rarc_codes.csv` is available.
- Falls back to synthetic templates if official CSV is missing.
- Maps real RARC descriptions into the 10 root-cause categories.
- Adds realistic claim-review context, neutral reference IDs, and light shorthand/typo noise.
- Avoids label-coded leakage.
- Splits data into train, validation, and test sets.

Current generated dataset:

| Split | Rows |
|---|---:|
| Train | 35,000 |
| Validation | 7,500 |
| Test | 7,500 |
| Total | 50,000 |

Quality checks:

- Unique text rows: 50,000
- Duplicate text rows: 0
- Labels covered by official RARC descriptions: 10
- Real RARC codes parsed from copied X12 text: 1,119

Why it matters:

- The NLP model now learns from industry-standard RARC wording plus realistic billing context.
- This creates a stronger root-cause classification demo than fully invented denial text.

### Step 11 - NLP Root-Cause Classifier

File:

- `step11_nlp_classifier.py`

What was done:

- Built NLP pipeline for 10-class root-cause classification.
- Trains a TF-IDF + Logistic Regression baseline.
- Fine-tuned DistilBERT on Kaggle GPU and compared it with a TF-IDF + Logistic Regression baseline.
- Made the code Kaggle-compatible:
  - inputs can be auto-detected from `/kaggle/input`
  - outputs save to `/kaggle/working/outputs/nlp`
- Added leakage checks:
  - train/validation overlap
  - train/test overlap
  - validation/test overlap
- Added warning logic for suspiciously perfect scores.
- Added model card outputs with source summary and limitations.

Latest NLP results after official RARC augmentation:

| Model | Training Time | Test Accuracy | Test Precision Weighted | Test Recall Weighted | Test F1 Weighted | Test F1 Macro |
|---|---:|---:|---:|---:|---:|---:|
| DistilBERT | 1,374.37 sec | 0.9896 | 0.9906 | 0.9896 | 0.9896 | 0.9896 |
| TF-IDF Logistic Regression | 6.84 sec | 0.9891 | 0.9898 | 0.9891 | 0.9890 | 0.9890 |

Best NLP model:

| Metric | Value |
|---|---:|
| Best model | DistilBERT |
| Best epoch | 1 |
| Validation F1 Weighted | 0.9892 |
| Test Accuracy | 0.9896 |
| Test Precision Weighted | 0.9906 |
| Test Recall Weighted | 0.9896 |
| Test F1 Weighted | 0.9896 |

Per-category observations:

- Most categories achieved near-perfect classification.
- `eligibility` achieved F1 0.9506.
- `not_covered` achieved F1 0.9451.
- The main remaining confusion is between eligibility/coverage-related language and not-covered/benefit-exclusion language, which is realistic because those categories can overlap in actual RCM workflows.

Leakage check:

| Check | Result |
|---|---:|
| Train/Validation text overlap | 0 |
| Train/Test text overlap | 0 |
| Validation/Test text overlap | 0 |

Why it matters:

- The NLP model can classify denial remark text into operational root-cause buckets.
- This supports routing claims to the correct fix team:
  - coding team
  - eligibility verification
  - authorization team
  - documentation team
  - COB team
  - appeal/timely filing team

Important note:

- The NLP result is strong because official RARC language is category-specific.
- DistilBERT slightly outperformed the TF-IDF baseline on the Kaggle GPU run.
- It should still be validated on real payer remittance records before production claims.

---

## 4. How the System Solves the Business Problem

### Problem 1: Denials Are Usually Found Too Late

In many RCM workflows, the team discovers a denial after the claim is submitted and processed. This can create a 30-45 day delay.

What this project changes:

- The ML model scores the claim before submission.
- High-risk claims can be reviewed first.
- Billing teams can fix likely issues before the denial happens.

### Problem 2: Teams Need to Know Why a Claim Is Risky

A risk score alone is not enough. The billing team needs an explanation.

What this project changes:

- The ML model produces top risk drivers.
- The NLP model classifies denial reason text into root-cause buckets.
- The system can support workqueue routing by reason category.

### Problem 3: Manual Workqueues Are Not Prioritized

RCM teams often review claims in simple queues, not by actual risk.

What this project changes:

- Risk deciles identify the highest-priority claims.
- The top-risk group can be reviewed first.
- This improves first-pass resolution and reduces rework.

### Problem 4: Denial Trends Are Hard to See

Without analytics, managers may not know which CPT/HCPCS codes, specialties, or providers create the most risk.

What this project changes:

- The pipeline creates reports by HCPCS, provider specialty, service location, and risk segment.
- These outputs become the foundation for dashboard trend views.

---

## 5. Technical Strengths

- Reproducible step-by-step pipeline.
- Public CMS source data.
- Official X12 RARC descriptions used for NLP.
- Multiple ML models compared instead of relying on one model.
- Hyperparameter tuning performed.
- Train/test leakage controls applied.
- High-cardinality categorical handling included.
- Balanced precision and recall.
- Model cards and quality reports generated.
- Kaggle GPU compatibility added for NLP.
- GitHub repository published and updated.

GitHub repositories:

- https://github.com/WELLMIND-DataSolutions/Claim-denial-prediction-root-cause-classifier
- https://github.com/ayeshazahid170125/Claim-denial-prediction-root-cause-classifier

---

## 6. Honest Limitations

### ML Target Limitation

The CMS dataset does not contain actual denial labels. The ML model predicts a denial/underpayment risk proxy based on the lowest 20% payment-to-charge ratio.

This is acceptable for a portfolio and proof-of-concept project, but production validation would require:

- real payer denial outcomes
- real claim status
- CARC/RARC codes from actual remittance data
- external holdout testing by payer/time period

### NLP Data Limitation

The NLP model uses official RARC descriptions plus synthetic claim context. This is stronger than fully fake text, but it is still not the same as noisy real payer remittance notes.

Production validation would require:

- real 835/remittance remark text
- payer-specific variations
- real labeled denial outcomes
- time-based evaluation

### FastAPI and Dashboard Status

FastAPI and Streamlit dashboard entrypoints are included in the current codebase:

- FastAPI app: `app/step12_fastapi.py`
- Streamlit dashboard: `app/step13_Dashboard.py`
- Launcher script: `run_app.ps1`

These components are suitable for a local portfolio demo. Production deployment would still require authentication, monitoring, secure data handling, real payer validation, and environment-specific configuration.

---

## 7. Difference This Project Will Create

If deployed with real claim and remittance data, this system can help an RCM team:

- identify high-risk claims before submission
- reduce first-pass denial rate
- prioritize review workqueues
- explain why a claim is risky
- route claims to the correct team faster
- monitor denial trends by code, provider specialty, and payer
- support ROI conversations with measurable denial reduction scenarios

Example ROI logic:

- If a provider submits $10M in claims
- and the system helps reduce preventable denials by 5%
- potential recovered/protected revenue impact can be framed as approximately $500K

This ROI should be validated with client-specific denial rates and recovery rates.

---

## 8. Current Sprint Status

| Sprint Item | Status | Notes |
|---|---|---|
| Month 1 - CMS data pipeline | Complete | Data loaded, profiled, cleaned, encoded |
| Month 1 - EDA | Complete | EDA, nulls, duplicates, outliers, premodel EDA |
| Month 1 - Feature engineering | Complete | Domain and model-stage features created |
| Month 2 - ML models | Complete | Multiple models compared |
| Month 2 - Tuning | Complete | HistGradientBoosting tuned with Optuna; baselines compared |
| Month 2 - Evaluation | Complete | ROC-AUC, PR-AUC, F1, precision, recall, calibration |
| Month 2 - Explainability | Mostly complete | Feature importance/SHAP-style outputs prepared |
| Month 3 - RARC taxonomy | Complete | 10 root-cause labels |
| Month 3 - NLP dataset | Complete | Official RARC descriptions + synthetic context |
| Month 3 - NLP classifier | Complete | TF-IDF baseline plus DistilBERT fine-tuned on Kaggle GPU; best F1 0.9896 |
| Month 3 - FastAPI endpoint | Complete | Local inference API in `app/step12_fastapi.py` |
| Month 4 - Dashboard | Complete | Streamlit dashboard in `app/step13_Dashboard.py` |
| Month 4 - ROI brief | Pending | Can be created from current outputs |
| Month 4 - Final write-up | In progress | This report is part of final documentation |

---

## 9. Recommended Next Steps

1. Build FastAPI endpoint.
   - Input: claim/provider/service features.
   - Output: risk score, risk tier, top 3 drivers, root-cause prediction.

2. Build dashboard.
   - Risk by HCPCS/CPT.
   - Risk by provider specialty.
   - Risk by location/state.
   - Trend and decile views.

3. Create ROI brief.
   - Include denial reduction examples.
   - Include model decile-lift interpretation.
   - Include limitation that CMS model target is a risk proxy.

4. Add methodology PDF.
   - CMS source citation.
   - X12 RARC source citation.
   - Modeling approach.
   - Leakage prevention.
   - Limitations and production validation plan.

5. Add real-world validation when available.
   - Actual denial labels.
   - Real CARC/RARC remittance records.
   - External holdout by payer or time period.

---

## 10. Final Summary for Supervisor

This project has built a professional end-to-end foundation for a claim denial prediction and root-cause classification system. The CMS-based ML pipeline predicts denial/underpayment risk using a carefully documented proxy target, compares multiple models, applies tuning, and produces strong evaluation results. The NLP pipeline has been upgraded from simple synthetic templates to official X12 RARC descriptions augmented with realistic claim-review context, making the root-cause classifier more credible and aligned with real healthcare remittance standards.

The project is technically strong for a portfolio/proof-of-concept. It is honest about its limitations: CMS public data does not include actual denial labels, and NLP still needs real payer remittance data for production validation. The next major deliverables are FastAPI inference, dashboard, ROI brief, and final methodology PDF.
