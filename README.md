# WellMind Data Solutions - Claim Denial Prediction & Root-Cause Classifier

An end-to-end healthcare revenue cycle management demo that predicts Medicare claim denial/underpayment risk, explains the top risk drivers, and prepares an NLP pipeline for denial root-cause classification.

This project uses public CMS Medicare Physician & Other Practitioners utilization/payment data and synthetic RARC-style denial text for NLP. It is designed as a reproducible portfolio project for RCM analytics, denial prevention, and first-pass resolution workflows.

Before a claim is submitted, the system estimates the probability that it will be denied or underpaid and explains why, so the billing team can fix issues before the rejection cycle begins.


Generated outputs are saved locally as CSV, PNG, PKL/JSON, and report artifacts. Large generated datasets are intentionally not recommended for GitHub commits.

## Data Source

Dataset:

- CMS Medicare Physician & Other Practitioners - by Provider and Service, 2023
- Public use file; no approval required


| Metric | Value |
|---|---:|
| Best model | LightGBM Tuned |
| Test ROC-AUC | 0.9819 |
| Test PR-AUC | 0.9480 |
| Test F1 | 0.8732 |
| Test Precision | 0.8746 |
| Test Recall | 0.8719 |
| Test Brier Score | 0.0453 |



| Decile | Risk rate | Lift | Captured risk |
|---:|---:|---:|---:|
| Top 10% | 99.39% | 4.97x | 49.69% |
| Top 20% | 75.23% | 3.76x | 37.61% |




## Installation

Create and activate a virtual environment:

```powershell
py -m venv env
.\env\Scripts\activate
```

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

If using the active environment:

```powershell
python -m pip install -r requirements.txt
```

The model is designed for ranking and prevention workflows:

- High PR-AUC means the model is strong at finding high-risk claims in an imbalanced setting.
- Balanced precision/recall reduces the chance of a model that over-flags too many claims.
- Decile lift shows business value: the top-risk deciles concentrate a large share of risky claims.
- SHAP/feature importance artifacts explain which features drive predictions.

Top model drivers observed:

- charge-per-service proxy
- HCPCS code frequency and smoothed target signal
- provider state signal
- provider specialty signal
- total services and beneficiary volume


This project demonstrates the analytics foundation for that workflow: predict risk before submission, identify the root cause, and route the claim to the right fix team.
