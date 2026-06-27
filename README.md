# Claim Denial Prediction & Root Cause Classifier

An end-to-end healthcare Revenue Cycle Management project that predicts claim denial risk and classifies denial remark text into operational root-cause categories.

The project combines structured CMS Medicare provider-service data with a RARC-style NLP dataset to demonstrate how billing teams can identify high-risk claims before submission and route denial reasons to the right team.

## Repository Order

```text
.
|-- app/                 FastAPI service and Streamlit dashboard
|-- src/                 Step-by-step data, ML, and NLP pipeline scripts
|-- reports/             EDA, cleaning, feature, and modeling summary CSVs
|-- outputs/             Lightweight NLP datasets, metrics, and charts
|-- docs/                model cards, target definition, and full report
|-- scripts/             small helper scripts and API smoke tests
|-- run_pipeline.ps1     runs the full project pipeline
|-- run_app.ps1          starts the API and dashboard
|-- setup_project.ps1    creates environment and installs requirements
|-- requirements.txt
`-- README.md
```

Large raw data files, trained model binaries, virtual environments, and generated heavy artifacts are intentionally ignored so the repository stays clean.

## Problem Statement

Healthcare billing teams lose time and revenue when claims are denied after submission. This project demonstrates a pre-submission analytics workflow that answers two questions:

1. How likely is this claim to be denied?
2. If denied, what is the most likely operational root cause?

## Data Sources

- CMS Medicare Physician & Other Practitioners by Provider and Service public use data
- X12 Remittance Advice Remark Code descriptions
- Synthetic RARC-style denial text generated from the documented taxonomy

No PHI or patient-level records are used.

## Data Preprocessing

The structured data pipeline is organized in `src/`:

1. `step01_eda.py` loads and profiles the raw CMS data.
2. `step02_null_detection.py` audits null patterns and string-based missing values.
3. `step03_duplicate_outlier.py` checks duplicates, outliers, and business-rule anomalies.
4. `step04_cleaning.py` creates the cleaned Week 1 dataset.
5. `step05_credentials.py` standardizes provider credentials.
6. `step06_encoding.py` builds modeling-ready flags, logs, and encoded features.
7. `step07_premodel_eda.py` defines the denial proxy target and pre-model summaries.

Summary outputs are saved under `reports/`.

## Modeling

The denial risk model is trained in `src/step08_model.py`.

The modeling workflow includes:

- train/validation/test split
- class imbalance handling
- baseline and boosted tree model comparison
- Optuna tuning
- threshold review
- feature importance and SHAP-style driver outputs
- bias and overfit audit reports

The selected model artifacts are stored locally under `model_outputs/`, which is ignored by Git because the files are large and generated.

## NLP Root-Cause Classifier

The NLP pipeline is organized as:

1. `step09_rarc_taxonomy.py` creates the root-cause taxonomy.
2. `step10_synthetic_rarc_data.py` builds the RARC-style text dataset.
3. `step11_nlp_classifier.py` trains and compares TF-IDF and DistilBERT classifiers.

Tracked lightweight NLP outputs are under `outputs/nlp/`.

## Results

Denial risk model:

- predicts a pre-submission denial-risk probability
- groups predictions into low, medium, and high risk tiers
- returns top drivers for review

Root-cause classifier:

- classifies denial remarks into operational categories
- returns confidence scores
- provides first-pass recommended fixes

See `docs/MODEL_CARD.md` and `docs/COMPLETE_PROJECT_REPORT.md` for the full limitations and methodology.

## Run Locally

Set up the environment:

```powershell
.\setup_project.ps1
```

Run the full pipeline:

```powershell
.\run_pipeline.ps1
```

Start the demo app:

```powershell
.\run_app.ps1
```

The app starts:

- FastAPI: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Streamlit dashboard: launched by Streamlit

## API Smoke Test

After starting the API:

```powershell
python scripts\sample_api_request.py
```

Main endpoints:

- `GET /health`
- `POST /predict/denial-risk`
- `POST /predict/root-cause`
- `POST /predict/full`

## Important Notes

- This is a portfolio demonstration, not a production billing system.
- The denial label is a documented proxy because CMS public PUF data does not include real payer denial outcomes.
- The NLP training data combines official RARC descriptions with synthetic RARC-style examples.
- This project should not be used to process PHI or make real coverage/payment decisions.
