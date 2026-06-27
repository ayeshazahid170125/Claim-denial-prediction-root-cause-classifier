# Claim Denial Prediction & Root Cause Classifier

Portfolio project for a healthcare Revenue Cycle Management (RCM) use case. The system predicts claim denial risk and classifies denial remark text into root-cause categories so billing teams can prioritize fixes before submission.

## What This Project Includes

- End-to-end data cleaning and feature engineering pipeline for CMS Medicare provider-service data
- Denial risk model with feature attribution and threshold review outputs
- RARC-style root-cause NLP classifier using TF-IDF and a DistilBERT comparison run
- FastAPI inference service for denial risk, root cause, and combined predictions
- Streamlit dashboard for a client-ready demo
- Model documentation in `MODEL_CARD.md`

## Project Structure

```text
.
|-- step01_eda.py
|-- step02_null_detection.py
|-- step03_duplicate_outlier.py
|-- step04_cleaning.py
|-- step05_credentials.py
|-- step06_encoding.py
|-- step07_premodel_eda.py
|-- step08_model.py
|-- step09_rarc_taxonomy.py
|-- step10_synthetic_rarc_data.py
|-- step10a_prepare_real_rarc_codes.py
|-- step11_nlp_classifier.py
|-- step12_fastapi.py
|-- step13_Dashboard.py
|-- run_pipeline.ps1
|-- run_app.ps1
|-- setup_project.ps1
|-- sample_api_request.py
|-- requirements.txt
|-- MODEL_CARD.md
`-- target_definition_card.md
```

Generated datasets, trained model binaries, charts, and large CMS source files are intentionally excluded from Git by `.gitignore`.

## Setup

```powershell
.\setup_project.ps1
```

Manual setup is also supported: create a Python virtual environment, activate it, upgrade `pip`, then install `requirements.txt`.

## Run the Full Pipeline

```powershell
.\run_pipeline.ps1
```

This runs the data preparation, modeling, taxonomy, and NLP training steps in order. The pipeline expects the CMS source data to be available locally in the same folder structure used during development.

## Run the Demo App

```powershell
.\run_app.ps1
```

The script starts:

- FastAPI at `http://localhost:8000`
- Swagger docs at `http://localhost:8000/docs`
- Streamlit dashboard in the browser

## API Endpoints

- `GET /health`
- `POST /predict/denial-risk`
- `POST /predict/root-cause`
- `POST /predict/full`

Example:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/predict/root-cause" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"remark_text":"Claim denied because prior authorization was missing."}'
```

After the API starts, you can also run:

```powershell
python sample_api_request.py
```

## Important Notes

- This is a portfolio/demo project, not a production billing system.
- CMS public data does not contain real payer denial outcomes, so the denial target is a documented proxy.
- The NLP dataset combines official RARC descriptions with synthetic RARC-style examples for underrepresented categories.
- Do not use this project to process PHI or make real coverage/payment decisions.

## Data Sources

- CMS Medicare Physician & Other Practitioners by Provider and Service public use data
- X12 Remittance Advice Remark Code descriptions
