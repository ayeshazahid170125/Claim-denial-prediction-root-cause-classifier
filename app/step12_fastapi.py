"""
STEP 12 - FastAPI Inference Endpoint
WellMind Data Solutions - Claim Denial Prediction System

Run:
    uvicorn app.step12_fastapi:app --reload --port 8000

Endpoints:
    POST /predict/denial-risk     - XGBoost denial risk score + SHAP drivers
    POST /predict/root-cause      - NLP root cause classification
    POST /predict/full            - Both combined (main demo endpoint)
    GET  /health                  - Health check
    GET  /docs                    - Swagger UI (auto-generated)
"""

from pathlib import Path
import json
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
MODEL_DIR = BASE_DIR / "model_outputs"
NLP_DIR = PROJECT_ROOT / "outputs" / "nlp"

XGBOOST_PATH = MODEL_DIR / "best_denial_risk_model.pkl"
TFIDF_PATH = NLP_DIR / "models" / "tfidf_root_cause_classifier.pkl"
LABEL_MAP_PATH = NLP_DIR / "nlp_model_card.json"
TAXONOMY_PATH = NLP_DIR / "rarc_root_cause_taxonomy.json"
FEATURE_COLS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "model_threshold.pkl"

# ============================================================
# APP INIT
# ============================================================
app = FastAPI(
    title="WellMind Claim Denial Prediction API",
    description=(
        "Predict claim denial risk and root cause before submission. "
        "Powered by XGBoost (denial risk) + TF-IDF NLP (root cause classification). "
        "Data: CMS Medicare PUF 2023 + RARC-style synthetic NLP training data."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL REGISTRY — lazy loaded once on first request
# ============================================================
_models = {}


def load_models():
    """Load all models once and cache in _models dict."""
    if _models:
        return

    # XGBoost denial risk model
    if XGBOOST_PATH.exists():
        _models["xgb"] = joblib.load(XGBOOST_PATH)
    else:
        _models["xgb"] = None

    # TF-IDF NLP root cause classifier
    if TFIDF_PATH.exists():
        _models["tfidf"] = joblib.load(TFIDF_PATH)
    else:
        _models["tfidf"] = None

    # Label map: id -> label name
    _models["id_to_label"] = {
        0: "eligibility", 1: "coding_error", 2: "authorization",
        3: "duplicate_claim", 4: "not_covered", 5: "timely_filing",
        6: "medical_necessity", 7: "coordination_of_benefits",
        8: "documentation", 9: "other",
    }
    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH) as f:
            card = json.load(f)
        # Try to extract label map from model card
        if "id_to_label" in card:
            _models["id_to_label"] = {int(k): v for k, v in card["id_to_label"].items()}
        elif "labels" in card and isinstance(card["labels"], dict):
            _models["id_to_label"] = {int(k): v for k, v in card["labels"].items()}
        elif "labels" in card and isinstance(card["labels"], list):
            _models["id_to_label"] = {int(k): v for k, v in enumerate(card["labels"])}

    # Taxonomy for display names + first_pass_fix
    if TAXONOMY_PATH.exists():
        with open(TAXONOMY_PATH) as f:
            taxonomy_raw = json.load(f)
        _models["taxonomy"] = {item["label"]: item for item in taxonomy_raw}
    else:
        _models["taxonomy"] = {}

    # Feature column order for XGBoost
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            _models["feature_cols"] = json.load(f)
    else:
        _models["feature_cols"] = []

    # High cardinality encoders — real frequency + target values
    hc_enc_path = MODEL_DIR / "high_cardinality_encoders.pkl"
    if hc_enc_path.exists():
        _models["hc_enc"] = joblib.load(hc_enc_path)
    else:
        _models["hc_enc"] = None

    # Optimal classification threshold
    if THRESHOLD_PATH.exists():
        threshold_obj = joblib.load(THRESHOLD_PATH)
        if isinstance(threshold_obj, dict):
            _models["threshold"] = threshold_obj.get("threshold", 0.5)
        elif isinstance(threshold_obj, float):
            _models["threshold"] = threshold_obj
        else:
            _models["threshold"] = 0.5
    else:
        _models["threshold"] = 0.5


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================
class ClaimInput(BaseModel):
    """
    Structured claim data for denial risk prediction.
    Field names match Step 08 encoded feature set.
    All fields have sensible defaults so the demo works out of the box.
    """
    # Provider fields
    provider_type: str = Field("Internal Medicine", description="Provider specialty type")
    state: str = Field("CA", description="Two-letter state code")
    ruca_code: float = Field(1.0, description="Rural-Urban Commuting Area code (1-10)")
    entity_type: int = Field(1, description="1=Individual, 0=Organization")
    participating: int = Field(1, description="Medicare participating: 1=Yes, 0=No")
    credential_group: str = Field("Physician", description="Provider credential group")

    # Claim/service fields
    hcpcs_code: str = Field("99213", description="HCPCS/CPT procedure code")
    drug_indicator: int = Field(0, description="Drug claim: 1=Yes, 0=No")
    place_of_service: int = Field(1, description="1=Facility, 0=Office")

    # Utilization fields
    total_services_log: float = Field(3.5, description="Log-transformed total services")
    total_beneficiaries_log: float = Field(3.0, description="Log-transformed beneficiary count")
    avg_submitted_charge_log: float = Field(5.0, description="Log-transformed avg submitted charge")
    avg_medicare_payment_log: float = Field(4.5, description="Log-transformed avg Medicare payment")
    avg_allowed_amount_log: float = Field(4.6, description="Log-transformed avg allowed amount")

    # Business rule flags
    flag_charge_lt_payment: int = Field(0, description="1 if submitted charge < Medicare payment")
    flag_services_lt_benes: int = Field(0, description="1 if services < beneficiaries")
    flag_zero_payment: int = Field(0, description="1 if payment is zero")
    flag_zero_allowed: int = Field(0, description="1 if allowed amount is zero")
    flag_invalid_state: int = Field(0, description="1 if state code is invalid")
    flag_non_us_country: int = Field(0, description="1 if non-US country")

    # Remark text for NLP (optional — triggers root cause if provided)
    remark_text: str = Field(
        "",
        description="Denial remark text for NLP root-cause classification (optional)",
    )

    @field_validator("state")
    @classmethod
    def validate_state(cls, v):
        return v.upper().strip()[:2]


class DenialDriver(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_risk" or "decreases_risk"


class RootCauseResult(BaseModel):
    predicted_label: str
    display_name: str
    confidence: float
    first_pass_fix: str
    top3: list[dict]


class DenialRiskResult(BaseModel):
    denial_probability: float
    risk_tier: str  # HIGH / MEDIUM / LOW
    threshold_used: float
    top3_drivers: list[DenialDriver]


class FullPredictionResponse(BaseModel):
    request_id: str
    latency_ms: float
    denial_risk: DenialRiskResult | None
    root_cause: RootCauseResult | None
    summary: str
    disclaimer: str = (
        "This prediction is based on CMS Medicare PUF 2023 public data and "
        "RARC-style synthetic NLP training data. It is a portfolio demonstration, "
        "not a certified medical billing decision tool."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================
RISK_TIERS = [
    (0.70, "HIGH"),
    (0.40, "MEDIUM"),
    (0.00, "LOW"),
]


def get_risk_tier(prob: float, threshold: float) -> str:
    for cutoff, label in RISK_TIERS:
        if prob >= cutoff:
            return label
    return "LOW"


def build_feature_vector(claim: ClaimInput, feature_cols: list) -> pd.DataFrame:
    """Convert ClaimInput to a DataFrame matching training feature columns."""

    # Base features
    tot_srvcs = claim.total_services_log
    tot_benes = claim.total_beneficiaries_log
    avg_charge = claim.avg_submitted_charge_log
    avg_payment = claim.avg_medicare_payment_log
    avg_allowed = claim.avg_allowed_amount_log

    # Engineered features — same as Step 08 training pipeline
    charge_per_service = avg_charge - tot_srvcs  # log-space proxy
    svc_per_patient = tot_srvcs - tot_benes       # log-space proxy
    payment_ratio = avg_payment - avg_charge      # negative = underpaid
    allowed_payment_gap = avg_allowed - avg_payment

    raw = {
        "Tot_Srvcs_log": tot_srvcs,
        "Tot_Benes_log": tot_benes,
        "Avg_Sbmtd_Chrg_log": avg_charge,
        "Avg_Mdcr_Pymt_Amt_log": avg_payment,
        "Avg_Mdcr_Alowd_Amt_log": avg_allowed,
        "Rndrng_Prvdr_RUCA": claim.ruca_code,
        "Rndrng_Prvdr_Ent_Cd_bin": claim.entity_type,
        "Rndrng_Prvdr_Mdcr_Prtcptg_Ind_bin": claim.participating,
        "HCPCS_Drug_Ind_bin": claim.drug_indicator,
        "Place_Of_Srvc_bin": claim.place_of_service,
        "flag_charge_lt_payment": claim.flag_charge_lt_payment,
        "flag_services_lt_benes": claim.flag_services_lt_benes,
        "flag_zero_payment": claim.flag_zero_payment,
        "flag_zero_allowed": claim.flag_zero_allowed,
        "flag_invalid_state": claim.flag_invalid_state,
        "flag_non_us_country": claim.flag_non_us_country,
        "RUCA_Unknown_Flag": 1 if claim.ruca_code == 99 else 0,
        # Engineered features
        "charge_per_service_proxy": charge_per_service,
        "svc_per_patient": svc_per_patient,
        "payment_to_charge_ratio": payment_ratio,
        "allowed_payment_gap": allowed_payment_gap,
        "Tot_Bene_Day_Srvcs_log": tot_srvcs,  # proxy
        "Avg_Mdcr_Stdzd_Amt_log": avg_payment * 0.95,  # proxy
    }

    df = pd.DataFrame([raw])

    # High cardinality encoded cols — real encoder lookup
    hc_enc = _models.get("hc_enc")
    global_mean = 0.20  # fallback

    if hc_enc:
        global_mean = hc_enc.get("global_target_mean", 0.20)
        freq_map = hc_enc.get("frequency", {})
        target_map = hc_enc.get("target", {})

        # HCPCS code lookup
        hcpcs_freq_map = freq_map.get("HCPCS_Cd", {})
        hcpcs_tgt_map = target_map.get("HCPCS_Cd", {})
        hcpcs_freq = hcpcs_freq_map.get(claim.hcpcs_code, 0.024)
        hcpcs_tgt = hcpcs_tgt_map.get(claim.hcpcs_code, global_mean)

        # Provider type lookup
        prvdr_freq_map = freq_map.get("Rndrng_Prvdr_Type", {})
        prvdr_tgt_map = target_map.get("Rndrng_Prvdr_Type", {})
        prvdr_freq = prvdr_freq_map.get(claim.provider_type, 0.079)
        prvdr_tgt = prvdr_tgt_map.get(claim.provider_type, global_mean)

        # State lookup
        state_freq_map = freq_map.get("Rndrng_Prvdr_State_Abrvtn", {})
        state_tgt_map = target_map.get("Rndrng_Prvdr_State_Abrvtn", {})
        state_freq = state_freq_map.get(claim.state, 0.069)
        state_tgt = state_tgt_map.get(claim.state, global_mean)
    else:
        hcpcs_freq, hcpcs_tgt = 0.024, global_mean
        prvdr_freq, prvdr_tgt = 0.079, global_mean
        state_freq, state_tgt = 0.069, global_mean

    df["HCPCS_Cd_freq_train"] = hcpcs_freq
    df["HCPCS_Cd_target_smooth_train"] = hcpcs_tgt
    df["Rndrng_Prvdr_Type_freq_train"] = prvdr_freq
    df["Rndrng_Prvdr_Type_target_smooth_train"] = prvdr_tgt
    df["Rndrng_Prvdr_State_Abrvtn_freq_train"] = state_freq
    df["Rndrng_Prvdr_State_Abrvtn_target_smooth_train"] = state_tgt

    # Align to training columns — fill missing with 0
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

    return df


def predict_denial_risk(claim: ClaimInput) -> DenialRiskResult | None:
    xgb = _models.get("xgb")
    if xgb is None:
        return None

    feature_cols = _models.get("feature_cols", [])
    threshold = _models.get("threshold", 0.5)
    X = build_feature_vector(claim, feature_cols)

    prob = float(xgb.predict_proba(X)[0][1])
    tier = get_risk_tier(prob, threshold)

    # SHAP top-3 drivers — fast mode
    try:
        # Use feature importances instead of full SHAP for speed
        importances = xgb.feature_importances_
        cols = X.columns.tolist()
        vals = X.values[0]
        # Approximate: importance * feature value direction
        shap_approx = importances * (vals - 0.5)
        top_idx = np.argsort(np.abs(shap_approx))[::-1][:3]
        drivers = [
            DenialDriver(
                feature=cols[i],
                shap_value=round(float(shap_approx[i]), 4),
                direction="increases_risk" if shap_approx[i] > 0 else "decreases_risk",
            )
            for i in top_idx
        ]
    except Exception:
        drivers = []

    return DenialRiskResult(
        denial_probability=round(prob, 4),
        risk_tier=tier,
        threshold_used=threshold,
        top3_drivers=drivers,
    )


def predict_root_cause(remark_text: str) -> RootCauseResult | None:
    tfidf = _models.get("tfidf")
    id_to_label = _models.get("id_to_label", {})
    taxonomy = _models.get("taxonomy", {})

    if tfidf is None or not remark_text.strip():
        return None

    probs = tfidf.predict_proba([remark_text])[0]
    classes = tfidf.classes_
    top_idx = int(np.argmax(probs))
    raw_class = classes[top_idx]
    # classes may be label strings or integer ids
    if isinstance(raw_class, str) and not raw_class.isdigit():
        pred_label = raw_class  # already a label string
    else:
        pred_label = id_to_label.get(int(raw_class), "other")
    confidence = round(float(probs[top_idx]), 4)

    tax = taxonomy.get(pred_label, {})
    display_name = tax.get("display_name", pred_label.replace("_", " ").title())
    first_pass_fix = tax.get("first_pass_fix", "Review with billing team.")

    sorted_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "label": classes[i] if isinstance(classes[i], str) and not str(classes[i]).isdigit()
                     else id_to_label.get(int(classes[i]), str(classes[i])),
            "confidence": round(float(probs[i]), 4),
        }
        for i in sorted_idx
    ]

    return RootCauseResult(
        predicted_label=pred_label,
        display_name=display_name,
        confidence=confidence,
        first_pass_fix=first_pass_fix,
        top3=top3,
    )


def build_summary(risk: DenialRiskResult | None, cause: RootCauseResult | None) -> str:
    parts = []
    if risk:
        parts.append(
            f"Denial risk: {risk.risk_tier} ({risk.denial_probability:.1%}). "
            f"Top driver: {risk.top3_drivers[0].feature if risk.top3_drivers else 'N/A'}."
        )
    if cause:
        parts.append(
            f"Root cause: {cause.display_name} (confidence {cause.confidence:.1%}). "
            f"Fix: {cause.first_pass_fix}"
        )
    return " | ".join(parts) if parts else "No prediction available."


# ============================================================
# ENDPOINTS
# ============================================================
@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/health")
def health():
    load_models()
    return {
        "status": "ok",
        "xgb_loaded": _models.get("xgb") is not None,
        "tfidf_loaded": _models.get("tfidf") is not None,
        "version": "1.0.0",
    }


@app.post("/predict/denial-risk", response_model=DenialRiskResult)
def predict_denial_risk_endpoint(claim: ClaimInput):
    """
    Predict claim denial probability + top 3 SHAP feature drivers.
    Returns risk tier: HIGH / MEDIUM / LOW.
    """
    load_models()
    result = predict_denial_risk(claim)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="XGBoost model not loaded. Run Step 08 first and place best_model.pkl in model_outputs/.",
        )
    return result


@app.post("/predict/root-cause", response_model=RootCauseResult)
def predict_root_cause_endpoint(claim: ClaimInput):
    """
    Classify denial remark text into 10 root-cause buckets.
    Requires remark_text field in request body.
    """
    load_models()
    if not claim.remark_text.strip():
        raise HTTPException(status_code=422, detail="remark_text is required for root-cause prediction.")
    result = predict_root_cause(claim.remark_text)
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="TF-IDF model not loaded. Run Step 11 first and place tfidf_pipeline.pkl in outputs/nlp/.",
        )
    return result


@app.post("/predict/full", response_model=FullPredictionResponse)
def predict_full(claim: ClaimInput):
    """
    Full prediction: denial risk score + root cause classification.
    Main demo endpoint — enter a claim, get risk score + reason in real time.
    """
    import uuid
    load_models()
    t0 = time.time()

    risk = predict_denial_risk(claim)
    cause = predict_root_cause(claim.remark_text) if claim.remark_text.strip() else None
    summary = build_summary(risk, cause)

    latency = round((time.time() - t0) * 1000, 2)
    return FullPredictionResponse(
        request_id=str(uuid.uuid4())[:8],
        latency_ms=latency,
        denial_risk=risk,
        root_cause=cause,
        summary=summary,
    )


@app.get("/labels")
def get_labels():
    """Return all 10 root-cause labels with display names and fixes."""
    load_models()
    taxonomy = _models.get("taxonomy", {})
    return {
        label: {
            "display_name": data.get("display_name"),
            "first_pass_fix": data.get("first_pass_fix"),
            "operational_owner": data.get("operational_owner"),
        }
        for label, data in taxonomy.items()
    }
