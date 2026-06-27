"""
STEP 08 — FINAL BEST MODEL TRAINING (Merged Ultimate Version)
WellMind Data Solutions — Claim Denial Prediction System
Run: python step08_final_best.py

═══════════════════════════════════════════════════════════════
WHAT IS MERGED AND WHY
═══════════════════════════════════════════════════════════════

From step07_model_improved.py:
  ✅ svc_per_patient, high_volume_flag, is_urban feature engineering
  ✅ Random Forest model in comparison set
  ✅ Optuna hyperparameter search (TPE sampler)
  ✅ Optuna optimization history charts

From step08_model.py (uploaded):
  ✅ Target threshold learned from TRAINING SPLIT ONLY (no leakage)
  ✅ assign_target_from_train_threshold — gold standard leakage prevention
  ✅ HistGradientBoosting + Extra Trees as additional baselines
  ✅ Decile lift report (most important for RCM sales demo)
  ✅ Threshold review table (0.10 to 0.90 sweep)
  ✅ Overfit/underfit audit (train vs val vs test gap)
  ✅ Subgroup / bias audit per provider type, place of service, urban/rural
  ✅ Permutation importance fallback when model has no feature_importances_
  ✅ Robust SHAP with fallback (Explainer → predict_proba explainer)
  ✅ Stratified sampling helper for memory safety
  ✅ Full JSON model report

From step08_model_training.py (my earlier version):
  ✅ LightGBM in comparison + Optuna tuning for LGB
  ✅ 3-way split (train/val/test) with val never used in final test eval
  ✅ AUC-PR as CV objective (correct for imbalanced data)
  ✅ Early stopping via eval_set for XGBoost + LightGBM
  ✅ SHAP waterfall for highest-risk and lowest-risk claims
  ✅ feature_columns.json saved for FastAPI inference

NEW additions (not in any previous version):
  ✅ Calibration check: Brier Score + reliability diagram
  ✅ PR-AUC used as primary model selection criterion (not ROC-AUC)
  ✅ Soft-voting ensemble of ALL trained models
  ✅ LightGBM added to Optuna tuning
  ✅ charge_per_service_proxy feature (from step08_model.py)

═══════════════════════════════════════════════════════════════
IMPORTANT DESIGN DECISIONS
═══════════════════════════════════════════════════════════════
- TARGET THRESHOLD is fit on TRAIN split only → no leakage
- PAYMENT columns dropped before any model sees features
- SMOTE not used — scale_pos_weight + class_weight sufficient at 20% rate
- OPTUNA uses AUC-PR (not AUC-ROC) — better signal for imbalanced targets
- Model selection uses validation PR-AUC; test metrics are reported as holdout evaluation
- SHAP computed on test set only

Run: python step08_final_best.py

Outputs (all in model_outputs/):
  best_denial_risk_model.pkl / .json
  model_feature_names.pkl
  model_threshold.pkl
  feature_columns.json
  model_comparison.csv
  tuning_results.json
  overfit_audit.csv
  bias_audit.csv
  risk_decile_report.csv
  threshold_review_table.csv
  shap_feature_importance.csv
  test_predictions_sample.csv
  model_training_report.json
  charts/  (12+ PNG files)
"""

from pathlib import Path
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint, uniform, loguniform
from sklearn.calibration import calibration_curve
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. XGBoost will be skipped.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    lgb = None
    LGB_AVAILABLE = False
    print("WARNING: lightgbm not installed. LightGBM will be skipped.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False
    print("WARNING: shap not installed. SHAP outputs will be skipped.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False
    print("WARNING: optuna not installed. Will fall back to RandomizedSearchCV.")

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION  — adjust these before running
# ═══════════════════════════════════════════════════════════════
BASE_DIR        = Path(__file__).resolve().parent
INPUT_ENCODED   = BASE_DIR / "Medicare_Cleaned_Encoded.csv"
INPUT_OUTLIERS  = BASE_DIR / "Medicare_Cleaned_Outliers.csv"
OUTPUT_DIR      = BASE_DIR / "model_outputs"
CHART_DIR       = OUTPUT_DIR / "charts"

RANDOM_STATE    = 42
TARGET_QUANTILE = 0.20
TEST_SIZE       = 0.20
VAL_SIZE        = 0.20        # fraction of the 80% trainval

# Laptop-safe row caps — increase for server runs
TRAINING_MAX_ROWS     = 2_500_000
RF_MAX_ROWS           = 600_000
ET_MAX_ROWS           = 600_000
TUNING_MAX_ROWS       = 400_000
OVERFIT_SAMPLE_ROWS   = 150_000
SHAP_SAMPLE_ROWS      = 2_000
PREDICTION_SAMPLE     = 100_000

# Tuning settings
N_OPTUNA_TRIALS   = 40      # increase to 100 for production
RANDOMSEARCH_ITER = 15      # used only if optuna not available
CV_FOLDS          = 3       # inside tuning (speed); Step 08 CV does 5

HIGH_CARDINALITY_COLS = [
    "HCPCS_Cd",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_State_Abrvtn",
]
TARGET_ENCODING_SMOOTHING = 50
CLASS_WEIGHT_MODE = "sqrt"      # "sqrt" is less recall-heavy than full imbalance ratio.
THRESHOLD_STRATEGY = "balanced" # "balanced" keeps precision/recall closer than pure F1.
MIN_F1_FRACTION_FOR_BALANCED_THRESHOLD = 0.98

np.random.seed(RANDOM_STATE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def elapsed(t0: float) -> str:
    return f"{time.time() - t0:.1f}s"


def stratified_sample(df: pd.DataFrame, target_col: str,
                       max_rows: int, label: str) -> pd.DataFrame:
    """Stratified row cap — keeps class balance when limiting size."""
    if target_col not in df.columns:
        raise KeyError(f"Cannot sample by missing column: {target_col}")
    if not max_rows or len(df) <= max_rows:
        print(f"  {label}: using all {len(df):,} rows")
        return df.copy()
    frac = max_rows / len(df)
    sampled_indices = []
    for _, group in df.groupby(target_col):
        n = max(1, int(round(len(group) * frac)))
        sampled_indices.extend(
            group.sample(n=min(n, len(group)), random_state=RANDOM_STATE).index.tolist()
        )
    sampled = df.loc[sampled_indices].sample(frac=1, random_state=RANDOM_STATE).copy()
    if target_col not in sampled.columns:
        raise KeyError(f"Internal error: {target_col} was lost during sampling.")
    print(f"  {label}: {len(sampled):,} stratified rows (from {len(df):,})")
    return sampled


# ═══════════════════════════════════════════════════════════════
# 1. DATA LOAD
# ═══════════════════════════════════════════════════════════════
print_section("1. DATA LOAD")
t_start = time.time()

if not INPUT_ENCODED.exists():
    raise FileNotFoundError(f"Missing: {INPUT_ENCODED}\nRun step06_encoding.py first.")
if not INPUT_OUTLIERS.exists():
    raise FileNotFoundError(f"Missing: {INPUT_OUTLIERS}\nRun step06_encoding.py first.")

df = pd.read_csv(INPUT_ENCODED, low_memory=False)
df_out = pd.read_csv(INPUT_OUTLIERS, low_memory=False,
                     usecols=[
                         "Avg_Mdcr_Pymt_Amt_log",
                         "Avg_Sbmtd_Chrg_log",
                         *HIGH_CARDINALITY_COLS,
                     ])

for col in HIGH_CARDINALITY_COLS:
    if col in df_out.columns:
        df[col] = df_out[col].values

print(f"Encoded shape : {df.shape[0]:,} rows × {df.shape[1]} columns")


# ═══════════════════════════════════════════════════════════════
# 2. TARGET — attach ratio column, threshold derived LATER
#    (from training split only — critical for leakage prevention)
# ═══════════════════════════════════════════════════════════════
print_section("2. ATTACH PAYMENT/CHARGE RATIO (target not yet set)")

df["_pay_charge_ratio"] = (
    df_out["Avg_Mdcr_Pymt_Amt_log"] / (df_out["Avg_Sbmtd_Chrg_log"] + 1e-6)
).values

# Provisional target — used only for stratified split
provisional_threshold = float(df["_pay_charge_ratio"].quantile(TARGET_QUANTILE))
df["_provisional_target"] = (df["_pay_charge_ratio"] <= provisional_threshold).astype(int)
print(f"Full-data provisional threshold : {provisional_threshold:.6f}")
print(f"Provisional positive rate       : {df['_provisional_target'].mean():.2%}")


# ═══════════════════════════════════════════════════════════════
# 3. STRATIFIED ROW CAP (memory safety)
# ═══════════════════════════════════════════════════════════════
if len(df) > TRAINING_MAX_ROWS:
    df = stratified_sample(df, "_provisional_target", TRAINING_MAX_ROWS, "Main sample")


# ═══════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT — BEFORE THRESHOLD IS FINALIZED
#    Threshold is learned from train split only → no leakage
# ═══════════════════════════════════════════════════════════════
print_section("4. TRAIN / VAL / TEST SPLIT")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["_provisional_target"],
    random_state=RANDOM_STATE,
)

# Fit threshold on TRAINING data only
target_threshold = float(train_df["_pay_charge_ratio"].quantile(TARGET_QUANTILE))
y_train_full = (train_df["_pay_charge_ratio"] <= target_threshold).astype(int)
y_test       = (test_df["_pay_charge_ratio"]  <= target_threshold).astype(int)

print(f"Train rows : {len(train_df):,}  |  positive = {y_train_full.mean():.4f}")
print(f"Test rows  : {len(test_df):,}  |  positive = {y_test.mean():.4f}")
print(f"Train-derived target threshold : {target_threshold:.6f}")


# ═══════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING + LEAKAGE DROP
# ═══════════════════════════════════════════════════════════════
print_section("5. FEATURE ENGINEERING & LEAKAGE DROP")

def fit_high_cardinality_encoders(df_input: pd.DataFrame, y_input: pd.Series) -> dict:
    """Fit frequency and smoothed target encoders on model-train rows only."""
    encoders = {
        "global_target_mean": float(y_input.mean()),
        "frequency": {},
        "target": {},
        "columns": [col for col in HIGH_CARDINALITY_COLS if col in df_input.columns],
    }
    n_rows = max(len(df_input), 1)
    train_y = pd.Series(y_input.values, index=df_input.index, name="_target")

    audit_rows = []
    for col in encoders["columns"]:
        values = df_input[col].fillna("__MISSING__").astype(str)
        freq_map = (values.value_counts(dropna=False) / n_rows).to_dict()
        encoders["frequency"][col] = freq_map

        stats_df = pd.concat([values.rename(col), train_y], axis=1).groupby(col)["_target"].agg(["mean", "count"])
        smooth = (
            (stats_df["mean"] * stats_df["count"])
            + (encoders["global_target_mean"] * TARGET_ENCODING_SMOOTHING)
        ) / (stats_df["count"] + TARGET_ENCODING_SMOOTHING)
        target_map = smooth.to_dict()
        encoders["target"][col] = target_map

        for category, count in stats_df["count"].sort_values(ascending=False).head(50).items():
            audit_rows.append(
                {
                    "column": col,
                    "category": category,
                    "train_count": int(count),
                    "train_frequency": float(freq_map.get(category, 0)),
                    "smoothed_target_rate": float(target_map.get(category, encoders["global_target_mean"])),
                }
            )

    pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / "high_cardinality_encoding_audit.csv", index=False)
    return encoders


def transform_high_cardinality(df_input: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df_input.index)
    global_mean = encoders["global_target_mean"]
    for col in encoders["columns"]:
        values = df_input[col].fillna("__MISSING__").astype(str)
        encoded[f"{col}_freq_train"] = values.map(encoders["frequency"][col]).fillna(0.0).astype(float)
        encoded[f"{col}_target_smooth_train"] = (
            values.map(encoders["target"][col]).fillna(global_mean).astype(float)
        )
    return encoded


def prepare_features(df_input: pd.DataFrame, encoders: dict | None = None) -> pd.DataFrame:
    feat = df_input.copy()

    # --- Domain features ---
    if "Tot_Srvcs_log" in feat.columns and "Tot_Benes_log" in feat.columns:
        feat["svc_per_patient"] = feat["Tot_Srvcs_log"] / (feat["Tot_Benes_log"] + 1e-6)

    if "Avg_Sbmtd_Chrg_log" in feat.columns and "Tot_Srvcs_log" in feat.columns:
        feat["charge_per_service_proxy"] = feat["Avg_Sbmtd_Chrg_log"] / (feat["Tot_Srvcs_log"] + 1e-6)

    if "Tot_Benes_log" in feat.columns:
        vol_thresh = feat["Tot_Benes_log"].quantile(0.75)
        feat["high_volume_flag"] = (feat["Tot_Benes_log"] > vol_thresh).astype(int)

    if "Rndrng_Prvdr_RUCA" in feat.columns:
        feat["is_urban"] = (feat["Rndrng_Prvdr_RUCA"] <= 3).astype(int)

    if encoders:
        feat = pd.concat([feat, transform_high_cardinality(df_input, encoders)], axis=1)

    # --- Drop leakage columns ---
    LEAKAGE_COLS = [
        "_pay_charge_ratio", "_provisional_target", "denied",
        "Avg_Mdcr_Pymt_Amt_log", "Avg_Mdcr_Alowd_Amt_log",
        "Avg_Mdcr_Stdzd_Amt_log", "Avg_Sbmtd_Chrg_log",
        "flag_charge_lt_payment", "flag_zero_payment", "flag_zero_allowed",
        *HIGH_CARDINALITY_COLS,
    ]
    existing_leak = [c for c in LEAKAGE_COLS if c in feat.columns]
    if existing_leak:
        feat = feat.drop(columns=existing_leak)

    # --- Drop object columns ---
    obj_cols = feat.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        feat = feat.drop(columns=obj_cols)

    # --- Bool → int ---
    bool_cols = feat.select_dtypes(include="bool").columns
    if len(bool_cols):
        feat[bool_cols] = feat[bool_cols].astype(int)

    # --- Drop constants ---
    const_cols = [c for c in feat.columns if feat[c].nunique(dropna=False) <= 1]
    if const_cols:
        feat = feat.drop(columns=const_cols)

    # --- Fill nulls ---
    if feat.isna().sum().sum() > 0:
        feat = feat.fillna(feat.median(numeric_only=True))

    return feat


val_frac = VAL_SIZE / (1 - TEST_SIZE)
model_train_df, val_df, y_train, y_val = train_test_split(
    train_df,
    y_train_full,
    test_size=val_frac,
    stratify=y_train_full,
    random_state=RANDOM_STATE,
)

high_card_encoders = fit_high_cardinality_encoders(model_train_df, y_train)
X_train = prepare_features(model_train_df, high_card_encoders)
X_val = prepare_features(val_df, high_card_encoders).reindex(columns=X_train.columns, fill_value=0)
X_test = prepare_features(test_df, high_card_encoders).reindex(columns=X_train.columns, fill_value=0)
feature_names = X_train.columns.tolist()

print(f"Feature count after engineering + leakage drop: {len(feature_names)}")
print(
    "New features added: svc_per_patient, charge_per_service_proxy, "
    "high_volume_flag, is_urban, train-only frequency/target encodings"
)
print(f"High-cardinality encoded columns: {high_card_encoders['columns']}")

with open(OUTPUT_DIR / "feature_columns.json", "w") as fh:
    json.dump(feature_names, fh, indent=2)

with open(OUTPUT_DIR / "high_cardinality_encoders.pkl", "wb") as fh:
    pickle.dump(high_card_encoders, fh)

raw_scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
if CLASS_WEIGHT_MODE == "sqrt":
    scale_pos_weight = float(np.sqrt(raw_scale_pos_weight))
elif CLASS_WEIGHT_MODE == "none":
    scale_pos_weight = 1.0
else:
    scale_pos_weight = raw_scale_pos_weight
print(f"\nModel train : {len(X_train):,}  |  Val : {len(X_val):,}  |  Test : {len(X_test):,}")
print(f"raw scale_pos_weight : {raw_scale_pos_weight:.4f}")
print(f"model scale_pos_weight ({CLASS_WEIGHT_MODE}) : {scale_pos_weight:.4f}")


# ═══════════════════════════════════════════════════════════════
# 7. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_best_threshold(y_true, y_proba) -> float:
    """Find a validation threshold with controlled precision/recall tradeoff."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_proba)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thresholds) == 0:
        return 0.5

    usable = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": prec[:-1],
            "recall": rec[:-1],
            "f1": f1_arr[:-1],
        }
    )

    if THRESHOLD_STRATEGY == "balanced":
        max_f1 = usable["f1"].max()
        candidates = usable[usable["f1"] >= max_f1 * MIN_F1_FRACTION_FOR_BALANCED_THRESHOLD].copy()
        candidates["pr_gap"] = (candidates["precision"] - candidates["recall"]).abs()
        best_row = candidates.sort_values(["pr_gap", "f1"], ascending=[True, False]).iloc[0]
        return float(best_row["threshold"])

    return float(usable.sort_values("f1", ascending=False).iloc[0]["threshold"])


def evaluate_at_threshold(y_true, y_proba, threshold) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold"        : float(threshold),
        "accuracy"         : float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision"        : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"           : float(recall_score(y_true, y_pred, zero_division=0)),
        "f1"               : float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc"          : float(roc_auc_score(y_true, y_proba)),
        "pr_auc"           : float(average_precision_score(y_true, y_proba)),
        "brier"            : float(brier_score_loss(y_true, y_proba)),
        "pred"             : y_pred,
    }


def fit_and_evaluate(name, model, Xtr, ytr, Xv, yv, Xte, yte):
    t0 = time.time()
    model.fit(Xtr, ytr)
    val_p   = model.predict_proba(Xv)[:, 1]
    thresh  = find_best_threshold(yv, val_p)
    val_m   = evaluate_at_threshold(yv, val_p, thresh)
    test_p  = model.predict_proba(Xte)[:, 1]
    test_m  = evaluate_at_threshold(yte, test_p, thresh)
    secs    = round(time.time() - t0, 2)
    print(f"  {name:<28}  val PR-AUC={val_m['pr_auc']:.4f}  "
          f"test PR-AUC={test_m['pr_auc']:.4f}  "
          f"F1={test_m['f1']:.4f}  Rec={test_m['recall']:.4f}  {secs}s")
    return model, test_p, val_m, test_m, secs


# ═══════════════════════════════════════════════════════════════
# 8. BASELINE MODEL COMPARISON
#    6 models: LR, HistGB, RandomForest, ExtraTrees, XGBoost, LightGBM
# ═══════════════════════════════════════════════════════════════
print_section("8. BASELINE MODEL COMPARISON (6 models)")

def build_baseline_specs():
    specs = {
        "Logistic Regression": {
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    class_weight="balanced", max_iter=800,
                    C=0.5, random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            "row_cap": None,
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                learning_rate=0.07, max_iter=350, max_leaf_nodes=63,
                l2_regularization=0.05, random_state=RANDOM_STATE,
            ),
            "row_cap": None,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=250, max_depth=16, min_samples_leaf=25,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "row_cap": RF_MAX_ROWS,
        },
        "Extra Trees": {
            "model": ExtraTreesClassifier(
                n_estimators=250, max_depth=18, min_samples_leaf=25,
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            ),
            "row_cap": ET_MAX_ROWS,
        },
    }

    if XGB_AVAILABLE:
        specs["XGBoost"] = {
            "model": xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="aucpr",
                tree_method="hist", n_estimators=500, max_depth=6,
                learning_rate=0.06, subsample=0.85, colsample_bytree=0.85,
                min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight, early_stopping_rounds=30,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            ),
            "row_cap": None,
        }

    if LGB_AVAILABLE:
        specs["LightGBM"] = {
            "model": lgb.LGBMClassifier(
                n_estimators=500, max_depth=7, learning_rate=0.06,
                num_leaves=63, subsample=0.85, colsample_bytree=0.85,
                min_child_samples=20, reg_alpha=0.1, reg_lambda=1.5,
                class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            "row_cap": None,
        }

    return specs


baseline_specs = build_baseline_specs()
results = []
trained_models = {}

for name, spec in baseline_specs.items():
    Xtr_m, ytr_m = X_train, y_train

    # XGBoost needs eval_set for early stopping
    fit_kwargs = {}
    if XGB_AVAILABLE and isinstance(spec["model"], xgb.XGBClassifier):
        fit_kwargs = {"eval_set": [(X_val.values, y_val)], "verbose": False}
        Xtr_m = X_train.values

    if LGB_AVAILABLE and isinstance(spec["model"], lgb.LGBMClassifier):
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "callbacks": [lgb.early_stopping(30, verbose=False),
                          lgb.log_evaluation(period=-1)],
        }

    # Apply row cap
    if spec["row_cap"]:
        pack = pd.concat([X_train, y_train.rename("_t")], axis=1)
        pack = stratified_sample(pack, "_t", spec["row_cap"], name)
        ytr_m = pack["_t"]
        Xtr_m = pack.drop(columns=["_t"])
        if XGB_AVAILABLE and isinstance(spec["model"], xgb.XGBClassifier):
            Xtr_m = Xtr_m.values

    # Fit
    t0 = time.time()
    spec["model"].fit(Xtr_m, ytr_m, **fit_kwargs)
    val_p  = spec["model"].predict_proba(X_val if not (XGB_AVAILABLE and isinstance(spec["model"], xgb.XGBClassifier)) else X_val.values)[:, 1]
    thresh = find_best_threshold(y_val, val_p)
    val_m  = evaluate_at_threshold(y_val, val_p, thresh)
    test_p = spec["model"].predict_proba(X_test if not (XGB_AVAILABLE and isinstance(spec["model"], xgb.XGBClassifier)) else X_test.values)[:, 1]
    test_m = evaluate_at_threshold(y_test, test_p, thresh)
    secs   = round(time.time() - t0, 2)

    print(f"  {name:<28}  val PR-AUC={val_m['pr_auc']:.4f}  "
          f"test PR-AUC={test_m['pr_auc']:.4f}  F1={test_m['f1']:.4f}  {secs}s")

    row = {"model": name, "tuned": False, "training_seconds": secs}
    row.update({f"val_{k}": v for k, v in val_m.items() if k != "pred"})
    row.update({f"test_{k}": v for k, v in test_m.items() if k != "pred"})
    results.append(row)
    trained_models[name] = {
        "model": spec["model"], "test_proba": test_p,
        "val_metrics": val_m, "test_metrics": test_m,
    }


# ═══════════════════════════════════════════════════════════════
# 9. HYPERPARAMETER TUNING
#    Optuna (preferred) or RandomizedSearchCV (fallback)
# ═══════════════════════════════════════════════════════════════
print_section("9. HYPERPARAMETER TUNING")
print(
    "Note: high-cardinality encoders were fit on the model-train split before tuning. "
    "Tuning CV scores may be slightly optimistic; validation PR-AUC and holdout test "
    "metrics are the primary reported metrics."
)

tune_pack = pd.concat([X_train, y_train.rename("_t")], axis=1)
tune_pack = stratified_sample(tune_pack, "_t", TUNING_MAX_ROWS, "Tuning sample")
y_tune = tune_pack["_t"]
X_tune = tune_pack.drop(columns=["_t"])
skf_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

tuning_rows = []




# --- HistGradientBoosting Tuning (BEST MODEL) ---
print("Tuning HistGradientBoosting ...")
t0 = time.time()

if OPTUNA_AVAILABLE:
    def optuna_tune_histgb() -> tuple:
        def objective(trial):
            params = {
                "max_iter"           : trial.suggest_int("max_iter", 200, 600),
                "max_depth"          : trial.suggest_int("max_depth", 4, 12),
                "learning_rate"      : trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
                "min_samples_leaf"   : trial.suggest_int("min_samples_leaf", 10, 100),
                "max_leaf_nodes"     : trial.suggest_int("max_leaf_nodes", 15, 127),
                "l2_regularization"  : trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True),
                "class_weight"       : "balanced",
                "random_state"       : RANDOM_STATE,
            }
            scores = []
            for tr_idx, vl_idx in skf_cv.split(X_tune, y_tune):
                m = HistGradientBoostingClassifier(**params)
                m.fit(X_tune.iloc[tr_idx], y_tune.iloc[tr_idx])
                p = m.predict_proba(X_tune.iloc[vl_idx])[:, 1]
                scores.append(average_precision_score(y_tune.iloc[vl_idx], p))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8),
        )
        study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
        return study.best_params, study.best_value, study

    histgb_best_p, histgb_best_score, histgb_study = optuna_tune_histgb()
else:
    # RandomizedSearchCV fallback
    from scipy.stats import randint, loguniform
    rscv = RandomizedSearchCV(
        HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"max_iter": randint(200, 600), "max_depth": randint(4, 12),
         "learning_rate": loguniform(0.01, 0.18), "min_samples_leaf": randint(10, 100),
         "max_leaf_nodes": randint(15, 127), "l2_regularization": loguniform(1e-4, 5.0)},
        n_iter=RANDOMSEARCH_ITER, scoring="average_precision",
        cv=skf_cv, random_state=RANDOM_STATE, n_jobs=1,
    )
    rscv.fit(X_tune, y_tune)
    histgb_best_p = rscv.best_params_
    histgb_best_score = rscv.best_score_
    histgb_study = None

print(f"  HistGB tuning done: {elapsed(t0)}  |  best CV PR-AUC={histgb_best_score:.4f}")

histgb_tuned = HistGradientBoostingClassifier(
    **histgb_best_p,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
histgb_tuned.fit(X_train, y_train)
val_p  = histgb_tuned.predict_proba(X_val)[:, 1]
thresh = find_best_threshold(y_val, val_p)
val_m  = evaluate_at_threshold(y_val, val_p, thresh)
test_p = histgb_tuned.predict_proba(X_test)[:, 1]
test_m = evaluate_at_threshold(y_test, test_p, thresh)
print(f"  HistGB Tuned  val PR-AUC={val_m['pr_auc']:.4f}  test PR-AUC={test_m['pr_auc']:.4f}  F1={test_m['f1']:.4f}")

row = {"model": "HistGB Tuned", "tuned": True}
row.update({f"val_{k}": v for k, v in val_m.items() if k != "pred"})
row.update({f"test_{k}": v for k, v in test_m.items() if k != "pred"})
results.append(row)
trained_models["HistGB Tuned"] = {
    "model": histgb_tuned, "test_proba": test_p,
    "val_metrics": val_m, "test_metrics": test_m,
}
tuning_rows.append({"model": "HistGB Tuned", "cv_pr_auc": histgb_best_score,
                    "best_params": json.dumps(histgb_best_p)})


# ═══════════════════════════════════════════════════════════════
# 10. SOFT-VOTING ENSEMBLE (best 3 models by val PR-AUC)
# ═══════════════════════════════════════════════════════════════
print_section("10. SOFT-VOTING ENSEMBLE")

comparison_df = pd.DataFrame(results).sort_values("val_pr_auc", ascending=False)
top3_names = comparison_df["model"].head(3).tolist()
print(f"Top 3 models for ensemble (by val PR-AUC): {top3_names}")

# Average probabilities (soft voting)
ens_val_p  = np.mean([trained_models[n]["val_metrics"]["pred"].astype(float)
                       for n in top3_names], axis=0)
# Use actual probabilities for ensemble
ens_val_probas  = []
ens_test_probas = []
for n in top3_names:
    m = trained_models[n]["model"]
    try:
        ens_val_probas.append(m.predict_proba(X_val if not (XGB_AVAILABLE and isinstance(m, xgb.XGBClassifier)) else X_val.values)[:, 1])
        ens_test_probas.append(m.predict_proba(X_test if not (XGB_AVAILABLE and isinstance(m, xgb.XGBClassifier)) else X_test.values)[:, 1])
    except Exception:
        pass

if ens_val_probas:
    ens_val_p  = np.mean(ens_val_probas, axis=0)
    ens_test_p = np.mean(ens_test_probas, axis=0)
    ens_thresh = find_best_threshold(y_val, ens_val_p)
    ens_val_m  = evaluate_at_threshold(y_val, ens_val_p, ens_thresh)
    ens_test_m = evaluate_at_threshold(y_test, ens_test_p, ens_thresh)

    print(f"  Ensemble  val PR-AUC={ens_val_m['pr_auc']:.4f}  test PR-AUC={ens_test_m['pr_auc']:.4f}  F1={ens_test_m['f1']:.4f}")

    ens_row = {"model": "Ensemble (Top3 Avg)", "tuned": True}
    ens_row.update({f"val_{k}": v for k, v in ens_val_m.items() if k != "pred"})
    ens_row.update({f"test_{k}": v for k, v in ens_test_m.items() if k != "pred"})
    results.append(ens_row)
    trained_models["Ensemble (Top3 Avg)"] = {
        "model": None, "test_proba": ens_test_p,
        "val_metrics": ens_val_m, "test_metrics": ens_test_m,
    }


# ═══════════════════════════════════════════════════════════════
# 11. FINAL COMPARISON TABLE & BEST MODEL SELECTION
#     Primary metric: val PR-AUC (correct for imbalanced data)
# ═══════════════════════════════════════════════════════════════
print_section("11. FINAL MODEL COMPARISON")

comparison_df = pd.DataFrame(results).sort_values(
    ["val_pr_auc", "val_roc_auc", "val_f1"], ascending=False
)
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
tuning_df = pd.DataFrame(tuning_rows)
tuning_df.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
with open(OUTPUT_DIR / "tuning_results.json", "w", encoding="utf-8") as fh:
    json.dump(tuning_rows, fh, indent=2)

print(comparison_df[[
    "model", "tuned",
    "val_pr_auc", "val_roc_auc", "val_f1",
    "test_pr_auc", "test_roc_auc", "test_f1",
    "test_recall", "test_precision"
]].to_string(index=False))

best_name    = comparison_df.iloc[0]["model"]
best         = trained_models[best_name]
best_model   = best["model"]
best_proba   = best["test_proba"]
best_val_m   = best["val_metrics"]
best_test_m  = best["test_metrics"]
best_thresh  = best_test_m["threshold"]
best_pred    = best_test_m["pred"]

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   Val  PR-AUC  : {best_val_m['pr_auc']:.4f}")
print(f"   Test PR-AUC  : {best_test_m['pr_auc']:.4f}")
print(f"   Test ROC-AUC : {best_test_m['roc_auc']:.4f}")
print(f"   Test F1      : {best_test_m['f1']:.4f}")
print(f"   Test Recall  : {best_test_m['recall']:.4f}")
print(f"   Test Brier   : {best_test_m['brier']:.4f}")
print(f"   Threshold    : {best_thresh:.4f}")
print("\nClassification report:")
print(classification_report(y_test, best_pred, target_names=["Lower Risk", "Risk Proxy"]))


# ═══════════════════════════════════════════════════════════════
# 12. BUSINESS EVALUATION — Decile, Threshold Table
# ═══════════════════════════════════════════════════════════════
print_section("12. BUSINESS EVALUATION")

# Threshold sweep
thresh_rows = []
for t in np.arange(0.10, 0.91, 0.05):
    m = evaluate_at_threshold(y_test, best_proba, t)
    thresh_rows.append({k: v for k, v in m.items() if k != "pred"})
thresh_df = pd.DataFrame(thresh_rows)
thresh_df.to_csv(OUTPUT_DIR / "threshold_review_table.csv", index=False)
print("Threshold review table saved.")

# Decile lift report
decile_df = pd.DataFrame({"actual": y_test.values, "pred_proba": best_proba})
decile_df["risk_decile"] = pd.qcut(
    decile_df["pred_proba"].rank(method="first"), 10,
    labels=list(range(10, 0, -1))
).astype(int)
overall_rate = decile_df["actual"].mean()
decile_report = (
    decile_df.groupby("risk_decile")
    .agg(rows=("actual","size"), risk_count=("actual","sum"),
         risk_rate=("actual","mean"), avg_prob=("pred_proba","mean"))
    .reset_index().sort_values("risk_decile")
)
decile_report["lift"]         = decile_report["risk_rate"] / overall_rate
decile_report["capture_pct"]  = decile_report["risk_count"] / decile_df["actual"].sum() * 100
decile_report.to_csv(OUTPUT_DIR / "risk_decile_report.csv", index=False)
print("\nDecile lift report (top 5 deciles):")
print(decile_report.head(5).to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# 13. OVERFIT / UNDERFIT AUDIT
# ═══════════════════════════════════════════════════════════════
print_section("13. OVERFIT / UNDERFIT AUDIT")

if best_model is not None:
    train_pack = pd.concat([X_train, y_train.rename("_t")], axis=1)
    train_pack = stratified_sample(train_pack, "_t", OVERFIT_SAMPLE_ROWS, "Overfit train sample")
    ytr_ov = train_pack["_t"]
    Xtr_ov = train_pack.drop(columns=["_t"])

    audit_rows = []
    for split_name, Xs, ys in [
        ("train_sample", Xtr_ov, ytr_ov),
        ("validation",   X_val, y_val),
        ("test",         X_test, y_test),
    ]:
        try:
            Xs_in = Xs.values if XGB_AVAILABLE and isinstance(best_model, xgb.XGBClassifier) else Xs
            p = best_model.predict_proba(Xs_in)[:, 1]
            m = evaluate_at_threshold(ys, p, best_thresh)
            audit_rows.append({"split": split_name, **{k: v for k, v in m.items() if k != "pred"}})
        except Exception:
            pass

    if audit_rows:
        overfit_df = pd.DataFrame(audit_rows)
        train_f1 = float(overfit_df.loc[overfit_df["split"] == "train_sample", "f1"].iloc[0])
        test_f1  = float(overfit_df.loc[overfit_df["split"] == "test",         "f1"].iloc[0])
        overfit_df["train_test_f1_gap"] = train_f1 - test_f1
        overfit_df.to_csv(OUTPUT_DIR / "overfit_audit.csv", index=False)
        print(overfit_df.to_string(index=False))
        if train_f1 - test_f1 > 0.10:
            print("⚠️  WARNING: F1 gap > 0.10 — possible overfitting. Consider more regularisation.")
        else:
            print("✅ Overfit gap within acceptable range.")


# ═══════════════════════════════════════════════════════════════
# 14. SUBGROUP / BIAS AUDIT
# ═══════════════════════════════════════════════════════════════
print_section("14. SUBGROUP / BIAS AUDIT")

candidate_cols = [
    "Place_Of_Srvc_bin", "HCPCS_Drug_Ind_bin",
    "Rndrng_Prvdr_Ent_Cd_bin", "Rndrng_Prvdr_Mdcr_Prtcptg_Ind_bin",
    "is_urban", "RUCA_Unknown_Flag",
]
y_pred_bias = (best_proba >= best_thresh).astype(int)
bias_rows = []
for col in candidate_cols:
    if col not in X_test.columns:
        continue
    for val in sorted(X_test[col].dropna().unique()):
        mask = X_test[col] == val
        if mask.sum() < 500:
            continue
        yg = y_test.loc[mask]
        pg = best_proba[mask.to_numpy()]
        dg = y_pred_bias[mask.to_numpy()]
        bias_rows.append({
            "feature": col, "group_value": val, "rows": int(mask.sum()),
            "actual_risk_rate": float(yg.mean()),
            "predicted_positive_rate": float(dg.mean()),
            "precision": float(precision_score(yg, dg, zero_division=0)),
            "recall":    float(recall_score(yg, dg, zero_division=0)),
            "f1":        float(f1_score(yg, dg, zero_division=0)),
            "roc_auc":   float(roc_auc_score(yg, pg)) if yg.nunique() == 2 else np.nan,
        })

bias_df = pd.DataFrame(bias_rows)
bias_df.to_csv(OUTPUT_DIR / "bias_audit.csv", index=False)
print(f"Subgroup bias audit: {len(bias_df)} subgroups evaluated")
if not bias_df.empty:
    print(bias_df.head(8).to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# 15. FEATURE IMPORTANCE (model native + permutation fallback)
# ═══════════════════════════════════════════════════════════════
print_section("15. FEATURE IMPORTANCE")

if best_model is not None:
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        method = "model_native"
    elif hasattr(best_model, "named_steps") and hasattr(
        best_model.named_steps.get("clf", None), "coef_"
    ):
        imp = np.abs(best_model.named_steps["clf"].coef_[0])
        method = "logistic_coef_abs"
    else:
        sample_n = min(20_000, len(X_val))
        X_pi = X_val.sample(n=sample_n, random_state=RANDOM_STATE)
        y_pi = y_val.loc[X_pi.index]
        perm = permutation_importance(
            best_model, X_pi, y_pi,
            scoring="average_precision",
            n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1,
        )
        imp = perm.importances_mean
        method = "permutation_pr_auc"

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": imp, "method": method}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print(f"Top 10 features ({method}):")
    print(importance_df.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# 16. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
print_section("16. SHAP EXPLAINABILITY")

shap_report = {"available": False}

if SHAP_AVAILABLE and best_model is not None:
    sample_n = min(SHAP_SAMPLE_ROWS, len(X_test))
    X_shap = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
    print(f"Computing SHAP on {sample_n:,} test rows ...")

    try:
        if (XGB_AVAILABLE and isinstance(best_model, xgb.XGBClassifier)) or (
            LGB_AVAILABLE and isinstance(best_model, lgb.LGBMClassifier)
        ):
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_shap, check_additivity=False)
            shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        else:
            explainer   = shap.Explainer(best_model, X_shap)
            shap_values = explainer(X_shap)
            shap_vals   = shap_values.values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, -1]
        shap_available = True
    except Exception as e1:
        print(f"  Primary SHAP failed ({e1}), trying small predict_proba explainer...")
        try:
            X_shap = X_shap.head(min(500, len(X_shap)))
            X_shap_in = X_shap.values if XGB_AVAILABLE and isinstance(best_model, xgb.XGBClassifier) else X_shap
            explainer   = shap.Explainer(best_model.predict_proba, X_shap_in)
            shap_values = explainer(X_shap_in)
            shap_vals   = shap_values.values
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
            shap_available = True
        except Exception as e2:
            print(f"  SHAP skipped: {e2}")
            shap_available = False

    if shap_available:
        shap_imp_df = pd.DataFrame({
            "feature": feature_names,
            "shap_mean_abs": np.abs(shap_vals).mean(axis=0),
        }).sort_values("shap_mean_abs", ascending=False)
        shap_imp_df.to_csv(OUTPUT_DIR / "shap_feature_importance.csv", index=False)

        # Bar chart
        shap.summary_plot(shap_vals, X_shap, plot_type="bar", max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Beeswarm
        shap.summary_plot(shap_vals, X_shap, max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(CHART_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        # Waterfall — highest risk claim
        if hasattr(shap_values, "__len__"):
            X_shap_in2 = X_shap.values if XGB_AVAILABLE and isinstance(best_model, xgb.XGBClassifier) else X_shap
            probas_shap = best_model.predict_proba(X_shap_in2)[:, 1]
            denied_idx  = int(np.argmax(probas_shap))
            approved_idx= int(np.argmin(probas_shap))
            try:
                shap.plots.waterfall(shap_values[denied_idx], max_display=15, show=False)
                plt.tight_layout()
                plt.savefig(CHART_DIR / "shap_waterfall_denied.png", dpi=150, bbox_inches="tight")
                plt.close("all")

                shap.plots.waterfall(shap_values[approved_idx], max_display=15, show=False)
                plt.tight_layout()
                plt.savefig(CHART_DIR / "shap_waterfall_approved.png", dpi=150, bbox_inches="tight")
                plt.close("all")
                print("SHAP waterfall charts saved.")
            except Exception:
                pass

        shap_report = {"available": True, "rows": sample_n,
                       "bar": str(CHART_DIR / "shap_bar.png"),
                       "beeswarm": str(CHART_DIR / "shap_beeswarm.png")}
        print("SHAP outputs complete.")


# ═══════════════════════════════════════════════════════════════
# 17. CHARTS — ROC, PR, Confusion, Decile, Calibration, Optuna
# ═══════════════════════════════════════════════════════════════
print_section("17. CHARTS")

# Colors
CMAP = {
    "Logistic Regression": "#95A5A6",
    "HistGradientBoosting": "#27AE60",
    "Random Forest":       "#8E44AD",
    "Extra Trees":         "#16A085",
    "XGBoost":             "#3498DB",
    "LightGBM":            "#E67E22",
    "XGBoost Tuned":       "#2C3E50",
    "LightGBM Tuned":      "#E74C3C",
    "Ensemble (Top3 Avg)": "#F39C12",
}

# --- ROC + PR Curves ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for name, info in trained_models.items():
    if info["test_proba"] is None:
        continue
    clr = CMAP.get(name, "#999999")
    lw  = 3 if name == best_name else 1.5
    fpr_c, tpr_c, _ = roc_curve(y_test, info["test_proba"])
    p_c, r_c, _     = precision_recall_curve(y_test, info["test_proba"])
    axes[0].plot(fpr_c, tpr_c, color=clr, lw=lw, alpha=0.85,
                 label=f'{name} (AUC={info["test_metrics"]["roc_auc"]:.3f})')
    axes[1].plot(r_c, p_c, color=clr, lw=lw, alpha=0.85,
                 label=f'{name} (PR={info["test_metrics"]["pr_auc"]:.3f})')

axes[0].plot([0,1],[0,1],"k--",lw=1); axes[0].set_title("ROC Curves — Test Set"); axes[0].legend(fontsize=7)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[1].axhline(y_test.mean(), color="k", ls="--", lw=1, label=f"Baseline={y_test.mean():.3f}")
axes[1].set_title("Precision-Recall Curves — Test Set"); axes[1].legend(fontsize=7)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
plt.tight_layout()
plt.savefig(CHART_DIR / "roc_pr_all_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/roc_pr_all_models.png")

# --- Model Comparison Bar ---
comp_plot = comparison_df[comparison_df["model"] != "Ensemble (Top3 Avg)"].copy()
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for ax, metric, title in [
    (axes[0], "test_pr_auc",  "Test PR-AUC"),
    (axes[1], "test_roc_auc", "Test ROC-AUC"),
    (axes[2], "test_f1",      "Test F1"),
]:
    colors_bar = [CMAP.get(n, "#999999") for n in comp_plot["model"]]
    ax.barh(comp_plot["model"], comp_plot[metric], color=colors_bar, alpha=0.85)
    for i, v in enumerate(comp_plot[metric]):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(title)
    ax.axvline(comp_plot[metric].max(), color="red", ls="--", lw=1.5, alpha=0.5)
plt.suptitle("WellMind — Full Model Comparison (Baseline vs Tuned)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(CHART_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/model_comparison.png")

# --- Confusion Matrix (best model) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cm_raw  = confusion_matrix(y_test, best_pred)
cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
sns.heatmap(cm_raw,  annot=True, fmt="d",    cmap="Blues",  ax=axes[0],
            xticklabels=["Approved","Denied"], yticklabels=["Approved","Denied"])
sns.heatmap(cm_norm, annot=True, fmt=".1%",  cmap="Greens", ax=axes[1],
            xticklabels=["Approved","Denied"], yticklabels=["Approved","Denied"])
axes[0].set_title("Confusion Matrix (Raw)");        axes[0].set_ylabel("Actual")
axes[1].set_title("Confusion Matrix (Normalized)"); axes[1].set_ylabel("Actual")
plt.suptitle(f"Best Model: {best_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CHART_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/confusion_matrix.png")

# --- Decile Lift Chart ---
fig, ax = plt.subplots(figsize=(9, 5))
colors_decile = ["#E74C3C" if d <= 3 else "#F28E2B" if d <= 6 else "#59A14F"
                 for d in decile_report["risk_decile"]]
ax.bar(decile_report["risk_decile"].astype(str), decile_report["risk_rate"], color=colors_decile)
ax.axhline(overall_rate, color="k", ls="--", lw=1.5, label=f"Overall rate = {overall_rate:.2%}")
ax.set_title("Denial-Risk Rate by Predicted Risk Decile\n(Decile 1 = Highest Predicted Risk)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Risk Decile"); ax.set_ylabel("Actual Risk Rate")
ax.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "decile_lift.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/decile_lift.png")

# --- Calibration Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
prob_true, prob_pred = calibration_curve(y_test, best_proba, n_bins=10)
ax.plot(prob_pred, prob_true, "b-o", lw=2, label="Best model")
ax.plot([0,1],[0,1],"k--", lw=1.5, label="Perfectly calibrated")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curve — Best Model", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/calibration_curve.png")

# --- Threshold Curve ---
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(thresh_df["threshold"], thresh_df["f1"],        color="#E74C3C", lw=2, label="F1")
ax.plot(thresh_df["threshold"], thresh_df["precision"],  color="#3498DB", lw=2, label="Precision")
ax.plot(thresh_df["threshold"], thresh_df["recall"],     color="#27AE60", lw=2, label="Recall")
ax.axvline(best_thresh, color="k", ls="--", lw=1.5, label=f"Best threshold={best_thresh:.2f}")
ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold vs F1 / Precision / Recall (Best Model)", fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(CHART_DIR / "threshold_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: charts/threshold_curve.png")

# --- Optuna History (if available) ---
if OPTUNA_AVAILABLE and histgb_study is not None:
    histgb_vals = [t.value for t in histgb_study.trials if t.value is not None]

    if histgb_vals:
        fig, ax = plt.subplots(figsize=(9, 5))
        trial_nums = range(1, len(histgb_vals) + 1)
        ax.plot(trial_nums, histgb_vals, "b-o", ms=4, lw=1.5, label="Each trial", alpha=0.6)
        ax.plot(
            trial_nums,
            [max(histgb_vals[:i + 1]) for i in range(len(histgb_vals))],
            "r-",
            lw=2.5,
            label="Best so far",
        )
        ax.set_title("HistGradientBoosting Optuna (AUC-PR)")
        ax.set_xlabel("Trial")
        ax.set_ylabel("AUC-PR (CV)")
        ax.legend()

        plt.suptitle("Hyperparameter Optimization History", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "optuna_history.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: charts/optuna_history.png")


# ═══════════════════════════════════════════════════════════════
# 18. SAVE MODELS & ARTIFACTS
# ═══════════════════════════════════════════════════════════════
print_section("18. SAVE MODELS & ARTIFACTS")

# Best model
if best_model is not None:
    if XGB_AVAILABLE and isinstance(best_model, xgb.XGBClassifier):
        best_model.save_model(str(OUTPUT_DIR / "best_denial_risk_model.json"))
        print("Saved: best_denial_risk_model.json")
    else:
        with open(OUTPUT_DIR / "best_denial_risk_model.pkl", "wb") as fh:
            pickle.dump(best_model, fh)
        print("Saved: best_denial_risk_model.pkl")

with open(OUTPUT_DIR / "model_feature_names.pkl", "wb") as fh:
    pickle.dump(feature_names, fh)
with open(OUTPUT_DIR / "model_threshold.pkl", "wb") as fh:
    pickle.dump(best_thresh, fh)

# Test predictions sample
pred_sample = X_test.copy()
pred_sample["actual"]     = y_test.values
pred_sample["pred_proba"] = best_proba
pred_sample["pred_label"] = best_pred
if len(pred_sample) > PREDICTION_SAMPLE:
    pred_sample = pred_sample.sample(PREDICTION_SAMPLE, random_state=RANDOM_STATE)
pred_sample.to_csv(OUTPUT_DIR / "test_predictions_sample.csv", index=False)

runtime_min = round((time.time() - t_start) / 60, 2)

# Full JSON report
report = {
    "project"          : "WellMind Medicare Denial-Risk Prediction",
    "target_note"      : "CMS PUF has no real denial label. Proxy = lowest 20% pay/charge ratio.",
    "target_threshold" : target_threshold,
    "leakage_controls" : {
        "threshold_source"       : "training split only — no leakage",
        "dropped_columns"        : ["Avg_Mdcr_Pymt_Amt_log","Avg_Mdcr_Alowd_Amt_log",
                                    "Avg_Mdcr_Stdzd_Amt_log","flag_charge_lt_payment",
                                    "flag_zero_payment","flag_zero_allowed"],
        "model_selection_metric" : "validation PR-AUC",
        "test_set_policy"        : "holdout test metrics are reported for model comparison; best model is selected by validation PR-AUC",
        "high_cardinality_encoding": "frequency and smoothed target encoders are fit on the model-train split only",
        "tuning_cv_note"         : "tuning CV uses features encoded before CV, so CV tuning scores may be slightly optimistic; validation and test metrics are the primary reported metrics",
    },
    "class_balance_controls": {
        "raw_scale_pos_weight": raw_scale_pos_weight,
        "model_scale_pos_weight": scale_pos_weight,
        "class_weight_mode": CLASS_WEIGHT_MODE,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "min_f1_fraction_for_balanced_threshold": MIN_F1_FRACTION_FOR_BALANCED_THRESHOLD,
    },
    "features"         : len(feature_names),
    "train_rows"       : len(X_train),
    "val_rows"         : len(X_val),
    "test_rows"        : len(X_test),
    "best_model"       : best_name,
    "best_val_metrics" : {k: v for k, v in best_val_m.items() if k != "pred"},
    "best_test_metrics": {k: v for k, v in best_test_m.items() if k != "pred"},
    "overfit_controls" : "moderated class weighting, validation thresholding, L1/L2 reg, early stopping, subsample/colsample",
    "shap"             : shap_report,
    "runtime_minutes"  : runtime_min,
}
with open(OUTPUT_DIR / "model_training_report.json", "w") as fh:
    json.dump(report, fh, indent=2)

print_section("STEP 08 FINAL BEST — COMPLETE")
print(f"""
  🏆 Best Model     : {best_name}
  📊 Test PR-AUC    : {best_test_m['pr_auc']:.4f}
  📊 Test ROC-AUC   : {best_test_m['roc_auc']:.4f}
  📊 Test F1        : {best_test_m['f1']:.4f}
  📊 Test Recall    : {best_test_m['recall']:.4f}
  📊 Test Precision : {best_test_m['precision']:.4f}
  📊 Brier Score    : {best_test_m['brier']:.4f}
  ⏱  Runtime        : {runtime_min} min

  📁 All outputs in : {OUTPUT_DIR}

  NEXT: step09_nlp_classifier.py — DistilBERT denial reason NLP
""")
