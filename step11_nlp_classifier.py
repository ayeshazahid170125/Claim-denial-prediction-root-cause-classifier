"""
STEP 11 - NLP Root-Cause Classifier
WellMind Data Solutions - Claim Denial Prediction System

Run:
    python step11_nlp_classifier.py

Purpose:
Train denial reason root-cause classifiers from the synthetic RARC-style
dataset generated in Step 10.

This script always trains a strong TF-IDF + Logistic Regression baseline.
If torch + transformers are installed and the DistilBERT model is available,
it also fine-tunes DistilBERT and compares results.

Inputs:
    nlp_outputs/rarc_train.csv
    nlp_outputs/rarc_val.csv
    nlp_outputs/rarc_test.csv
    nlp_outputs/rarc_root_cause_taxonomy.csv

Outputs:
    nlp_outputs/models/tfidf_root_cause_classifier.pkl
    nlp_outputs/models/distilbert_root_cause_classifier/
    nlp_outputs/nlp_model_comparison.csv
    nlp_outputs/nlp_test_predictions.csv
    nlp_outputs/nlp_training_history.csv
    nlp_outputs/nlp_model_card.json
    nlp_outputs/nlp_charts/
"""

from pathlib import Path
import json
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
    TORCH_TRANSFORMERS_AVAILABLE = True
except Exception:
    torch = None
    Dataset = object
    TORCH_TRANSFORMERS_AVAILABLE = False

warnings.filterwarnings("ignore")


IS_KAGGLE = Path("/kaggle/input").exists()

# Works in .py scripts, notebooks, and Kaggle.
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

if IS_KAGGLE:
    INPUT_ROOT = Path("/kaggle/input")
    NLP_DIR = Path("/kaggle/working/nlp_outputs")
else:
    INPUT_ROOT = BASE_DIR / "nlp_outputs"
    NLP_DIR = BASE_DIR / "nlp_outputs"

MODEL_DIR = NLP_DIR / "models"
CHART_DIR = NLP_DIR / "nlp_charts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

def find_input_file(filename):
    """Find an input file locally or inside Kaggle's mounted dataset folders."""
    direct_path = INPUT_ROOT / filename
    if direct_path.exists():
        return direct_path

    if INPUT_ROOT.exists():
        matches = list(INPUT_ROOT.rglob(filename))
        if matches:
            return matches[0]

    # Fallback for local runs where files are under nlp_outputs.
    return BASE_DIR / "nlp_outputs" / filename


TRAIN_PATH = find_input_file("rarc_train.csv")
VAL_PATH = find_input_file("rarc_val.csv")
TEST_PATH = find_input_file("rarc_test.csv")
TAXONOMY_PATH = find_input_file("rarc_root_cause_taxonomy.csv")

TFIDF_MODEL_PATH = MODEL_DIR / "tfidf_root_cause_classifier.pkl"
DISTILBERT_MODEL_PATH = MODEL_DIR / "distilbert_root_cause_classifier"
MODEL_COMPARISON_PATH = NLP_DIR / "nlp_model_comparison.csv"
TEST_PREDICTIONS_PATH = NLP_DIR / "nlp_test_predictions.csv"
TRAINING_HISTORY_PATH = NLP_DIR / "nlp_training_history.csv"
MODEL_CARD_PATH = NLP_DIR / "nlp_model_card.json"

RANDOM_STATE = 42
DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
MAX_EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
PERFECT_SCORE_WARNING_THRESHOLD = 0.995

# Keep local CPU runs practical. On GPU, full data is used by default.
CPU_TRAIN_LIMIT = 12_000
CPU_VAL_LIMIT = 3_000
CPU_TEST_LIMIT = 3_000

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


def print_section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def require_file(path, step_hint):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. {step_hint}")


def normalize_input_frame(df):
    """Support both old notebook columns and current Step 10 columns."""
    df = df.copy()
    if "label_id" not in df.columns and "label" in df.columns:
        # Old data sometimes used numeric label in column named label.
        if pd.api.types.is_numeric_dtype(df["label"]):
            df["label_id"] = df["label"].astype(int)
    if "label_name" not in df.columns:
        if "label" in df.columns and not pd.api.types.is_numeric_dtype(df["label"]):
            df["label_name"] = df["label"].astype(str)
        elif "category" in df.columns:
            df["label_name"] = df["category"].astype(str)
    if "label_id" not in df.columns:
        raise KeyError("Expected label_id column or numeric label column.")
    if "label_name" not in df.columns:
        df["label_name"] = df["label_id"].astype(str)
    if "text" not in df.columns:
        raise KeyError("Expected text column.")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    df["label_id"] = df["label_id"].astype(int)
    return df


def load_data():
    require_file(TRAIN_PATH, "Run step10_synthetic_rarc_data.py first.")
    require_file(VAL_PATH, "Run step10_synthetic_rarc_data.py first.")
    require_file(TEST_PATH, "Run step10_synthetic_rarc_data.py first.")

    train_df = normalize_input_frame(pd.read_csv(TRAIN_PATH))
    val_df = normalize_input_frame(pd.read_csv(VAL_PATH))
    test_df = normalize_input_frame(pd.read_csv(TEST_PATH))

    if TAXONOMY_PATH.exists():
        taxonomy = pd.read_csv(TAXONOMY_PATH).sort_values("label_id")
        id_to_label = dict(zip(taxonomy["label_id"].astype(int), taxonomy["label"].astype(str)))
    else:
        all_labels = (
            pd.concat([train_df[["label_id", "label_name"]], val_df[["label_id", "label_name"]], test_df[["label_id", "label_name"]]])
            .drop_duplicates("label_id")
            .sort_values("label_id")
        )
        id_to_label = dict(zip(all_labels["label_id"].astype(int), all_labels["label_name"].astype(str)))

    label_ids = sorted(id_to_label.keys())
    label_names = [id_to_label[i] for i in label_ids]
    return train_df, val_df, test_df, id_to_label, label_ids, label_names


def check_text_overlap(train_df, val_df, test_df):
    train_text = set(train_df["text"].str.lower().str.strip())
    val_text = set(val_df["text"].str.lower().str.strip())
    test_text = set(test_df["text"].str.lower().str.strip())
    return {
        "train_val_overlap": len(train_text & val_text),
        "train_test_overlap": len(train_text & test_text),
        "val_test_overlap": len(val_text & test_text),
    }


def build_quality_warnings(comparison_df, overlap_report):
    warnings_list = []
    if any(value > 0 for value in overlap_report.values()):
        warnings_list.append(
            "Duplicate text was found across train/validation/test splits. Rebuild Step 10 splits before reporting metrics."
        )

    if not comparison_df.empty and comparison_df["test_f1_weighted"].max() >= PERFECT_SCORE_WARNING_THRESHOLD:
        warnings_list.append(
            "NLP test F1 is near-perfect on synthetic data. Treat this as a controlled benchmark, not a real-world payer generalization estimate."
        )
        warnings_list.append(
            "Synthetic templates may remain too separable even without exact duplicate leakage; validate with real CARC/RARC remark text before buyer claims."
        )
    return warnings_list


def evaluate_predictions(y_true, y_pred, label_ids):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_tfidf_baseline(train_df, val_df, test_df, label_ids, label_names):
    print_section("TF-IDF + LOGISTIC REGRESSION BASELINE")
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=1200,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    t0 = time.time()
    model.fit(train_df["text"], train_df["label_id"])
    seconds = round(time.time() - t0, 2)

    val_pred = model.predict(val_df["text"])
    test_pred = model.predict(test_df["text"])
    val_metrics = evaluate_predictions(val_df["label_id"], val_pred, label_ids)
    test_metrics = evaluate_predictions(test_df["label_id"], test_pred, label_ids)

    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Training seconds: {seconds}")
    print(f"Validation F1 weighted: {val_metrics['f1_weighted']:.4f}")
    print(f"Test F1 weighted      : {test_metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_df["label_id"],
            test_pred,
            labels=label_ids,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "name": "TF-IDF Logistic Regression",
        "model": model,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_seconds": seconds,
    }


if TORCH_TRANSFORMERS_AVAILABLE:
    class DenialTextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.texts = list(texts)
            self.labels = list(labels)
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                truncation=True,
                max_length=MAX_LEN,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
            }


def make_loader(df, tokenizer, batch_size, shuffle):
    dataset = DenialTextDataset(df["text"].values, df["label_id"].values, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def torch_eval(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += float(outputs.loss.item())
            probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())
    return {
        "loss": total_loss / max(len(loader), 1),
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
    }


def torch_train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())
        preds = torch.argmax(outputs.logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())
        if step % 250 == 0:
            print(f"  batch {step:,}/{len(loader):,} | loss={loss.item():.4f}")
    return {
        "loss": total_loss / max(len(loader), 1),
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
    }


def maybe_limit_for_cpu(train_df, val_df, test_df):
    if torch.cuda.is_available():
        return train_df, val_df, test_df, False

    def sample(df, n):
        if len(df) <= n:
            return df
        return (
            df.groupby("label_id", group_keys=False)
            .apply(lambda group: group.sample(frac=n / len(df), random_state=RANDOM_STATE))
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    return (
        sample(train_df, CPU_TRAIN_LIMIT),
        sample(val_df, CPU_VAL_LIMIT),
        sample(test_df, CPU_TEST_LIMIT),
        True,
    )


def train_distilbert(train_df, val_df, test_df, label_ids, label_names, id_to_label):
    print_section("DISTILBERT FINE-TUNING")
    if not TORCH_TRANSFORMERS_AVAILABLE:
        print("torch/transformers not installed. Skipping DistilBERT.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df, cpu_limited = maybe_limit_for_cpu(train_df, val_df, test_df)
    batch_size = 32 if torch.cuda.is_available() else 8
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    if cpu_limited:
        print(
            f"CPU practical mode: train={len(train_df):,}, val={len(val_df):,}, test={len(test_df):,}. "
            "Use GPU for full DistilBERT training."
        )

    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(
            DISTILBERT_MODEL_NAME,
            num_labels=len(label_ids),
            id2label={int(k): v for k, v in id_to_label.items()},
            label2id={v: int(k) for k, v in id_to_label.items()},
            dropout=0.25,
            attention_dropout=0.25,
        )
    except Exception as exc:
        print(f"DistilBERT could not be loaded: {exc}")
        print("Skipping DistilBERT. The TF-IDF baseline remains available.")
        return None

    model.to(device)
    train_loader = make_loader(train_df, tokenizer, batch_size, shuffle=True)
    val_loader = make_loader(val_df, tokenizer, batch_size, shuffle=False)
    test_loader = make_loader(test_df, tokenizer, batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")
        train_eval = torch_train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_eval = torch_eval(model, val_loader, device)

        train_f1 = f1_score(train_eval["labels"], train_eval["preds"], average="weighted", zero_division=0)
        val_f1 = f1_score(val_eval["labels"], val_eval["preds"], average="weighted", zero_division=0)
        val_acc = accuracy_score(val_eval["labels"], val_eval["preds"])
        gap = train_f1 - val_f1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_eval["loss"],
                "train_f1": train_f1,
                "val_loss": val_eval["loss"],
                "val_f1": val_f1,
                "val_accuracy": val_acc,
                "train_val_f1_gap": gap,
            }
        )
        print(
            f"  train_loss={train_eval['loss']:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_eval['loss']:.4f} val_f1={val_f1:.4f} gap={gap:.4f}"
        )

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(DISTILBERT_MODEL_PATH)
            tokenizer.save_pretrained(DISTILBERT_MODEL_PATH)
            print(f"  saved best DistilBERT model: val_f1={best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  early stopping counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                break

    best_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_MODEL_PATH)
    best_model.to(device)
    test_eval = torch_eval(best_model, test_loader, device)
    test_metrics = evaluate_predictions(test_eval["labels"], test_eval["preds"], label_ids)
    val_metrics = {"f1_weighted": best_val_f1}
    seconds = round(time.time() - t0, 2)

    pd.DataFrame(history).to_csv(TRAINING_HISTORY_PATH, index=False)
    print(f"\nBest epoch: {best_epoch}")
    print(f"DistilBERT test F1 weighted: {test_metrics['f1_weighted']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_eval["labels"],
            test_eval["preds"],
            labels=label_ids,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "name": "DistilBERT",
        "model": best_model,
        "test_pred": test_eval["preds"],
        "test_probs": test_eval["probs"],
        "test_labels": test_eval["labels"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_seconds": seconds,
        "best_epoch": best_epoch,
        "cpu_limited": cpu_limited,
    }


def save_confusion_matrix(y_true, y_pred, label_ids, label_names, filename, title):
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metric_chart(comparison_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = comparison_df.sort_values("test_f1_weighted")
    ax.barh(plot_df["model"], plot_df["test_f1_weighted"], color="#4E79A7")
    for i, value in enumerate(plot_df["test_f1_weighted"]):
        ax.text(value + 0.003, i, f"{value:.4f}", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Weighted F1")
    ax.set_title("NLP Root-Cause Classifier Comparison")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "nlp_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_predictions(test_df, id_to_label, best_result):
    pred_df = test_df.copy()
    pred_df["predicted_label_id"] = best_result["test_pred"]
    pred_df["predicted_label"] = [id_to_label[int(i)] for i in best_result["test_pred"]]
    pred_df["correct"] = pred_df["label_id"].values == np.array(best_result["test_pred"])
    if "test_probs" in best_result:
        pred_df["confidence"] = np.max(best_result["test_probs"], axis=1)
    pred_df.to_csv(TEST_PREDICTIONS_PATH, index=False)


def save_model_card(comparison_df, best_result, overlap_report, id_to_label, quality_warnings):
    card = {
        "project": "WellMind denial root-cause NLP classifier",
        "task": "10-class synthetic RARC-style denial reason classification",
        "best_model": best_result["name"],
        "best_test_metrics": best_result["test_metrics"],
        "data_source": "Synthetic RARC-style dataset generated from Step 09 taxonomy and Step 10 templates.",
        "real_denial_text_used": False,
        "labels": {str(k): v for k, v in id_to_label.items()},
        "leakage_check": overlap_report,
        "quality_warnings": quality_warnings,
        "metric_interpretation": {
            "synthetic_benchmark": True,
            "real_world_generalization_claim": False,
            "perfect_score_threshold": PERFECT_SCORE_WARNING_THRESHOLD,
        },
        "model_comparison": comparison_df.to_dict(orient="records"),
        "limitations": [
            "Synthetic text is useful for demo/model plumbing, not a substitute for real payer remittance data.",
            "Near-perfect metrics can occur because category-specific template language is easier than real payer remark text.",
            "A production deployment should retrain or calibrate with real CARC/RARC remarks and adjudication outcomes.",
        ],
        "recommended_next_steps": [
            "Add real payer remittance remark text when available.",
            "Evaluate on a time-based or payer-based external holdout.",
            "Continue adding ambiguous, cross-category, and noisy examples to the synthetic generator.",
        ],
    }
    MODEL_CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")


def main():
    print_section("STEP 11 - NLP ROOT-CAUSE CLASSIFIER")
    train_df, val_df, test_df, id_to_label, label_ids, label_names = load_data()

    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Labels: {label_names}")

    overlap_report = check_text_overlap(train_df, val_df, test_df)
    print(f"Leakage check: {overlap_report}")
    if any(value > 0 for value in overlap_report.values()):
        print("Warning: duplicate text exists across splits. Review Step 10 split generation.")

    tfidf_result = train_tfidf_baseline(train_df, val_df, test_df, label_ids, label_names)
    results = [tfidf_result]

    distilbert_result = train_distilbert(train_df, val_df, test_df, label_ids, label_names, id_to_label)
    if distilbert_result is not None:
        results.append(distilbert_result)

    comparison_rows = []
    for result in results:
        row = {"model": result["name"], "training_seconds": result["training_seconds"]}
        row.update({f"test_{k}": v for k, v in result["test_metrics"].items()})
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows).sort_values("test_f1_weighted", ascending=False)
    comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    quality_warnings = build_quality_warnings(comparison_df, overlap_report)

    best_result = results[int(np.argmax([r["test_metrics"]["f1_weighted"] for r in results]))]
    save_predictions(test_df, id_to_label, best_result)
    save_confusion_matrix(
        test_df["label_id"] if best_result["name"] == "TF-IDF Logistic Regression" else best_result["test_labels"],
        best_result["test_pred"],
        label_ids,
        label_names,
        "nlp_confusion_matrix_best.png",
        f"Best NLP Model Confusion Matrix - {best_result['name']}",
    )
    save_metric_chart(comparison_df)
    save_model_card(comparison_df, best_result, overlap_report, id_to_label, quality_warnings)

    print_section("FINAL NLP SUMMARY")
    print(comparison_df.to_string(index=False))
    if quality_warnings:
        print("\nQuality warnings:")
        for warning in quality_warnings:
            print(f"- {warning}")
    print(f"\nBest NLP model: {best_result['name']}")
    print(f"Best weighted F1: {best_result['test_metrics']['f1_weighted']:.4f}")
    print(f"Saved comparison: {MODEL_COMPARISON_PATH}")
    print(f"Saved predictions: {TEST_PREDICTIONS_PATH}")
    print(f"Saved model card: {MODEL_CARD_PATH}")
    print("STEP 11 COMPLETE")


if __name__ == "__main__":
    main()
