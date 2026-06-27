"""
STEP 07A - Pre-Model Target EDA & Charts
WellMind Data Solutions - Medicare Underpayment / Denial-Risk Proxy
Run: python step07_premodel_eda.py

Purpose:
Create target-level EDA and charts before model training.

Important:
The CMS public use file does not include real claim denial labels or RARC/CARC
denial codes. The "denied" column created here is a synthetic portfolio proxy:
rows in the lowest 20% payment-to-charge ratio are labeled as underpayment /
denial-risk. Treat this as a modeling demo target, not actual CMS denial data.

Input:
    Medicare_Cleaned_Outliers.csv

Outputs:
    Medicare_Final_WithTarget.csv
    premodel_eda_summary.csv
    target_definition_card.md
    modeling_feature_policy.csv
    charts/chart_01_target_variable.png
    charts/chart_02_payment_distribution.png
    charts/chart_03_boxplots.png
    charts/chart_04_denial_by_specialty.png
    charts/chart_05_place_of_service.png
    charts/chart_06_top_denied_hcpcs.png
    charts/chart_07_correlation_heatmap.png
    charts/chart_08_ruca_analysis.png
    charts/chart_09_credentials_analysis.png
    charts/chart_10_drug_analysis.png
    charts/chart_11_state_denial_rate.png
    charts/chart_12_provider_entity.png
    charts/chart_13_medicare_participation.png
    charts/chart_14_services_vs_payment.png
    charts/chart_15_charge_vs_payment.png
    charts/chart_16_specialty_bubble.png
    charts/chart_17_volume_segments.png
    charts/chart_18_allowed_vs_payment.png
    charts/chart_19_frequency_features.png
    charts/chart_20_feature_target_correlation.png
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu, ttest_ind

warnings.filterwarnings("ignore")


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Medicare_Cleaned_Outliers.csv"
FINAL_WITH_TARGET = BASE_DIR / "Medicare_Final_WithTarget.csv"
SUMMARY_PATH = BASE_DIR / "premodel_eda_summary.csv"
SPECIALTY_SUMMARY_PATH = BASE_DIR / "premodel_denial_by_specialty.csv"
HCPCS_SUMMARY_PATH = BASE_DIR / "premodel_top_denied_hcpcs.csv"
PLACE_SUMMARY_PATH = BASE_DIR / "premodel_denial_by_place.csv"
STATS_SUMMARY_PATH = BASE_DIR / "premodel_statistical_tests.csv"
TARGET_CARD_PATH = BASE_DIR / "target_definition_card.md"
FEATURE_POLICY_PATH = BASE_DIR / "modeling_feature_policy.csv"
CHART_DIR = BASE_DIR / "charts"

RANDOM_STATE = 42
SAMPLE_ROWS = 100_000
# False avoids writing another 1GB+ CSV. If a later step needs
# Medicare_Final_WithTarget.csv specifically, set this True and rerun Step 07.
SAVE_FULL_TARGET_DATASET = False

TARGET_INPUT_COLUMNS = [
    "Avg_Mdcr_Pymt_Amt_log",
    "Avg_Sbmtd_Chrg_log",
]

# These fields are too close to the synthetic target definition for honest
# model evaluation. Step 08 should exclude them from training features.
LEAKAGE_EXCLUDE_COLUMNS = [
    "Avg_Mdcr_Pymt_Amt_log",
    "Avg_Sbmtd_Chrg_log",
    "Avg_Mdcr_Alowd_Amt_log",
    "Avg_Mdcr_Stdzd_Amt_log",
]


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_chart(fig, filename, title=None):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = CHART_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {path}")


def create_target(df):
    """Create a synthetic denial-risk proxy from payment-to-charge ratio."""
    required = TARGET_INPUT_COLUMNS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Cannot create target. Missing columns: {missing}")

    ratio = df["Avg_Mdcr_Pymt_Amt_log"] / (df["Avg_Sbmtd_Chrg_log"] + 1e-6)
    threshold = ratio.quantile(0.20)
    target = (ratio <= threshold).astype(int)
    return target, float(threshold)


def add_summary(summary_rows, name, value):
    summary_rows.append({"metric": name, "value": value})


def save_target_card(threshold, denied_rate, total_rows):
    """Write a clear target definition note for model cards and buyers."""
    card = f"""# Target Definition Card

Project: WellMind Data Solutions - Claim Denial Prediction Demo

## Target Name
denied

## Target Type
Synthetic denial-risk / underpayment-risk proxy.

## Why This Is Synthetic
The CMS Medicare Physician & Other Practitioners public use file does not
include real claim denial labels, CARC codes, or RARC denial reason codes.
For a public-data portfolio demo, this project labels the lowest 20 percent
of payment-to-charge ratio rows as higher denial/underpayment risk.

## Formula
payment_to_charge_ratio = Avg_Mdcr_Pymt_Amt_log / (Avg_Sbmtd_Chrg_log + 1e-6)

denied = 1 when payment_to_charge_ratio <= {threshold:.8f}

denied = 0 otherwise

## Dataset Impact
Rows evaluated: {total_rows:,}

Synthetic denied rate: {denied_rate:.4f}%

## Modeling Safety Rule
Do not train the model using columns that directly create or closely mirror
the target. Step 08 must exclude the leakage-risk columns listed in
modeling_feature_policy.csv.

## Buyer-Facing Language
Call this a denial-risk proxy model, not a real CMS denial classifier, until
real payer denial labels or remittance codes are connected.
"""
    TARGET_CARD_PATH.write_text(card, encoding="utf-8")


def save_feature_policy(df):
    """Save which features are allowed/excluded for leakage-safe modeling."""
    rows = []
    for col in df.columns:
        if col == "denied":
            policy = "target"
            reason = "Prediction label"
        elif col in LEAKAGE_EXCLUDE_COLUMNS:
            policy = "exclude"
            reason = "Directly creates or closely mirrors the synthetic target"
        elif col in TARGET_INPUT_COLUMNS:
            policy = "exclude"
            reason = "Direct input to target formula"
        else:
            policy = "candidate"
            reason = "Available for Step 08 after train-fold preprocessing"

        rows.append(
            {
                "Column": col,
                "Policy": policy,
                "Reason": reason,
            }
        )

    pd.DataFrame(rows).to_csv(FEATURE_POLICY_PATH, index=False)


def safe_chi_square(df, col, target_col="denied"):
    """Chi-square test for categorical feature association with target."""
    if col not in df.columns:
        return None
    table = pd.crosstab(df[col].fillna("Missing"), df[target_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, p_value, dof, _ = chi2_contingency(table)
    return {
        "test": "Chi-square",
        "feature": col,
        "target": target_col,
        "statistic": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "interpretation": "Associated with target" if p_value < 0.05 else "No significant association",
    }


def safe_group_numeric_tests(df, group_col, value_col):
    """ANOVA and Kruskal-Wallis for numeric differences across groups."""
    if group_col not in df.columns or value_col not in df.columns:
        return []

    groups = []
    for _, series in df.groupby(group_col)[value_col]:
        clean = series.dropna()
        if len(clean) >= 30:
            groups.append(clean)

    if len(groups) < 2:
        return []

    results = []
    try:
        f_stat, p_value = f_oneway(*groups)
        results.append({
            "test": "ANOVA",
            "feature": group_col,
            "target": value_col,
            "statistic": float(f_stat),
            "p_value": float(p_value),
            "dof": "",
            "interpretation": "Group means differ" if p_value < 0.05 else "No significant mean difference",
        })
    except Exception as exc:
        results.append({
            "test": "ANOVA",
            "feature": group_col,
            "target": value_col,
            "statistic": "",
            "p_value": "",
            "dof": "",
            "interpretation": f"Skipped: {exc}",
        })

    try:
        h_stat, p_value = kruskal(*groups)
        results.append({
            "test": "Kruskal-Wallis",
            "feature": group_col,
            "target": value_col,
            "statistic": float(h_stat),
            "p_value": float(p_value),
            "dof": "",
            "interpretation": "Group distributions differ" if p_value < 0.05 else "No significant distribution difference",
        })
    except Exception as exc:
        results.append({
            "test": "Kruskal-Wallis",
            "feature": group_col,
            "target": value_col,
            "statistic": "",
            "p_value": "",
            "dof": "",
            "interpretation": f"Skipped: {exc}",
        })

    return results


def safe_binary_numeric_tests(df, group_col, value_col, positive_label=None):
    """Welch t-test and Mann-Whitney test for two-group numeric comparisons."""
    if group_col not in df.columns or value_col not in df.columns:
        return []

    values = df[group_col].dropna().unique()
    if len(values) != 2:
        return []

    a_label, b_label = values[0], values[1]
    if positive_label in values:
        b_label = positive_label
        a_label = [v for v in values if v != positive_label][0]

    a = df.loc[df[group_col] == a_label, value_col].dropna()
    b = df.loc[df[group_col] == b_label, value_col].dropna()
    if len(a) < 30 or len(b) < 30:
        return []

    results = []
    t_stat, p_value = ttest_ind(a, b, equal_var=False)
    results.append({
        "test": "Welch t-test",
        "feature": group_col,
        "target": value_col,
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "dof": "",
        "interpretation": f"{value_col} differs between {a_label} and {b_label}" if p_value < 0.05 else "No significant mean difference",
    })

    u_stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
    results.append({
        "test": "Mann-Whitney U",
        "feature": group_col,
        "target": value_col,
        "statistic": float(u_stat),
        "p_value": float(p_value),
        "dof": "",
        "interpretation": f"{value_col} distribution differs between {a_label} and {b_label}" if p_value < 0.05 else "No significant distribution difference",
    })
    return results


def main():
    print_section("STEP 07A - PRE-MODEL TARGET EDA")
    print(f"Input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df["denied"], target_threshold = create_target(df)

    total_rows = len(df)
    denied_count = int(df["denied"].sum())
    approved_count = total_rows - denied_count
    denied_rate = denied_count / total_rows * 100

    print(f"Rows: {total_rows:,}")
    print(f"Columns: {df.shape[1]}")
    print("Target definition: lowest 20% payment-to-charge ratio = denial/underpayment risk proxy")
    print("Important: this is a synthetic proxy target, not actual CMS denial data.")
    print(f"Target threshold: {target_threshold:.6f}")
    print(f"Approved (0): {approved_count:,} ({100 - denied_rate:.2f}%)")
    print(f"Denied (1): {denied_count:,} ({denied_rate:.2f}%)")

    if SAVE_FULL_TARGET_DATASET:
        print("\nSaving full target dataset. This can take a few minutes for 9M+ rows...")
        df.to_csv(FINAL_WITH_TARGET, index=False)
        print(f"Saved: {FINAL_WITH_TARGET}")
    else:
        print("\nSkipped full target dataset save to avoid extra 1GB+ duplicate file.")
        print("Target is created in memory; summaries, charts, and statistical tests are still saved.")

    sample_n = min(SAMPLE_ROWS, total_rows)
    df_sample = df.sample(n=sample_n, random_state=RANDOM_STATE)
    print(f"Chart sample rows: {sample_n:,}")
    print("Charts with heavy scatter/correlation/stat tests use df_sample for speed;")
    print("grouped business summaries use full df when practical.")

    summary_rows = []
    add_summary(summary_rows, "rows", total_rows)
    add_summary(summary_rows, "columns", df.shape[1])
    add_summary(summary_rows, "target_threshold", target_threshold)
    add_summary(summary_rows, "approved_count", approved_count)
    add_summary(summary_rows, "denied_count", denied_count)
    add_summary(summary_rows, "denied_rate_pct", round(denied_rate, 4))
    add_summary(summary_rows, "chart_sample_rows", sample_n)
    add_summary(summary_rows, "target_type", "synthetic proxy, not actual CMS denial label")

    save_target_card(target_threshold, denied_rate, total_rows)
    save_feature_policy(df)
    print(f"Saved target definition card: {TARGET_CARD_PATH}")
    print(f"Saved modeling feature policy: {FEATURE_POLICY_PATH}")

    print_section("STATISTICAL TESTS")
    stats_rows = []

    # Categorical association with target: appropriate for denial-risk EDA.
    categorical_tests = [
        "Place_Of_Srvc",
        "HCPCS_Drug_Ind",
        "Rndrng_Prvdr_Ent_Cd",
        "Rndrng_Prvdr_Mdcr_Prtcptg_Ind",
        "Crdntls_Group",
        "Rndrng_Prvdr_State_Abrvtn",
    ]
    for col in categorical_tests:
        result = safe_chi_square(df_sample, col)
        if result:
            stats_rows.append(result)

    # Numeric comparisons across clinically meaningful groups.
    stats_rows.extend(safe_group_numeric_tests(df_sample, "Crdntls_Group", "Avg_Mdcr_Pymt_Amt_log"))
    stats_rows.extend(safe_group_numeric_tests(df_sample, "Rndrng_Prvdr_Type", "Avg_Mdcr_Pymt_Amt_log"))
    stats_rows.extend(safe_binary_numeric_tests(df_sample, "HCPCS_Drug_Ind", "Avg_Mdcr_Pymt_Amt_log", positive_label="Y"))
    stats_rows.extend(safe_binary_numeric_tests(df_sample, "Place_Of_Srvc", "Avg_Mdcr_Pymt_Amt_log", positive_label="F"))

    if "Avg_Mdcr_Pymt_Amt_log" in df_sample.columns:
        normal_values = df_sample["Avg_Mdcr_Pymt_Amt_log"].dropna()
        if len(normal_values) >= 8:
            normal_sample = normal_values.sample(
                n=min(5_000, len(normal_values)),
                random_state=RANDOM_STATE,
            )
            stat, p_value = stats.normaltest(normal_sample)
            stats_rows.append({
                "test": "D'Agostino normality",
                "feature": "Avg_Mdcr_Pymt_Amt_log",
                "target": "normal distribution",
                "statistic": float(stat),
                "p_value": float(p_value),
                "dof": "",
                "interpretation": "Not normally distributed" if p_value < 0.05 else "Approximately normal",
            })

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df.to_csv(STATS_SUMMARY_PATH, index=False)
        print(stats_df[["test", "feature", "target", "p_value", "interpretation"]].to_string(index=False))
        print(f"Saved statistical tests: {STATS_SUMMARY_PATH}")
    else:
        print("No statistical tests were applicable with available columns.")

    print_section("CHART 1 - TARGET DISTRIBUTION")
    counts = df["denied"].value_counts().sort_index()
    labels = ["Approved / Lower Risk", "Denied / Underpayment Risk"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                colors=["#59A14F", "#E15759"])
    axes[0].set_title("Target Distribution")
    axes[1].bar(labels, values, color=["#59A14F", "#E15759"])
    axes[1].set_title("Claim Count by Target")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=15)
    save_chart(fig, "chart_01_target_variable.png", "Target Variable Distribution")

    print_section("CHART 2 - PAYMENT DISTRIBUTIONS")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, color, title in [
        (axes[0], "Avg_Mdcr_Pymt_Amt_log", "#4E79A7", "Medicare Payment"),
        (axes[1], "Avg_Sbmtd_Chrg_log", "#F28E2B", "Submitted Charge"),
    ]:
        if col in df_sample.columns:
            data = df_sample[col].dropna()
            ax.hist(data, bins=70, color=color, alpha=0.75)
            ax.axvline(data.mean(), color="black", linestyle="--", label="Mean")
            ax.axvline(data.median(), color="#E15759", linestyle="-.", label="Median")
            ax.set_title(f"{title} Distribution")
            ax.set_xlabel("Log value")
            ax.legend()
    save_chart(fig, "chart_02_payment_distribution.png", "Payment and Charge Distributions")

    print_section("CHART 3 - NUMERIC BOXPLOTS")
    box_cols = [
        "Tot_Benes_log",
        "Tot_Srvcs_log",
        "Avg_Sbmtd_Chrg_log",
        "Avg_Mdcr_Pymt_Amt_log",
    ]
    box_cols = [col for col in box_cols if col in df_sample.columns]
    if box_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot([df_sample[col].dropna() for col in box_cols],
                   tick_labels=[col.replace("_log", "") for col in box_cols])
        ax.set_title("Numeric Feature Distribution Overview")
        ax.set_ylabel("Log value")
        ax.tick_params(axis="x", rotation=20)
        save_chart(fig, "chart_03_boxplots.png", "Numeric Feature Boxplots")

    print_section("CHART 4 - DENIAL RISK BY SPECIALTY")
    if "Rndrng_Prvdr_Type" in df.columns:
        spec = df.groupby("Rndrng_Prvdr_Type")["denied"].agg(["mean", "count"])
        spec = spec[spec["count"] >= 500].sort_values("mean").tail(20)
        spec["denial_rate_pct"] = spec["mean"] * 100
        spec.to_csv(SPECIALTY_SUMMARY_PATH)
        if not spec.empty:
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.barh(spec.index, spec["denial_rate_pct"], color="#E15759")
            ax.set_xlabel("Denial-risk rate (%)")
            ax.set_title("Top Provider Specialties by Denial-Risk Rate")
            save_chart(fig, "chart_04_denial_by_specialty.png", "Denial Risk by Provider Specialty")

    print_section("CHART 5 - PLACE OF SERVICE")
    if "Place_Of_Srvc" in df.columns:
        place = df.groupby("Place_Of_Srvc")["denied"].agg(["mean", "count"]).sort_index()
        place["denial_rate_pct"] = place["mean"] * 100
        place.to_csv(PLACE_SUMMARY_PATH)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(place.index.astype(str), place["denial_rate_pct"], color="#76B7B2")
        ax.set_xlabel("Place of Service")
        ax.set_ylabel("Denial-risk rate (%)")
        ax.set_title("Denial Risk by Place of Service")
        save_chart(fig, "chart_05_place_of_service.png", "Place of Service Analysis")

    print_section("CHART 6 - TOP HCPCS CODES")
    if "HCPCS_Cd" in df.columns:
        hcpcs = df.groupby("HCPCS_Cd")["denied"].agg(["mean", "count"])
        hcpcs = hcpcs[hcpcs["count"] >= 100].sort_values("mean").tail(20)
        hcpcs["denial_rate_pct"] = hcpcs["mean"] * 100
        hcpcs.to_csv(HCPCS_SUMMARY_PATH)
        if not hcpcs.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(hcpcs.index.astype(str), hcpcs["denial_rate_pct"], color="#B07AA1")
            ax.set_xlabel("Denial-risk rate (%)")
            ax.set_title("Top HCPCS Codes by Denial-Risk Rate")
            save_chart(fig, "chart_06_top_denied_hcpcs.png", "Top Denial-Risk HCPCS Codes")

    print_section("CHART 7 - CORRELATION HEATMAP")
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "denied"][:18]
    if numeric_cols:
        corr = df_sample[numeric_cols + ["denied"]].corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        save_chart(fig, "chart_07_correlation_heatmap.png", "Correlation Heatmap")

    print_section("CHART 8 - RUCA ANALYSIS")
    if "Rndrng_Prvdr_RUCA" in df.columns:
        ruca_bins = [0, 3, 6, 10.6, 100]
        ruca_labels = ["Urban", "Suburban", "Rural", "Unknown/Special"]
        df["RUCA_Category"] = pd.cut(df["Rndrng_Prvdr_RUCA"], bins=ruca_bins, labels=ruca_labels, include_lowest=True)
        ruca = df.groupby("RUCA_Category")["denied"].agg(["mean", "count"])
        ruca["denial_rate_pct"] = ruca["mean"] * 100
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(ruca.index.astype(str), ruca["denial_rate_pct"], color="#4E79A7")
        ax.set_xlabel("RUCA Category")
        ax.set_ylabel("Denial-risk rate (%)")
        ax.set_title("Denial Risk by Rural/Urban Category")
        save_chart(fig, "chart_08_ruca_analysis.png", "RUCA Analysis")
        df.drop(columns=["RUCA_Category"], inplace=True, errors="ignore")

    print_section("CHART 9 - CREDENTIAL GROUP ANALYSIS")
    if "Crdntls_Group" in df.columns:
        cred = df.groupby("Crdntls_Group")["denied"].agg(["mean", "count"]).sort_values("mean")
        cred["denial_rate_pct"] = cred["mean"] * 100
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(cred.index.astype(str), cred["denial_rate_pct"], color="#F28E2B")
        ax.set_xlabel("Denial-risk rate (%)")
        ax.set_title("Denial Risk by Credential Group")
        save_chart(fig, "chart_09_credentials_analysis.png", "Credential Group Analysis")

    print_section("CHART 10 - DRUG INDICATOR ANALYSIS")
    if "HCPCS_Drug_Ind" in df.columns:
        drug = df.groupby("HCPCS_Drug_Ind")["denied"].agg(["mean", "count"]).sort_index()
        drug["denial_rate_pct"] = drug["mean"] * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(drug.index.astype(str), drug["denial_rate_pct"], color="#59A14F")
        ax.set_xlabel("HCPCS Drug Indicator")
        ax.set_ylabel("Denial-risk rate (%)")
        ax.set_title("Denial Risk by Drug Indicator")
        save_chart(fig, "chart_10_drug_analysis.png", "Drug Indicator Analysis")

    print_section("CHART 11 - STATE DENIAL RISK")
    if "Rndrng_Prvdr_State_Abrvtn" in df.columns:
        state = df.groupby("Rndrng_Prvdr_State_Abrvtn")["denied"].agg(["mean", "count"])
        state = state[state["count"] >= 500].sort_values("mean", ascending=False).head(25)
        state["denial_rate_pct"] = state["mean"] * 100
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(state.index.astype(str), state["denial_rate_pct"], color="#4E79A7")
        ax.set_xlabel("State")
        ax.set_ylabel("Denial-risk rate (%)")
        ax.set_title("Top States by Denial-Risk Rate")
        ax.tick_params(axis="x", rotation=45)
        save_chart(fig, "chart_11_state_denial_rate.png", "State Denial-Risk Rate")

    print_section("CHART 12 - PROVIDER ENTITY TYPE")
    if "Rndrng_Prvdr_Ent_Cd" in df.columns:
        ent = df.groupby("Rndrng_Prvdr_Ent_Cd")["denied"].agg(["mean", "count"]).sort_index()
        ent["denial_rate_pct"] = ent["mean"] * 100
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(ent.index.astype(str), ent["count"], color="#76B7B2")
        axes[0].set_title("Rows by Provider Entity Type")
        axes[0].set_ylabel("Rows")
        axes[1].bar(ent.index.astype(str), ent["denial_rate_pct"], color="#E15759")
        axes[1].set_title("Denial Risk by Entity Type")
        axes[1].set_ylabel("Denial-risk rate (%)")
        save_chart(fig, "chart_12_provider_entity.png", "Provider Entity Type Analysis")

    print_section("CHART 13 - MEDICARE PARTICIPATION")
    if "Rndrng_Prvdr_Mdcr_Prtcptg_Ind" in df.columns:
        part = df.groupby("Rndrng_Prvdr_Mdcr_Prtcptg_Ind")["denied"].agg(["mean", "count"]).sort_index()
        part["denial_rate_pct"] = part["mean"] * 100
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(part.index.astype(str), part["count"], color="#B07AA1")
        axes[0].set_title("Rows by Medicare Participation")
        axes[0].set_ylabel("Rows")
        axes[1].bar(part.index.astype(str), part["denial_rate_pct"], color="#F28E2B")
        axes[1].set_title("Denial Risk by Medicare Participation")
        axes[1].set_ylabel("Denial-risk rate (%)")
        save_chart(fig, "chart_13_medicare_participation.png", "Medicare Participation Analysis")

    print_section("CHART 14 - SERVICES VS PAYMENT")
    if {"Tot_Srvcs_log", "Avg_Mdcr_Pymt_Amt_log"}.issubset(df_sample.columns):
        scatter = df_sample.sample(n=min(8_000, len(df_sample)), random_state=RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = scatter["denied"].map({0: "#59A14F", 1: "#E15759"})
        ax.scatter(scatter["Tot_Srvcs_log"], scatter["Avg_Mdcr_Pymt_Amt_log"], c=colors, s=8, alpha=0.35)
        ax.set_xlabel("Total Services (log)")
        ax.set_ylabel("Medicare Payment (log)")
        ax.set_title("Services vs Payment by Target")
        save_chart(fig, "chart_14_services_vs_payment.png", "Services vs Payment")

    print_section("CHART 15 - CHARGE VS PAYMENT")
    if {"Avg_Sbmtd_Chrg_log", "Avg_Mdcr_Pymt_Amt_log"}.issubset(df_sample.columns):
        scatter = df_sample.sample(n=min(8_000, len(df_sample)), random_state=RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = scatter["denied"].map({0: "#59A14F", 1: "#E15759"})
        ax.scatter(scatter["Avg_Sbmtd_Chrg_log"], scatter["Avg_Mdcr_Pymt_Amt_log"], c=colors, s=8, alpha=0.35)
        ax.set_xlabel("Submitted Charge (log)")
        ax.set_ylabel("Medicare Payment (log)")
        ax.set_title("Submitted Charge vs Payment by Target")
        save_chart(fig, "chart_15_charge_vs_payment.png", "Submitted Charge vs Payment")

    print_section("CHART 16 - SPECIALTY BUBBLE CHART")
    if "Rndrng_Prvdr_Type" in df.columns:
        bubble = df.groupby("Rndrng_Prvdr_Type").agg(
            denial_rate=("denied", "mean"),
            count=("denied", "count"),
            avg_payment=("Avg_Mdcr_Pymt_Amt_log", "mean"),
        )
        bubble = bubble[bubble["count"] >= 1000].copy()
        if not bubble.empty:
            bubble["denial_rate_pct"] = bubble["denial_rate"] * 100
            bubble["size"] = (bubble["count"] / bubble["count"].max()) * 2200
            fig, ax = plt.subplots(figsize=(12, 8))
            sc = ax.scatter(
                bubble["avg_payment"],
                bubble["denial_rate_pct"],
                s=bubble["size"],
                c=bubble["count"],
                cmap="viridis",
                alpha=0.65,
                edgecolors="white",
            )
            top_labels = bubble.sort_values("count", ascending=False).head(8)
            for label, row in top_labels.iterrows():
                ax.annotate(str(label)[:18], (row["avg_payment"], row["denial_rate_pct"]), fontsize=8)
            plt.colorbar(sc, ax=ax, label="Row count")
            ax.set_xlabel("Average Medicare Payment (log)")
            ax.set_ylabel("Denial-risk rate (%)")
            ax.set_title("Specialty Payment vs Denial Risk")
            save_chart(fig, "chart_16_specialty_bubble.png", "Specialty Bubble Chart")

    print_section("CHART 17 - VOLUME SEGMENTS")
    if "Tot_Benes_log" in df.columns:
        df["Volume_Segment"] = pd.qcut(df["Tot_Benes_log"], q=4, labels=["Low", "Medium", "High", "Very High"], duplicates="drop")
        volume = df.groupby("Volume_Segment")["denied"].agg(["mean", "count"])
        volume["denial_rate_pct"] = volume["mean"] * 100
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(volume.index.astype(str), volume["count"], color="#4E79A7")
        axes[0].set_title("Rows by Volume Segment")
        axes[0].set_ylabel("Rows")
        axes[1].bar(volume.index.astype(str), volume["denial_rate_pct"], color="#E15759")
        axes[1].set_title("Denial Risk by Volume Segment")
        axes[1].set_ylabel("Denial-risk rate (%)")
        save_chart(fig, "chart_17_volume_segments.png", "Volume Segment Analysis")
        df.drop(columns=["Volume_Segment"], inplace=True, errors="ignore")

    print_section("CHART 18 - ALLOWED VS PAYMENT")
    if {"Avg_Mdcr_Alowd_Amt_log", "Avg_Mdcr_Pymt_Amt_log"}.issubset(df_sample.columns):
        scatter = df_sample.sample(n=min(8_000, len(df_sample)), random_state=RANDOM_STATE)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = scatter["denied"].map({0: "#59A14F", 1: "#E15759"})
        ax.scatter(scatter["Avg_Mdcr_Alowd_Amt_log"], scatter["Avg_Mdcr_Pymt_Amt_log"], c=colors, s=8, alpha=0.35)
        ax.set_xlabel("Allowed Amount (log)")
        ax.set_ylabel("Payment Amount (log)")
        ax.set_title("Allowed Amount vs Payment by Target")
        save_chart(fig, "chart_18_allowed_vs_payment.png", "Allowed vs Payment")

    print_section("CHART 19 - FREQUENCY FEATURES")
    freq_cols = [col for col in ["Rndrng_Prvdr_Type_freq", "Rndrng_Prvdr_State_Abrvtn_freq", "HCPCS_Cd_freq"] if col in df_sample.columns]
    if freq_cols:
        fig, axes = plt.subplots(1, len(freq_cols), figsize=(5 * len(freq_cols), 5))
        if len(freq_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, freq_cols):
            ax.hist(df_sample[col].dropna(), bins=50, color="#76B7B2", alpha=0.8)
            ax.set_title(col)
            ax.set_xlabel("Frequency encoded value")
        save_chart(fig, "chart_19_frequency_features.png", "Frequency Encoded Feature Distributions")

    print_section("CHART 20 - FEATURE TARGET CORRELATION")
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    if "denied" in numeric_cols:
        safe_numeric_cols = [
            col for col in numeric_cols
            if col not in LEAKAGE_EXCLUDE_COLUMNS
        ]
        corr_to_target = (
            df_sample[safe_numeric_cols]
            .corr(numeric_only=True)["denied"]
            .drop(labels=["denied"], errors="ignore")
            .dropna()
            .abs()
            .sort_values(ascending=False)
            .head(20)
        )
        if not corr_to_target.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(corr_to_target.index[::-1], corr_to_target.values[::-1], color="#E15759")
            ax.set_xlabel("Absolute correlation with target")
            ax.set_title("Top Numeric Features Correlated with Denial-Risk Target")
            save_chart(fig, "chart_20_feature_target_correlation.png", "Feature Target Correlation")

    pd.DataFrame(summary_rows).to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary: {SUMMARY_PATH}")
    print("STEP 07A COMPLETE")


if __name__ == "__main__":
    main()