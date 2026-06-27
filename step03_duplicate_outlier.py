"""
STEP 03 — Duplicate & Outlier Detection (Day 2)
WellMind Data Solutions — Claim Denial Prediction System
Run: python step03_duplicate_outlier.py
"""
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# ============================================================
# LOAD DATA
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "Medicare Physician & Other Practitioners - by Provider and Service"
    / "2024"
    / "PHY_R26_P05_V10_D24_Prov_Svc.csv"
)
CHART_PATH = BASE_DIR / "outlier_detection_charts.png"
DUPLICATE_SUMMARY_PATH = BASE_DIR / "duplicate_detection_summary.csv"
OUTLIER_SUMMARY_PATH = BASE_DIR / "outlier_numeric_summary.csv"
BUSINESS_RULE_SUMMARY_PATH = BASE_DIR / "business_rule_review_summary.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"File nahi mili: {DATA_PATH}")

# Extension check karke load karo
if DATA_PATH.suffix.lower() == ".csv":
    df = pd.read_csv(DATA_PATH, low_memory=False)
elif DATA_PATH.suffix.lower() in {".xlsx", ".xls"}:
    df = pd.read_excel(DATA_PATH)
else:
    raise ValueError("Sirf .csv ya .xlsx files supported hain")

print(f"\n✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   File: {DATA_PATH}\n")

# ============================================================
# PART 1 — FULL ROW DUPLICATES
# Sare 28 columns bilkul same
# ============================================================
print("\n" + "=" * 70)
print("PART 1 — FULL ROW DUPLICATES (sare 28 columns same)")
print("=" * 70)

full_dups = df.duplicated().sum()
print(f"  Count      : {full_dups:,}")
print(f"  Percentage : {full_dups/len(df)*100:.4f}%")

if full_dups > 0:
    dup_df = df[df.duplicated(keep=False)].sort_values(list(df.columns))
    print(f"\n  Sample (first 4 duplicate rows):")
    print(dup_df.head(4).to_string())
else:
    print("  ✅ Koi full row duplicate nahi mila!")

# ============================================================
# PART 2 — COLUMN DUPLICATES
# Koi 2 columns ki values bilkul same hain?
# ============================================================
print("\n" + "=" * 70)
print("PART 2 — COLUMN DUPLICATES (2 columns ki values same)")
print("=" * 70)

col_dup_found = False
cols = df.columns.tolist()
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if df[cols[i]].equals(df[cols[j]]):
            print(f"  ⚠️  '{cols[i]}' aur '{cols[j]}' — bilkul same values!")
            col_dup_found = True

if not col_dup_found:
    print("  ✅ Koi 2 columns bilkul same nahi hain!")

# High correlation wale columns (numeric)
print("\n  📌 Highly correlated numeric columns (r > 0.95):")
num_df = df.select_dtypes(include=[np.number])
corr = num_df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = [(c, r, round(upper.loc[r, c], 4))
             for c in upper.columns
             for r in upper.index
             if pd.notna(upper.loc[r, c]) and upper.loc[r, c] > 0.95]

if high_corr:
    print(f"  {'Column 1':<28} {'Column 2':<28} {'Correlation':>12}")
    print("  " + "-" * 70)
    for c1, c2, val in high_corr:
        print(f"  {c1:<28} {c2:<28} {val:>12.4f}")
else:
    print("  ✅ Koi highly correlated columns nahi (r > 0.95)")

# ============================================================
# PART 3 — BUSINESS KEY DUPLICATES
# Same NPI + HCPCS_Cd + Place_Of_Srvc
# ============================================================
print("\n" + "=" * 70)
print("PART 3 — BUSINESS KEY DUPLICATES")
print("  (Same doctor + same service code + same place)")
print("=" * 70)

key = ['Rndrng_NPI', 'HCPCS_Cd', 'Place_Of_Srvc']
biz_dups = df.duplicated(subset=key).sum()
print(f"  Count      : {biz_dups:,}")
print(f"  Percentage : {biz_dups/len(df)*100:.4f}%")

if biz_dups > 0:
    biz_df = df[df.duplicated(subset=key, keep=False)].sort_values(key)
    print(f"\n  Sample (top 6):")
    print(biz_df[key + ['Tot_Srvcs', 'Tot_Benes', 'Avg_Mdcr_Pymt_Amt']].head(6).to_string())

# ============================================================
# PART 4 — PARTIAL DUPLICATES
# Same rows hain lekin kuch columns mein values alag hain
# ============================================================
print("\n" + "=" * 70)
print("PART 4 — PARTIAL DUPLICATES")
print("  (Rows almost same hain — sirf kuch columns mein farq)")
print("=" * 70)

partial_keys = [
    {
        'name': 'NPI + HCPCS (Place alag)',
        'cols': ['Rndrng_NPI', 'HCPCS_Cd'],
        'note': 'Same doctor, same service — Office aur Facility dono mein'
    },
    {
        'name': 'NPI only',
        'cols': ['Rndrng_NPI'],
        'note': 'Ek doctor ki kitni alag alag services hain'
    },
    {
        'name': 'Last Name + First Name + Crdntls',
        'cols': ['Rndrng_Prvdr_Last_Org_Name', 'Rndrng_Prvdr_First_Name', 'Rndrng_Prvdr_Crdntls'],
        'note': 'Same naam — NPI alag? Same doctor, 2 NPIs?'
    },
    {
        'name': 'HCPCS_Cd + Place_Of_Srvc',
        'cols': ['HCPCS_Cd', 'Place_Of_Srvc'],
        'note': 'Same service — kitne alag doctors provide karte hain'
    },
]

for pk in partial_keys:
    count = df.duplicated(subset=pk['cols']).sum()
    pct = count / len(df) * 100
    print(f"\n  📌 {pk['name']}")
    print(f"     Note    : {pk['note']}")
    print(f"     Count   : {count:,}  ({pct:.2f}%)")

# ============================================================
# PART 5 — INCONSISTENT DUPLICATES
# Same rows lekin ek mein zyada info, doosre mein kam
# Jaise: same NPI — ek row mein MI hai, doosre mein NaN
# ============================================================
print("\n" + "=" * 70)
print("PART 5 — INCONSISTENT DUPLICATES")
print("  (Same doctor — ek row mein info zyada, doosre mein kam)")
print("=" * 70)

# Same NPI wali rows dhundo
npi_groups = df.groupby('Rndrng_NPI')

# Columns jinmein inconsistency check karni hai
check_cols = [
    'Rndrng_Prvdr_First_Name',
    'Rndrng_Prvdr_MI',
    'Rndrng_Prvdr_Crdntls',
    'Rndrng_Prvdr_St1',
    'Rndrng_Prvdr_City',
    'Rndrng_Prvdr_State_Abrvtn',
]

print(f"\n  Column-wise inconsistency — same NPI mein alag values:")
print(f"  {'Column':<30} {'NPIs with conflict':>20} {'%':>8}")
print("  " + "-" * 62)

inconsistency_results = {}
for col in check_cols:
    # Har NPI ke liye is column mein kitni unique non-null values hain
    unique_per_npi = npi_groups[col].nunique(dropna=True)
    conflict_npis = (unique_per_npi > 1).sum()
    pct = conflict_npis / df['Rndrng_NPI'].nunique() * 100
    inconsistency_results[col] = conflict_npis
    print(f"  {col:<30} {conflict_npis:>20,} {pct:>7.2f}%")

# Sample — sabse zyada conflict wala column
worst_col = max(inconsistency_results, key=inconsistency_results.get)
if inconsistency_results[worst_col] > 0:
    print(f"\n  📌 Sample conflict — '{worst_col}' column:")
    conflict_npis_list = npi_groups[worst_col].nunique(dropna=True)
    conflict_npis_list = conflict_npis_list[conflict_npis_list > 1].index[:3]
    for npi in conflict_npis_list:
        rows = df[df['Rndrng_NPI'] == npi][
            ['Rndrng_NPI', 'Rndrng_Prvdr_Last_Org_Name',
             'Rndrng_Prvdr_First_Name', worst_col]
        ].drop_duplicates()
        print(f"\n  NPI: {npi}")
        print(rows.to_string(index=False))

# ============================================================
# PART 6 — NULL vs FILLED DUPLICATES
# Same rows — ek mein null hai, doosre mein value hai
# ============================================================
print("\n" + "=" * 70)
print("PART 6 — NULL vs FILLED DUPLICATES")
print("  (Almost same rows — ek mein null, doosre mein value)")
print("=" * 70)

# NPI + HCPCS + Place same ho lekin null columns mein farq ho
key2 = ['Rndrng_NPI', 'HCPCS_Cd', 'Place_Of_Srvc']
nullable_cols = ['Rndrng_Prvdr_MI', 'Rndrng_Prvdr_Crdntls',
                 'Rndrng_Prvdr_St2', 'Rndrng_Prvdr_First_Name']

has_biz_dups = df.duplicated(subset=key2, keep=False).sum()

if has_biz_dups > 0:
    dup_group = df[df.duplicated(subset=key2, keep=False)]
    print(f"  {has_biz_dups:,} rows mein business key duplicate hai")
    print(f"\n  In duplicate rows mein null vs filled check:")
    for col in nullable_cols:
        null_in_dups = dup_group[col].isnull().sum()
        filled_in_dups = dup_group[col].notna().sum()
        if null_in_dups > 0 and filled_in_dups > 0:
            print(f"  {col:<30} null: {null_in_dups:>6,}  |  filled: {filled_in_dups:>6,}  ← merge possible!")
        else:
            print(f"  {col:<30} null: {null_in_dups:>6,}  |  filled: {filled_in_dups:>6,}")
else:
    print("  Business key duplicates nahi hain — Part 6 skip!")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY — DUPLICATE DETECTION")
print("=" * 70)
print(f"  {'Type':<45} {'Count':>12}")
print("  " + "-" * 60)
print(f"  {'1. Full row duplicates':<45} {full_dups:>12,}")
print(f"  {'2. Column duplicates (identical cols)':<45} {'check above':>12}")
print(f"  {'3. Business key (NPI+HCPCS+Place)':<45} {biz_dups:>12,}")
print(f"  {'4. NPI + HCPCS partial':<45} {df.duplicated(subset=['Rndrng_NPI','HCPCS_Cd']).sum():>12,}")
print(f"  {'5. Inconsistent (same NPI, alag info)':<45} {'check above':>12}")
print(f"  {'6. Null vs filled duplicates':<45} {'check above':>12}")

duplicate_summary = pd.DataFrame([
    {"check": "Full row duplicates", "count": int(full_dups), "note": "All 28 columns same"},
    {"check": "Business key duplicates", "count": int(biz_dups), "note": "Same NPI + HCPCS + Place"},
    {"check": "NPI + HCPCS partial duplicates", "count": int(df.duplicated(subset=['Rndrng_NPI','HCPCS_Cd']).sum()), "note": "Same doctor/service across places"},
])
duplicate_summary.to_csv(DUPLICATE_SUMMARY_PATH, index=False)
print(f"\n  ✅ Saved: {DUPLICATE_SUMMARY_PATH}")

print("\n" + "=" * 70)
print("✅ DUPLICATE DETECTION COMPLETE!")
print("=" * 70)

# ============================================================ outlier detection ke liye numeric aur categorical columns alag kar lete hain ===============================

numeric_cols = ['Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs',
                'Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt',
                'Avg_Mdcr_Pymt_Amt', 'Avg_Mdcr_Stdzd_Amt',
                'Rndrng_Prvdr_RUCA']

cat_cols = ['Rndrng_Prvdr_Last_Org_Name', 'Rndrng_Prvdr_First_Name',
            'Rndrng_Prvdr_MI', 'Rndrng_Prvdr_Crdntls', 'Rndrng_Prvdr_Ent_Cd',
            'Rndrng_Prvdr_St1', 'Rndrng_Prvdr_St2', 'Rndrng_Prvdr_City',
            'Rndrng_Prvdr_State_Abrvtn', 'Rndrng_Prvdr_State_FIPS',
            'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_RUCA_Desc',
            'Rndrng_Prvdr_Cntry', 'Rndrng_Prvdr_Type',
            'Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 'HCPCS_Cd', 'HCPCS_Desc',
            'HCPCS_Drug_Ind', 'Place_Of_Srvc']

id_col = ['Rndrng_NPI']

# ============================================================
# PART 1 — IQR METHOD
# ============================================================
print("\n" + "=" * 70)
print("PART 1 — IQR METHOD (Q1 - 1.5*IQR  |  Q3 + 1.5*IQR)")
print("=" * 70)
print(f"{'Column':<25} {'Q1':>10} {'Q3':>10} {'Lower':>10} {'Upper':>12} {'Outliers':>10} {'%':>8}")
print("-" * 90)

iqr_res = {}
for col in numeric_cols:
    s = df[col].dropna()
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    out = ((s < lo) | (s > hi)).sum()
    iqr_res[col] = {'Q1':Q1,'Q3':Q3,'IQR':IQR,'lower':lo,'upper':hi,'outliers':out,'pct':out/len(s)*100}
    print(f"{col:<25} {Q1:>10.2f} {Q3:>10.2f} {lo:>10.2f} {hi:>12.2f} {out:>10,} {out/len(s)*100:>7.2f}%")

# ============================================================
# PART 2 — Z-SCORE METHOD
# ============================================================
print("\n" + "=" * 70)
print("PART 2 — Z-SCORE METHOD (|Z| > 3)")
print("=" * 70)
print(f"{'Column':<25} {'Mean':>12} {'Std':>12} {'Outliers':>10} {'%':>8}")
print("-" * 70)

zscore_res = {}
for col in numeric_cols:
    s = df[col].dropna()
    mean, std = s.mean(), s.std()
    if std == 0: continue
    z = np.abs((s - mean) / std)
    out = (z > 3).sum()
    zscore_res[col] = {'mean':mean,'std':std,'outliers':out,'pct':out/len(s)*100}
    print(f"{col:<25} {mean:>12.2f} {std:>12.2f} {out:>10,} {out/len(s)*100:>7.2f}%")

# ============================================================
# PART 3 — PERCENTILE METHOD (99th and 99.9th)
# ============================================================
print("\n" + "=" * 70)
print("PART 3 — PERCENTILE METHOD (above 99th % = extreme)")
print("=" * 70)
print(f"{'Column':<25} {'p1':>10} {'p99':>10} {'p99.9':>10} {'Above p99':>12} {'Above p99.9':>13}")
print("-" * 83)

pct_res = {}
for col in numeric_cols:
    s = df[col].dropna()
    p1   = s.quantile(0.01)
    p99  = s.quantile(0.99)
    p999 = s.quantile(0.999)
    above99  = (s > p99).sum()
    above999 = (s > p999).sum()
    pct_res[col] = {'p1':p1,'p99':p99,'p999':p999,'above99':above99,'above999':above999}
    print(f"{col:<25} {p1:>10.2f} {p99:>10.2f} {p999:>10.2f} {above99:>12,} {above999:>13,}")

# ============================================================
# PART 4 — MODIFIED Z-SCORE (Median Based — robust)
# ============================================================
print("\n" + "=" * 70)
print("PART 4 — MODIFIED Z-SCORE (Median based, threshold > 3.5)")
print("=" * 70)
print(f"{'Column':<25} {'Median':>12} {'MAD':>12} {'Outliers':>10} {'%':>8}")
print("-" * 70)

mod_z_res = {}
for col in numeric_cols:
    s = df[col].dropna()
    median = s.median()
    mad = np.median(np.abs(s - median))
    if mad == 0: mad = 1e-10
    mod_z = 0.6745 * np.abs(s - median) / mad
    out = (mod_z > 3.5).sum()
    mod_z_res[col] = {'median':median,'mad':mad,'outliers':out,'pct':out/len(s)*100}
    print(f"{col:<25} {median:>12.2f} {mad:>12.4f} {out:>10,} {out/len(s)*100:>7.2f}%")

# ============================================================
# PART 5 — CATEGORICAL OUTLIERS
# Rare categories jo bahut kam baar aati hain
# ============================================================
print("\n" + "=" * 70)
print("PART 5 — CATEGORICAL OUTLIERS (rare values < 0.01% frequency)")
print("=" * 70)

cat_outlier_res = {}
for col in cat_cols:
    s = df[col].dropna()
    total = len(s)
    vc = s.value_counts()
    threshold = total * 0.0001
    rare = vc[vc < threshold]
    rare_count = rare.sum()
    cat_outlier_res[col] = {'rare_cats': len(rare), 'rare_rows': rare_count,
                             'pct': rare_count/total*100, 'examples': rare.head(5)}
    if len(rare) > 0:
        print(f"\n  📌 {col}")
        print(f"     Rare categories : {len(rare):,}")
        print(f"     Rare rows       : {rare_count:,}  ({rare_count/total*100:.4f}%)")
        print(f"     Examples: {dict(rare.head(5))}")

# ============================================================
# PART 6 — BUSINESS LOGIC OUTLIERS
# Domain-specific rules
# ============================================================
print("\n" + "=" * 70)
print("PART 6 — BUSINESS LOGIC OUTLIERS (domain review rules)")
print("=" * 70)

# Rule 1: Payment > Allowed amount (impossible normally)
rule1 = (df['Avg_Mdcr_Pymt_Amt'] > df['Avg_Mdcr_Alowd_Amt']).sum()
print(f"\n  Rule 1: Payment > Allowed amount (potential issue)")
print(f"          Count: {rule1:,}")

# Rule 2: Submitted charge < Medicare payment (doctor charged less than Medicare paid)
rule2 = (df['Avg_Sbmtd_Chrg'] < df['Avg_Mdcr_Pymt_Amt']).sum()
print(f"\n  Rule 2: Submitted charge < Medicare payment")
print(f"          Count: {rule2:,}")

# Rule 3: Tot_Srvcs < Tot_Benes (services less than patients — unusual)
rule3 = (df['Tot_Srvcs'] < df['Tot_Benes']).sum()
print(f"\n  Rule 3: Total services < Total patients")
print(f"          Count: {rule3:,}")

# Rule 4: RUCA value not in valid range (1-10.6 valid, 99 = special)
valid_ruca = df['Rndrng_Prvdr_RUCA'].dropna()
rule4 = ((valid_ruca > 10.6) & (valid_ruca != 99)).sum()
print(f"\n  Rule 4: RUCA value outside valid range (not 1-10.6 or 99)")
print(f"          Count: {rule4:,}")

# Rule 5: NPI not 10 digits
rule5 = (~df['Rndrng_NPI'].astype(str).str.match(r'^\d{10}$')).sum()
print(f"\n  Rule 5: NPI not exactly 10 digits")
print(f"          Count: {rule5:,}")

# Rule 6: Country not US but state is US state
rule6 = ((df['Rndrng_Prvdr_Cntry'] != 'US') & (df['Rndrng_Prvdr_State_Abrvtn'].notna())).sum()
print(f"\n  Rule 6: Non-US country but state field listed (potential issue)")
print(f"          Count: {rule6:,}")

# ============================================================
# PART 7 — EXTREME VALUE INSPECTION
# ============================================================
print("\n" + "=" * 70)
print("PART 7 — EXTREME VALUES (Top 5 & Bottom 5 per numeric col)")
print("=" * 70)

for col in numeric_cols:
    s = df[col].dropna()
    print(f"\n  📌 {col}")
    print(f"     Min    : {s.min():>15,.4f}")
    print(f"     p0.1   : {s.quantile(0.001):>15,.4f}")
    print(f"     p1     : {s.quantile(0.01):>15,.4f}")
    print(f"     Median : {s.median():>15,.4f}")
    print(f"     p99    : {s.quantile(0.99):>15,.4f}")
    print(f"     p99.9  : {s.quantile(0.999):>15,.4f}")
    print(f"     Max    : {s.max():>15,.4f}")
    print(f"     Mean   : {s.mean():>15,.4f}")
    print(f"     Std    : {s.std():>15,.4f}")

# ============================================================
# PART 8 — COMBINED SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("PART 8 — COMBINED OUTLIER SUMMARY")
print("=" * 70)
print(f"{'Column':<25} {'IQR':>10} {'Z-Score':>10} {'p99+':>10} {'Mod-Z':>10}")
print("-" * 68)
for col in numeric_cols:
    iqr_o   = iqr_res[col]['outliers']
    z_o     = zscore_res.get(col, {}).get('outliers', 0)
    p99_o   = pct_res[col]['above99']
    modz_o  = mod_z_res[col]['outliers']
    print(f"{col:<25} {iqr_o:>10,} {z_o:>10,} {p99_o:>10,} {modz_o:>10,}")

# ============================================================
# VISUALIZATION — Professional Chart
# ============================================================
print("\n🎨 Charts ban rahe hain...")

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

colors = {
    'IQR':    '#4E79A7',
    'ZScore': '#F28E2B',
    'p99':    '#E15759',
    'ModZ':   '#76B7B2'
}

col_labels = {
    'Tot_Benes':           'Tot_Benes',
    'Tot_Srvcs':           'Tot_Srvcs',
    'Tot_Bene_Day_Srvcs':  'Tot_Bene\nDay_Srvcs',
    'Avg_Sbmtd_Chrg':      'Avg_Sbmtd\nChrg',
    'Avg_Mdcr_Alowd_Amt':  'Avg_Alowd\nAmt',
    'Avg_Mdcr_Pymt_Amt':   'Avg_Pymt\nAmt',
    'Avg_Mdcr_Stdzd_Amt':  'Avg_Stdzd\nAmt',
    'Rndrng_Prvdr_RUCA':   'RUCA'
}

x = np.arange(len(numeric_cols))
labels = [col_labels[c] for c in numeric_cols]
width = 0.2

# ---- Chart 1: Outlier count by method ----
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('white')
iqr_vals  = [iqr_res[c]['outliers'] for c in numeric_cols]
z_vals    = [zscore_res.get(c,{}).get('outliers',0) for c in numeric_cols]
p99_vals  = [pct_res[c]['above99'] for c in numeric_cols]
modz_vals = [mod_z_res[c]['outliers'] for c in numeric_cols]

b1 = ax1.bar(x - 1.5*width, iqr_vals,  width, label='IQR Method',          color=colors['IQR'],    alpha=0.85)
b2 = ax1.bar(x - 0.5*width, z_vals,    width, label='Z-Score (>3)',         color=colors['ZScore'], alpha=0.85)
b3 = ax1.bar(x + 0.5*width, p99_vals,  width, label='Percentile (>p99)',    color=colors['p99'],    alpha=0.85)
b4 = ax1.bar(x + 1.5*width, modz_vals, width, label='Modified Z-Score',     color=colors['ModZ'],   alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel('Outlier Count', fontsize=11)
ax1.set_title('Outlier Count — 4 Methods Compared (Numeric Columns)', fontsize=13, fontweight='bold', pad=12)
ax1.legend(fontsize=10, loc='upper right')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v):,}'))
ax1.set_facecolor('white')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ---- Chart 2: Outlier % by IQR ----
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('white')
pcts = [iqr_res[c]['pct'] for c in numeric_cols]
bars = ax2.barh(labels, pcts, color=colors['IQR'], alpha=0.85)
for bar, pct in zip(bars, pcts):
    ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{pct:.1f}%', va='center', fontsize=9)
ax2.set_xlabel('Outlier %', fontsize=10)
ax2.set_title('IQR Outlier % per Column', fontsize=11, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# ---- Chart 3: Outlier % by Z-Score ----
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('white')
zpcts = [zscore_res.get(c,{}).get('pct',0) for c in numeric_cols]
bars3 = ax3.barh(labels, zpcts, color=colors['ZScore'], alpha=0.85)
for bar, pct in zip(bars3, zpcts):
    ax3.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{pct:.2f}%', va='center', fontsize=9)
ax3.set_xlabel('Outlier %', fontsize=10)
ax3.set_title('Z-Score Outlier % per Column', fontsize=11, fontweight='bold')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='x', alpha=0.3, linestyle='--')

# ---- Chart 4: Boxplot numeric cols ----
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor('white')
plot_cols = ['Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt', 'Avg_Mdcr_Stdzd_Amt']
plot_data  = [df[c].dropna().values for c in plot_cols]
plot_labels = ['Sbmtd_Chrg', 'Alowd_Amt', 'Pymt_Amt', 'Stdzd_Amt']
bp = ax4.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True,
                 showfliers=True, flierprops=dict(marker='.', markersize=1, alpha=0.2, color='#E15759'))
box_colors = ['#AEC6CF','#B5EAD7','#FFB7B2','#C7CEEA']
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.8)
ax4.set_title('Boxplot — Payment & Charge Columns (dots = outliers)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Amount ($)', fontsize=10)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${int(v):,}'))
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# ---- Chart 5: Heatmap — method agreement ----
ax5 = fig.add_subplot(gs[3, 0])
ax5.set_facecolor('white')
methods = ['IQR', 'Z-Score', 'p99+', 'Mod-Z']
heatmap_data = np.array([
    [iqr_res[c]['pct'] for c in numeric_cols],
    [zscore_res.get(c,{}).get('pct',0) for c in numeric_cols],
    [pct_res[c]['above99']/len(df[c].dropna())*100 for c in numeric_cols],
    [mod_z_res[c]['pct'] for c in numeric_cols]
])
im = ax5.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
ax5.set_xticks(range(len(numeric_cols)))
ax5.set_xticklabels(labels, fontsize=8, rotation=35, ha='right')
ax5.set_yticks(range(len(methods)))
ax5.set_yticklabels(methods, fontsize=10)
for i in range(len(methods)):
    for j in range(len(numeric_cols)):
        ax5.text(j, i, f'{heatmap_data[i,j]:.1f}', ha='center', va='center', fontsize=7.5)
plt.colorbar(im, ax=ax5, label='Outlier %')
ax5.set_title('Heatmap — Outlier % by Method & Column', fontsize=11, fontweight='bold')

# ---- Chart 6: Business rules violations ----
ax6 = fig.add_subplot(gs[3, 1])
ax6.set_facecolor('white')
rule_names  = ['Pymt > Allowed', 'Charge < Pymt', 'Srvcs < Benes', 'Bad RUCA', 'Bad NPI', 'Non-US + State']
rule_counts = [rule1, rule2, rule3, rule4, rule5, rule6]
bar_colors  = ['#E15759' if v > 0 else '#76B7B2' for v in rule_counts]
bars6 = ax6.barh(rule_names, rule_counts, color=bar_colors, alpha=0.85)
for bar, val in zip(bars6, rule_counts):
    ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:,}', va='center', fontsize=9)
ax6.set_xlabel('Review Count', fontsize=10)
ax6.set_title('Business Logic Review Flags', fontsize=11, fontweight='bold')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.grid(axis='x', alpha=0.3, linestyle='--')

fig.suptitle('Medicare Data — Outlier & Data Quality Review\n(8 Numeric Columns | Categorical Rare Values | Business Rules)',
             fontsize=15, fontweight='bold', y=0.995)

plt.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
print(f"✅ Charts saved: {CHART_PATH}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL OUTLIER SUMMARY — NUMERIC, CATEGORICAL & BUSINESS REVIEW")
print("=" * 70)
print("\n🔢 NUMERIC COLUMNS (8) — Statistical outliers:")
print(f"  {'Column':<25} {'IQR':>8} {'Z>3':>8} {'p99+':>8} {'ModZ':>8}  {'Severity'}")
print("  " + "-" * 72)
for col in numeric_cols:
    iqr_o  = iqr_res[col]['outliers']
    z_o    = zscore_res.get(col,{}).get('outliers',0)
    p99_o  = pct_res[col]['above99']
    modz_o = mod_z_res[col]['outliers']
    avg    = (iqr_res[col]['pct'] + zscore_res.get(col,{}).get('pct',0)) / 2
    sev    = '🔴 HIGH' if avg > 5 else ('🟡 MED' if avg > 1 else '🟢 LOW')
    print(f"  {col:<25} {iqr_o:>8,} {z_o:>8,} {p99_o:>8,} {modz_o:>8,}  {sev}")

print("\n📝 CATEGORICAL COLUMNS (19) — Rare value outliers:")
print(f"  {'Column':<30} {'Rare Cats':>10} {'Rare Rows':>12} {'%':>8}  {'Status'}")
print("  " + "-" * 72)
for col in cat_cols:
    r = cat_outlier_res.get(col, {'rare_cats':0,'rare_rows':0,'pct':0})
    status = '🟡 Has rare' if r['rare_cats'] > 0 else '🟢 Clean'
    print(f"  {col:<30} {r['rare_cats']:>10,} {r['rare_rows']:>12,} {r['pct']:>7.4f}%  {status}")

print("\n🔑 ID COLUMN (1):")
print(f"  Rndrng_NPI — Bad format: {rule5:,} rows")

print("\n⚖️  BUSINESS LOGIC:")
print(f"  Payment > Allowed   : {rule1:,}")
print(f"  Charge < Payment    : {rule2:,}")
print(f"  Services < Patients : {rule3:,}")

outlier_summary_rows = []
for col in numeric_cols:
    outlier_summary_rows.append({
        "column": col,
        "iqr_outliers": int(iqr_res[col]["outliers"]),
        "iqr_pct": round(float(iqr_res[col]["pct"]), 4),
        "zscore_outliers": int(zscore_res.get(col, {}).get("outliers", 0)),
        "zscore_pct": round(float(zscore_res.get(col, {}).get("pct", 0)), 4),
        "p99_outliers": int(pct_res[col]["above99"]),
        "modz_outliers": int(mod_z_res[col]["outliers"]),
        "modz_pct": round(float(mod_z_res[col]["pct"]), 4),
    })
pd.DataFrame(outlier_summary_rows).to_csv(OUTLIER_SUMMARY_PATH, index=False)

business_rule_summary = pd.DataFrame([
    {"rule": "Payment > Allowed", "count": int(rule1), "interpretation": "Potential issue, review before removal"},
    {"rule": "Charge < Payment", "count": int(rule2), "interpretation": "Potential issue, review before removal"},
    {"rule": "Services < Patients", "count": int(rule3), "interpretation": "Potential issue, review before removal"},
    {"rule": "Bad RUCA", "count": int(rule4), "interpretation": "Potential issue"},
    {"rule": "Bad NPI", "count": int(rule5), "interpretation": "Potential issue"},
    {"rule": "Non-US + state field", "count": int(rule6), "interpretation": "Potential issue, not automatic error"},
])
business_rule_summary.to_csv(BUSINESS_RULE_SUMMARY_PATH, index=False)

print("\n📁 SAVED SUMMARY FILES:")
print(f"  ✅ {OUTLIER_SUMMARY_PATH}")
print(f"  ✅ {BUSINESS_RULE_SUMMARY_PATH}")
print("\nNOTE:")
print("  Outliers review flags hain. Medicare payment/service data naturally skewed hota hai,")
print("  isliye high values ko blindly remove nahi karna chahiye.")

print("\n" + "=" * 70)
print("✅ OUTLIER DETECTION COMPLETE — Chart saved!")
print("=" * 70)