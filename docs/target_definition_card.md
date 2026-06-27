# Target Definition Card

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

denied = 1 when payment_to_charge_ratio <= 0.63484673

denied = 0 otherwise

## Dataset Impact
Rows evaluated: 9,781,673

Synthetic denied rate: 20.0000%

## Modeling Safety Rule
Do not train the model using columns that directly create or closely mirror
the target. Step 08 must exclude the leakage-risk columns listed in
modeling_feature_policy.csv.

## Buyer-Facing Language
Call this a denial-risk proxy model, not a real CMS denial classifier, until
real payer denial labels or remittance codes are connected.
