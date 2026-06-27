"""Small smoke-test client for the FastAPI demo.

Run after starting the API:
    python sample_api_request.py
"""

import json

import requests


API_URL = "http://localhost:8000/predict/full"

payload = {
    "provider_type": "Internal Medicine",
    "state": "CA",
    "ruca_code": 1.0,
    "entity_type": 1,
    "participating": 1,
    "credential_group": "Physician",
    "hcpcs_code": "99214",
    "drug_indicator": 0,
    "place_of_service": 0,
    "total_services_log": 4.8,
    "total_beneficiaries_log": 4.4,
    "avg_submitted_charge_log": 5.2,
    "avg_medicare_payment_log": 4.3,
    "avg_allowed_amount_log": 4.5,
    "flag_charge_lt_payment": 0,
    "flag_services_lt_benes": 0,
    "flag_zero_payment": 0,
    "flag_zero_allowed": 0,
    "flag_invalid_state": 0,
    "flag_non_us_country": 0,
    "remark_text": "Claim denied because prior authorization was missing.",
}


def main():
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()
