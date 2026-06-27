"""
STEP 10A - Prepare Official RARC Code CSV
WellMind Data Solutions - Claim Denial Prediction System

Run:
    python src/step10a_prepare_real_rarc_codes.py path_to_x12_copied_text.txt

Purpose:
Parse copied text from the X12 Remittance Advice Remark Codes page into
outputs/nlp/real_rarc_codes.csv for Step 10.

Expected output columns:
    code, description, start_date, last_modified, notes

Important:
This script does not download code lists. It only converts text that you have
already copied from the official X12 page into a local CSV artifact.
"""

from pathlib import Path
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "nlp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "real_rarc_codes.csv"

CODE_LINE_RE = re.compile(r"^(MA\d+|M\d+|N\d+)\s+(.+)$")
START_RE = re.compile(r"^Start:\s*(.+)$")
LAST_MODIFIED_RE = re.compile(r"^Start:\s*.+?\|\s*Last Modified:\s*(.+)$")
NOTES_RE = re.compile(r"^Notes:\s*(.+)$")


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_rarc_text(raw_text):
    rows = []
    current = None

    for raw_line in raw_text.splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue

        match = CODE_LINE_RE.match(line)
        if match:
            if current:
                rows.append(current)
            current = {
                "code": match.group(1).upper(),
                "description": match.group(2).strip(),
                "start_date": "",
                "last_modified": "",
                "notes": "",
            }
            continue

        if current is None:
            continue

        start_match = START_RE.match(line)
        if start_match:
            current["start_date"] = start_match.group(1).split("|")[0].strip()
            modified_match = LAST_MODIFIED_RE.match(line)
            if modified_match:
                current["last_modified"] = modified_match.group(1).strip()
            continue

        notes_match = NOTES_RE.match(line)
        if notes_match:
            current["notes"] = notes_match.group(1).strip()
            continue

        if not line.startswith(("Home", "Products", "Start:", "Notes:")):
            current["description"] = normalize_text(f"{current['description']} {line}")

    if current:
        rows.append(current)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No RARC codes were parsed. Check that the input text contains lines like 'M19 Missing ...'.")

    df = df.drop_duplicates("code").sort_values("code").reset_index(drop=True)
    return df


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python step10a_prepare_real_rarc_codes.py path_to_x12_copied_text.txt"
        )

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw_text = input_path.read_text(encoding="utf-8", errors="replace")
    df = parse_rarc_text(raw_text)
    df.to_csv(OUTPUT_PATH, index=False)

    print("STEP 10A COMPLETE")
    print(f"Parsed codes : {len(df):,}")
    print(f"Saved CSV    : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
