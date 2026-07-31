import json
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


# =====================================================
# GPT 5.5
# =====================================================
client = OpenAI(
api_key="Your KEY"
)
MODEL = "gpt-5.5"


# =====================================================
# PATH
# =====================================================

DATA_DIR = Path("test-data") / "#1"
OUTPUT_DIR = Path("test/test-results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "test#1.csv"


# =====================================================
# PROMPT
# =====================================================


SYSTEM_PROMPT ="""
You are an expert evidence synthesis researcher.
Your task is to extract outcome data from randomized controlled trial (RCT) articles.
You must return ONLY valid JSON. No explanation and no extra text.
-----------------------
OUTPUT FORMAT
-----------------------
{
  "outcome": "",
  "intervention_n": "",
  "intervention_event": "",
  "control_n": "",
  "control_event": ""
}
-----------------------
OUTCOME DEFINITION
-----------------------
The outcome measure you need to extract is the Objective Response Rate, which is defined as:
ORR=CR (Complete Response)+PR (Partial Response).
If efficient-related outcome indicators are reported but do not match what I requested, do not extract them.
-----------------------
TASK 1: SAMPLE SIZE EXTRACTION
-----------------------
Extract the total sample sizes for the outcome in both the intervention arm and the control arm.
-----------------------
TASK 2: EVENT COUNT EXTRACTION
-----------------------
Extract the event counts of the outcome after intervention for both the intervention arm and the control arm.
"""


def extract_from_md(md_text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": md_text
            }
        ]
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except Exception:
        print("JSON Parse Error")
        print(content)

        return {
            "outcome": "",
            "intervention_n": "",
            "intervention_event": "",
            "control_n": "",
            "control_event": ""
        }


# =====================================================
# MAIN
# =====================================================

results = []

md_files = sorted(DATA_DIR.glob("*.md"))

for md_file in tqdm(md_files):

    study_id = md_file.stem

    print(f"Processing {study_id}")

    md_text = md_file.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    result = extract_from_md(md_text)

    results.append({
        "study_id": study_id,
        "cot": result.get("cot", ""),
        "outcome": result.get("outcome", ""),
        "intervention_n": result.get("intervention_n", ""),
        "intervention_event": result.get("intervention_event", ""),
        "control_n": result.get("control_n", ""),
        "control_event": result.get("control_event", "")
    })

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nFinished.")
print(f"Saved to: {OUTPUT_FILE.resolve()}")