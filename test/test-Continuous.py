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

DATA_DIR = Path("test-data") / "#328"
OUTPUT_DIR = Path("test/test-results")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "test#328.csv"

# =====================================================
# PROMPT
# =====================================================


SYSTEM_PROMPT ="""
You are an expert evidence synthesis researcher.
Your task is to extract outcome data from randomized controlled trial (RCT) articles.
Think step-by-step before answering.
⸻
OUTPUT FORMAT
Return ONLY valid JSON.
{
  "outcome": "",
  "intervention_mean": "",
  "intervention_sd": "",
  "intervention_n": "",
  "control_mean": "",
  "control_sd": "",
  "control_n": ""
}
⸻
COT INSTRUCTIONS
In the “cot” field:
1. Identify CD4+ level, outcomes in the article and check if any strictly match the requested definition. If none match, prepare to return null.
2. Locate the total sample sizes (n) for both the intervention and control arms for the matched outcome.
3. Locate the mean values of the outcome after intervention for both the intervention and control arms.
4. Locate the standard deviations (SDs) of the outcome after intervention for both the intervention and control arms.
Use concise reasoning (3–10 sentences).
⸻
OUTCOME DEFINITION
The outcome measure you need to extract is the CD4+ level.
Extract the CD4+ level outcome reported after intervention.
If immunoglobulin-related outcome indicators are reported but do not match CD4+ level (e.g., IgA, IgG, IgE, total immunoglobulin, or other immune markers), do not extract them.
return
{
“cot”: “…”,
“outcome”: null,
“intervention_n”: null,
“intervention_event”: null,
“control_n”: null,
“control_event”: null
}
⸻
TASK 1: SAMPLE SIZE EXTRACTION
-----------------------
Extract the total sample sizes for the outcome in both the intervention arm and the control arm.
-----------------------
TASK 3: MEAN EXTRACTION
-----------------------
Extract the mean value of the outcome after intervention for both the intervention arm and the control arm.
TASK 4: STANDARD DEVIATION (SD) EXTRACTION
-----------------------
Extract the standard deviation (SD) of the outcome after intervention for both the intervention arm and the control arm.
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
            "intervention_mean": "",
            "intervention_sd": "",
            "intervention_n": "",
            "control_mean": "",
            "control_sd": "",
            "control_n": ""
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
    "intervention_mean": result.get("intervention_mean", ""),
    "intervention_sd": result.get("intervention_sd", ""),
    "intervention_n": result.get("intervention_n", ""),
    "control_mean": result.get("control_mean", ""),
    "control_sd": result.get("control_sd", ""),
    "control_n": result.get("control_n", "")
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