import json
import re
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List
from langchain_openai import ChatOpenAI


# =====================================================
# 1. 路径配置
# =====================================================
# OUTPUT_DIR = Path("results")
DATA_DIR = Path("../data") / "#323"
OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / ("#1不对.csv")
# =====================================================
# 2. LLM 初始化
# =====================================================

gpt = ChatOpenAI(
    model="gpt-5.5",
    api_key="Your KEY",
    temperature=1
)

deepseek = ChatOpenAI(
    model="deepseek-v4-pro",
    base_url="https://api.deepseek.com",
    api_key="Your KEY",
    temperature=1
)


# =====================================================
# 3. CoT Prompt
# =====================================================

EXTRACTION_PROMPT = """
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
1. Identify pain intensity, outcomes in the article and check if any strictly match the requested definition. If none match, prepare to return null.
2. Locate the total sample sizes (n) for both the intervention and control arms for the matched outcome.
3. Locate the mean values of the outcome after intervention for both the intervention and control arms.
4. Locate the standard deviations (SDs) of the outcome after intervention for both the intervention and control arms.
Use concise reasoning (3–10 sentences).
⸻
OUTCOME DEFINITION
The outcome measure you need to extract is the pain intensity.
Extract the pain intensity reported after intervention.
If pain intensity-related outcome indicators are reported but do not match, do not extract them.
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
# =====================================================
# 4. LLM 提取（time + tokens）
# =====================================================

def extract(llm, text: str) -> Dict:

    prompt = EXTRACTION_PROMPT + "\n\nTEXT:\n" + text

    start_time = time.time()
    res = llm.invoke(prompt)
    elapsed_time = round(time.time() - start_time, 2)

    content = res.content
    usage = getattr(res, "usage_metadata", {}) or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    try:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise ValueError("No JSON found")

        data = json.loads(match.group())

    except Exception as e:
        print("JSON parse error:", e)

        data = {
            "cot": "",
            "outcome": "",
            "intervention_mean": "",
            "intervention_sd": "",
            "intervention_n": "",
            "control_mean": "",
            "control_sd": "",
            "control_n": ""
        }

    data["time"] = elapsed_time
    data["input_tokens"] = input_tokens
    data["output_tokens"] = output_tokens
    data["total_tokens"] = total_tokens

    return data


# =====================================================
# 5. 清洗
# =====================================================

def clean(x):
    if x is None:
        return ""
    x = str(x).strip().lower()
    if x in ["none", "null", "nan"]:
        return ""
    return x


# =====================================================
# 6. 一致性判断
# =====================================================

def is_consistent(a: Dict, b: Dict) -> bool:
    keys = [
        "intervention_mean",
        "intervention_sd",
        "intervention_n",
        "control_mean",
        "control_sd",
        "control_n"
    ]

    for k in keys:
        va, vb = clean(a.get(k)), clean(b.get(k))

        if va == "" and vb == "":
            continue
        if va != vb:
            return False

    return True


# =====================================================
# 7. 人工仲裁（关键：返回 human_time）
# =====================================================

def human_agent(gpt_out, deep_out, text):

    print("\n=========== HUMAN REVIEW ===========\n")
    print(text[:1200])

    print("\n--- GPT ---")
    print(json.dumps(gpt_out, indent=2, ensure_ascii=False))

    print("\n--- DEEPSEEK ---")
    print(json.dumps(deep_out, indent=2, ensure_ascii=False))

    start = time.time()

    choice = input("\nFinal choice (gpt / deepseek / manual): ").strip()

    if choice == "gpt":
        result = gpt_out
    elif choice == "deepseek":
        result = deep_out
    else:
        result = {
            "cot": "",
            "outcome": input("outcome: "),
            "intervention_mean": input("intervention_mean: "),
            "intervention_sd": input("intervention_sd: "),
            "intervention_n": input("intervention_n: "),
            "control_mean": input("control_mean: "),
            "control_sd": input("control_sd: "),
            "control_n": input("control_n: ")
        }

    human_time = round(time.time() - start, 2)

    result["human_time"] = human_time

    return result


# =====================================================
# 8. 读取 MD
# =====================================================

def load_md_files(data_dir: Path):

    texts, names = [], []

    for file in sorted(data_dir.rglob("*.md")):
        texts.append(file.read_text(encoding="utf-8"))
        names.append(file.name)

    return texts, names


# =====================================================
# 9. pipeline（关键修复）
# =====================================================

def run_pipeline(text: str) -> Dict:

    gpt_out = extract(gpt, text)
    deep_out = extract(deepseek, text)

    human_time = 0   # ⭐ 默认没有人工介入

    if is_consistent(gpt_out, deep_out):
        final = gpt_out
        agent = "consensus"
    else:
        final = human_agent(gpt_out, deep_out, text)
        agent = "human_override"

        # ⭐ 只有发生人工才记录
        human_time = final.get("human_time", 0)

    return {
        "gpt": gpt_out,
        "deepseek": deep_out,
        "final": final,
        "agent": agent,

        # ⭐ 关键修复：必须显式返回
        "human_time": human_time
    }

# =====================================================
# 10. batch（修复 human_time 统计）
# =====================================================

def run_batch(texts: List[str], names: List[str]) -> pd.DataFrame:

    results = []

    for i, text in enumerate(texts):

        print(f"\nProcessing {i+1}/{len(texts)}: {names[i]}")

        res = run_pipeline(text)
        file_id = Path(names[i]).stem

        # =========================
        # GPT
        # =========================
        results.append({
            "id": file_id,
            "model": "gpt",
            "outcome": res["gpt"].get("outcome", ""),
            "cot": res["gpt"].get("cot", ""),

            "intervention_mean": res["gpt"].get("intervention_mean", ""),
            "intervention_sd": res["gpt"].get("intervention_sd", ""),
            "intervention_n": res["gpt"].get("intervention_n", ""),
            "control_mean": res["gpt"].get("control_mean", ""),
            "control_sd": res["gpt"].get("control_sd", ""),
            "control_n": res["gpt"].get("control_n", ""),

            "time": res["gpt"].get("time", 0),
            "human_time": 0,

            "input_tokens": res["gpt"].get("input_tokens", 0),
            "output_tokens": res["gpt"].get("output_tokens", 0),
            "total_tokens": res["gpt"].get("total_tokens", 0)
        })

        # =========================
        # DeepSeek
        # =========================
        results.append({
            "id": file_id,
            "model": "deepseek",
            "outcome": res["deepseek"].get("outcome", ""),
            "cot": res["deepseek"].get("cot", ""),

            "intervention_mean": res["deepseek"].get("intervention_mean", ""),
            "intervention_sd": res["deepseek"].get("intervention_sd", ""),
            "intervention_n": res["deepseek"].get("intervention_n", ""),
            "control_mean": res["deepseek"].get("control_mean", ""),
            "control_sd": res["deepseek"].get("control_sd", ""),
            "control_n": res["deepseek"].get("control_n", ""),

            "time": res["deepseek"].get("time", 0),
            "human_time": 0,

            "input_tokens": res["deepseek"].get("input_tokens", 0),
            "output_tokens": res["deepseek"].get("output_tokens", 0),
            "total_tokens": res["deepseek"].get("total_tokens", 0)
        })

        # =========================
        # FINAL
        # =========================
        results.append({
            "id": file_id,
            "model": "final_" + res["agent"],
            "outcome": res["final"].get("outcome", ""),
            "cot": res["final"].get("cot", ""),

            "intervention_mean": res["final"].get("intervention_mean", ""),
            "intervention_sd": res["final"].get("intervention_sd", ""),
            "intervention_n": res["final"].get("intervention_n", ""),
            "control_mean": res["final"].get("control_mean", ""),
            "control_sd": res["final"].get("control_sd", ""),
            "control_n": res["final"].get("control_n", ""),

            "time":
                float(res["gpt"].get("time", 0))
                + float(res["deepseek"].get("time", 0)),

            "human_time": res.get("human_time", 0),

            "input_tokens":
                int(res["gpt"].get("input_tokens", 0))
                + int(res["deepseek"].get("input_tokens", 0)),

            "output_tokens":
                int(res["gpt"].get("output_tokens", 0))
                + int(res["deepseek"].get("output_tokens", 0)),

            "total_tokens":
                int(res["gpt"].get("total_tokens", 0))
                + int(res["deepseek"].get("total_tokens", 0))
        })

    return pd.DataFrame(results)[
        [
            "id",
            "model",
            "outcome",
            "cot",
            "intervention_mean",
            "intervention_sd",
            "intervention_n",
            "control_mean",
            "control_sd",
            "control_n",
            "time",
            "human_time",
            "input_tokens",
            "output_tokens",
            "total_tokens"
        ]
    ]


# =====================================================
# 11. 保存 CSV
# =====================================================

def save_csv(df: pd.DataFrame, output_file: Path):

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {output_file}")


# =====================================================
# 12. 主程序
# =====================================================

if __name__ == "__main__":

    texts, names = load_md_files(DATA_DIR)

    df = run_batch(texts, names)

    save_csv(df, OUTPUT_FILE)