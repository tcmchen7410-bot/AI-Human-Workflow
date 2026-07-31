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
DATA_DIR = Path("../data") / "#1"
OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / ("#111确.csv")
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
EXAMPLE 1
方法
临床疗效按照国际通用RECIST标准进行评价。
客观缓解率（ORR）定义为：完全缓解（CR）与部分缓解（PR）例数之和占总例数的百分比。
共纳入80例患者：
* 康莱特注射液联合化疗组40例
* 单纯化疗组40例
结果
治疗后，康莱特注射液联合化疗组40例中，完全缓解（CR）10例，部分缓解（PR）14例，稳定（SD）11例，进展（PD）5例，客观缓解（CR+PR）为24例；化疗组40例中，CR 4例，PR 8例，SD 18例，PD 10例，客观缓解（CR+PR）为12例。
Output
{
"outcome": "客观缓解率",
"intervention_n": "40",
"intervention_event": "24",
"control_n": "40",
"control_event": "12"
}
EXAMPLE 2
方法
近期疗效参照固体瘤疗效评价标准评价。
客观缓解率（ORR）等于完全缓解（CR）例数加部分缓解（PR）例数。
患者随机分为：
* 康莱特注射液联合化疗组56例
* 化疗组54例
结果
| 组别 | n | CR(例) | PR(例) | SD(例) | PD(例) |
|---|---|---|---|---|---|
| 康莱特+化疗组 | 56 | 12 | 20 | 18 | 6 |
| 化疗组 | 54 | 6 | 12 | 22 | 14 |
Output
{
"outcome": "客观缓解率",
"intervention_n": "56",
"intervention_event": "32",
"control_n": "54",
"control_event": "18"
}
EXAMPLE 3
方法
观察并记录两组患者治疗后的临床疗效。
患者随机分为：
* 康莱特注射液联合化疗组50例
* 化疗组50例
结果
| 组别 | n | 临床有效(例) |
|---|---|---|
| 康莱特+化疗组 | 50 | 38 |
| 化疗组 | 50 | 25 |
Output
{
"outcome": "",
"intervention_n": "",
"intervention_event": "",
"control_n": "",
"control_event": ""
}
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
            "intervention_n": "",
            "intervention_event": "",
            "control_n": "",
            "control_event": ""
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

    keys = ["intervention_n", "intervention_event", "control_n", "control_event"]

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
            "intervention_n": input("intervention_n: "),
            "intervention_event": input("intervention_event: "),
            "control_n": input("control_n: "),
            "control_event": input("control_event: ")
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

            "intervention_event": res["gpt"].get("intervention_event", ""),
            "intervention_n": res["gpt"].get("intervention_n", ""),
            "control_event": res["gpt"].get("control_event", ""),
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

            "intervention_event": res["deepseek"].get("intervention_event", ""),
            "intervention_n": res["deepseek"].get("intervention_n", ""),
            "control_event": res["deepseek"].get("control_event", ""),
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

            "intervention_event": res["final"].get("intervention_event", ""),
            "intervention_n": res["final"].get("intervention_n", ""),
            "control_event": res["final"].get("control_event", ""),
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
            "intervention_event",
            "intervention_n",
            "control_event",
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