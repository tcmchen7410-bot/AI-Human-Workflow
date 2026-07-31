import pandas as pd
from pathlib import Path
from bert_score import score
from tqdm import tqdm


# ==============================
# 路径
# ==============================

DATA_DIR = Path(__file__).parent

INPUT_FILE = DATA_DIR / "Inconsistent.csv"

OUTPUT_FILE = DATA_DIR / "bert_score_inconsistent.csv"


# ==============================
# 读取数据
# ==============================

df = pd.read_csv(INPUT_FILE)

print(df.head())


# ==============================
# 构建GPT-DeepSeek配对
# ==============================

ids = []
references = []
candidates = []


for ID, group in tqdm(df.groupby("ID")):

    gpt_text = group.loc[
        group["modal"].str.lower()=="gpt",
        "CoT"
    ].values


    deepseek_text = group.loc[
        group["modal"].str.lower()=="deepseek",
        "CoT"
    ].values


    if len(gpt_text)==0 or len(deepseek_text)==0:
        continue


    ids.append(ID)

    # GPT reference
    references.append(gpt_text[0])

    # DeepSeek candidate
    candidates.append(deepseek_text[0])


print("有效配对数量:", len(ids))


# ==============================
# 一次性计算BERTScore
# ==============================

P, R, F1 = score(
    candidates,
    references,
    model_type="roberta-large",
    lang="en",
    batch_size=8,
    verbose=True
)


# ==============================
# 保存结果
# ==============================

result_df = pd.DataFrame({

    "ID": ids,

    "BERT_Precision": P.tolist(),

    "BERT_Recall": R.tolist(),

    "BERT_F1": F1.tolist()

})


result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("完成！")

print(result_df.head())

print(
    f"结果保存至: {OUTPUT_FILE}"
)