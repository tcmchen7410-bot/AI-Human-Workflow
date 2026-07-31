#/Users/xiaobanxia/PycharmProjects/PythonProject5/.venv/bin/pip install openai pandas scikit-learn tqdm
#终端运行
#终端：第一步--export OPENAI_API_KEY="your openai key"
#第二步：python Similarity/"text 3 large.py"

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. OpenAI配置
# ==========================================



api_key = os.getenv("OPENAI_API_KEY")#这里不要进行任何形式的删改

if not api_key:

    raise ValueError(

        "❌ 未检测到 OPENAI_API_KEY，请先设置环境变量"

    )

client = OpenAI(

    api_key=api_key

)


# ==========================================
# 2. 文件路径
# ==========================================


DATA_DIR = Path(__file__).parent

# 定义输入文件路径
INPUT_FILE = DATA_DIR / "Inconsistent.csv"

# 定义输出文件路径
OUTPUT_FILE = DATA_DIR / "openai_embedding_inconsistent.csv"


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到文件:\n{INPUT_FILE.resolve()}"
    )


# ==========================================
# 3. 读取数据
# ==========================================

print("正在读取数据...")

df = pd.read_csv(INPUT_FILE)


for col in [
    "ID",
    "modal",
    "CoT"
]:

    df[col] = (
        df[col]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.strip()
    )


# ==========================================
# 4. 整理GPT和DeepSeek文本
# ==========================================

pairs = []


for id_val, group in df.groupby("ID"):

    gpt_texts = (
        group[group["modal"]=="gpt"]
        ["CoT"]
        .tolist()
    )

    deepseek_texts = (
        group[group["modal"]=="deepseek"]
        ["CoT"]
        .tolist()
    )


    for gpt, ds in zip(
        gpt_texts,
        deepseek_texts
    ):

        pairs.append({

            "ID": id_val,

            "GPT_CoT": gpt,

            "DeepSeek_CoT": ds

        })


print(
    f"共发现 {len(pairs)} 对 CoT"
)


# ==========================================
# 5. 批量embedding
# ==========================================

all_texts = []


for item in pairs:

    all_texts.append(
        item["GPT_CoT"]
    )

    all_texts.append(
        item["DeepSeek_CoT"]
    )


print("正在调用 text-embedding-3-large...")


response_vectors = []


batch_size = 100


for i in tqdm(
    range(
        0,
        len(all_texts),
        batch_size
    )
):

    batch = all_texts[
        i:i+batch_size
    ]


    response = client.embeddings.create(

        model="text-embedding-3-large",

        input=batch

    )


    vectors = [
        item.embedding
        for item in response.data
    ]


    response_vectors.extend(
        vectors
    )



# ==========================================
# 6. 计算cosine similarity
# ==========================================

print("正在计算相似度...")


results = []


for idx, item in enumerate(pairs):


    gpt_vec = response_vectors[
        idx*2
    ]

    ds_vec = response_vectors[
        idx*2+1
    ]


    sim = cosine_similarity(

        [gpt_vec],

        [ds_vec]

    )[0][0]


    results.append({

        "ID": item["ID"],

        "GPT_CoT": item["GPT_CoT"],

        "DeepSeek_CoT": item["DeepSeek_CoT"],

        "OpenAI_embedding_similarity":
            round(float(sim),4)

    })



# ==========================================
# 7. 保存
# ==========================================

result_df = pd.DataFrame(results)


result_df.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8-sig"

)


print("\n"+"="*50)

print("✅ 计算完成")

print(
    f"结果保存:\n{OUTPUT_FILE.resolve()}"
)

print("="*50)


print("\n前5行:")

print(
    result_df[
        [
            "ID",
            "OpenAI_embedding_similarity"
        ]
    ].head()
)
