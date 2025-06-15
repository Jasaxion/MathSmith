import os
import re
import json
from datasets import load_dataset
from tqdm import tqdm

# 输出目录
os.makedirs("./open_data_problem", exist_ok=True)

# 数据集配置列表: (dataset_name, config_name, split, field)
DATASETS = [
    ("Maxwell-Jia/AIME_2024", None, "train", "Problem"),
    ("opencompass/AIME2025", "AIME2025-I", "test", "question"),
    ("opencompass/AIME2025", "AIME2025-II", "test", "question"),
    ("openai/gsm8k", "main", "train", "question"),
    ("HuggingFaceH4/MATH-500", None, "test", "problem"),
    ("xl-zhao/PromptCoT-DS-Dataset", "default", "train", "prompt"),
    ("meta-math/MetaMathQA", "default", "train", "query"),
    ("nvidia/OpenMathInstruct-2", "default", "train", "problem"),
    ("AI-MO/NuminaMath-CoT", "default", "train", "problem"),
    ("microsoft/orca-math-word-problems-200k", "default", "train", "question"),
    ("ToheartZhang/JiuZhang3.0-Corpus-SFT", "default", "train", "input"),
]

def extract_prompt_problem(prompt: str) -> str:
    # 替换全角竖线为半角
    prompt = prompt.replace("｜", "|")
    # 正则提取 <|User|> 和 <|Assistant|> 之间内容
    match = re.search(r"<\|User\|>(.*?)<\|Assistant\|>", prompt, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_prompt_problem_from_input(prompt: str) -> str:
    match = re.search(r"## Question\s*(.*?)\s*## Solution", prompt, re.DOTALL)
    return match.group(1).strip() if match else None

for dataset_name, config, split, field in DATASETS:
    tag = dataset_name.split("/")[-1]
    if config:
        tag += f"_{config}"
    output_path = f"./open_data_problem/{tag}_problem.jsonl"

    print(f"🔍 Processing {dataset_name} ({config or 'no config'})...")

    try:
        dataset = (
            load_dataset(dataset_name, name=config, split=split)
            if config else
            load_dataset(dataset_name, split=split)
        )

        with open(output_path, "w", encoding="utf-8") as f_out:
            for item in tqdm(dataset):
                if field == "prompt":
                    raw_prompt = item.get("prompt", "")
                    problem = extract_prompt_problem(raw_prompt)
                    if not problem:
                        print(f"[WARN] No problem found in prompt:\n{raw_prompt}\n")
                elif field == "input":
                    raw_prompt = item.get("input", "")
                    problem = extract_prompt_problem_from_input(raw_prompt)
                else:
                    problem = item.get(field, "")
                if problem:
                    f_out.write(json.dumps({"problem": problem}) + "\n")

        print(f"✅ Saved to {output_path}")
    except Exception as e:
        print(f"❌ Error processing {dataset_name} ({config or 'no config'}): {e}")
