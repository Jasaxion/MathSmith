import json
import random
import re
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# configure dataset file paths (from opendatasets or local files)
file_paths = {
    "AIME_2024": "",
    "AIME2025-I": "",
    "AIME2025-II": "",
    "gsm8k_main": "",
    "MATH-500": "",
    "PromptCoT": ""
}

vllm_api_url = "http://localhost:8000/v1/completions" # setup a vllm server for this script
model_name = "Qwen/Qwen3-30B-A3B"
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def load_problems(file_path, n=10):
    problems = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "problem" in obj:
                problems.append(obj["problem"])
    return random.sample(problems, min(n, len(problems)))

def build_prompt(problem):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

def query_vllm(prompts):
    payload = {
        "prompt": list(prompts),
        "max_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "n": 1
    }
    response = requests.post(vllm_api_url, json=payload, timeout=3000)
    response.raise_for_status()
    result = response.json()
    choices = result.get("choices", [])
    outputs = []
    output_list = []
    for j in range(1):
        choice = choices[j]
        text = choice.get("text", "").strip()
        output_list.append(text)
    outputs.append(output_list)
    return outputs

def extract_think_content(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""

def count_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False))

result = {}

for name, path in file_paths.items():
    print(f"\nProcessing dataset: {name}")
    try:
        problems = load_problems(path)
        print(len(problems), "problems loaded.")
    except Exception as e:
        print(f"Error loading {path}: {e}")
        result[name] = []
        continue

    think_lengths = []
    prompts = [build_prompt(prob) for prob in problems]
    try:
        print("len(prompt):", len(prompts))
        output_list = query_vllm(prompts)
        for output in output_list:
            think_text = extract_think_content(output[0])
            token_len = count_tokens(think_text)
            think_lengths.append(token_len)
    except Exception as e:
        print(f"Error on a problem: {e}")
        think_lengths.append(0)

    result[name] = think_lengths

plt.figure(figsize=(10, 6))
for name, lengths in result.items():
    if lengths:
        plt.plot(lengths, label=f"{name} (avg={sum(lengths)/len(lengths):.1f})", marker='o')

plt.xlabel("Sample Index")
plt.ylabel("Token Length of <think>")
plt.title("Token Length Distribution of <think> Responses per Dataset")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("think_token_length_distribution.pdf", dpi=300)
