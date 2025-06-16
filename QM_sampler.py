# 问题生成Prompt 模型生成问题，vllm 版本

from transformers import AutoTokenizer
import json
import os
import logging
import random
from datetime import datetime
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm

MAX_GEN = 100000
BATCH_SIZE = 1000
TENSOR_PARALLER_SIZE = 1
MAX_N = 1
model_list = [
    "/data/zhansx/Mathsmith-model/MathSmith-Qwen3-8B-add_30_percent_consist-Step100/MathSmith-Qwen3-8B-add_30_percent_consist-Step100",
]
SOURCE_DIR = "./data_collect/sampled_concept/collect_planetmath_grouped_deduplicated.jsonl"
SAMPLE_NUM = 5

# logging configuration
today_str = datetime.now().strftime("%m-%d-%Y %H:%M:%S") 
LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, f"QM-{today_str}.log")
from vllm import LLM, SamplingParams

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

QUESTION_INSTRUCTION = """
Given the Concepts and Explanations along with the instructions below, develop a **single challenging mathematics problem** suitable for advanced Olympiads

Instructions:
A. Select Relevant Concepts: 
  1. Carefully analyze the provided concepts and their explanations.
  2. Based on this analysis, choose a suitable subset of concepts that can be naturally and meaningfully integrated into a cohesive, well-focused mathematics problem.

B. Problem Difficulty Rules:
  1. The problem should require deep insight and non-obvious reasoning.
  2. It must include **at least two** of the following difficulty features:
    a. Multi-step Reasoning: Requires multiple sequential logical steps.
    b. Cross-topic Integration: Combines distinct mathematical topics.
    c. Implicit or Reverse Logic: Includes hidden conditions or reverse deduction.
    d. Distractors: Contains misleading or extraneous conditions.
    e. Abstract Modeling: Translates complex scenarios into mathematical form.
    f. Multiple Solution Paths: Allows various non-trivial solving methods.
    g. Advanced Manipulation: Necessitates sophisticated algebraic or geometric transformations.
    h. Extreme Conditions: Focuses on limits or boundary values.
    i. Non-standard Representation: Uses unconventional presentation of familiar concepts.

C. Output Format:
    Please strictly format your response as follows:
    
    <rationale>
    Step-by-step reasoning describing:
      - How the selected concepts are connected;
      - How they contribute to the structure and difficulty of the problem;
      - Which difficulty features (from B.2) are used, and how they are embedded.
    </rationale>
    
    <problem>
    [Write a single self-contained, high-level Olympiad mathematics problem here.]
    </problem>

Given concept and explanation below:
"""

def sample_concept_data(concept_data, sample_size: int):
    if sample_size > len(concept_data):
        raise ValueError(f"sample_size ({sample_size}) is greater than total number of samples ({len(concept_data)})")
    sampled_lines = random.sample(concept_data, sample_size)
    sampled_data = [json.loads(line) for line in sampled_lines]
    return sampled_data

def load_concept_data(source_file: str) -> list:
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def save_rationale_data(c_and_e, rationale_data, output_file: str):
    rationale_match = re.search(r"<rationale>(.*?)</rationale>", rationale_data, re.DOTALL)
    problem_match = re.search(r"<problem>(.*?)</problem>", rationale_data, re.DOTALL)
    if not rationale_match:
        logging.warning("⚠️Rationale not found in the response, skipping this entry.")
        return 
        # raise ValueError("Rationale not found in the response")
    if not problem_match:
        logging.warning("⚠️Problem not found in the response, skipping this entry.")
        return
        # raise ValueError("Problem not found in the response")
    entry = {
        "Sampled_concept": c_and_e,
        "Rationale": rationale_match.group(1).strip(),
        "Problem": problem_match.group(1).strip(),
        "Output": rationale_data,
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

if __name__ == "__main__":
    concept_data = load_concept_data(SOURCE_DIR)
    logging.info(f"The size of the concept dataset to be loaded: {len(concept_data)}")
    for model_name in model_list:
        logging.info(f"🤖Loading Model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        llm = LLM(model=model_name, 
                  trust_remote_code=True, 
                  tensor_parallel_size=TENSOR_PARALLER_SIZE,
                  gpu_memory_utilization=0.9,
                  dtype="float16")
        logging.info("Warming up vLLM...")
        llm.generate(prompts=["Hello, world!"], sampling_params=SamplingParams(max_tokens=10))

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
            max_tokens=8192,
            n=MAX_N
        )
        gen_num = 0
        with tqdm(total=MAX_GEN, desc=f"Generate mathematical problems ({os.path.basename(model_name)})") as pbar:
            while True:
                if gen_num >= MAX_GEN:
                    break
                batch_prompt = []
                c_and_e_list = []
                for i in range(BATCH_SIZE):
                    sampled_data = sample_concept_data(concept_data, SAMPLE_NUM)
                    idx = 1
                    c_and_e = ""
                    for idx in range(1, SAMPLE_NUM + 1):
                        concept_text = sampled_data[idx - 1]["Concept"]
                        explanation_text = sampled_data[idx - 1]["Explanation"]
                        merged_concept = str(idx) + ". " + concept_text + ": " + explanation_text + "\n"
                        c_and_e += merged_concept

                    # full_prompt = QUESTION_INSTRUCTION.replace("{CONCEPT_AND_EXPLANATION}", c_and_e)
                    req_prompt = tokenizer.apply_chat_template(
                        [{"role": "system", "content": QUESTION_INSTRUCTION},
                        {"role": "user", "content": c_and_e}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    batch_prompt.append(req_prompt)
                    c_and_e_list.append(c_and_e)
                
                outputs = llm.generate(prompts=batch_prompt, sampling_params=sampling_params)
                # print(outputs)
                for i, output in enumerate(outputs):
                    prompt = c_and_e_list[i]
                    # response = output.outputs[0].text.strip()
                    logging.info(f"The concept of sampling {prompt}")
                    for index in range(MAX_N):
                        response = output.outputs[index].text.strip()
                        logging.info(f"Model Response Index {index} :\n{response}\n")
                        if model_name.startswith('/'):
                            typed_model_name = model_name.split("/")[-1]
                        else:
                            typed_model_name = model_name.split("/")[1]
                        save_rationale_data(prompt, response, f"./data_collect/mathsmith-train/{typed_model_name}_generated_problems_{today_str}.jsonl")
                gen_num += BATCH_SIZE
                pbar.update(BATCH_SIZE)