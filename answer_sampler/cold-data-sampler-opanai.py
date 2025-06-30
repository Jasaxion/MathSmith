# 用于生成 rationale 冷启动数据-->进行 sft 维护模型
# 每次都会生成MAX_SFT_DATA_SIZE个，直接拓展原始的数据集

import os
import time
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import json
import random
import re
import logging
import openai 
from gpt_proxy_client import openai_proxy

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "sft_cold_data_generation.log")

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

# PROMPT_RATIONALE = """
# Given the Concepts and Explanations along with the instructions below, develop a **single challenging mathematics problem** suitable for advanced Olympiads

# ## Instructions:
# A. Select Relevant Concepts: 
#   1. Carefully analyze the provided concepts and their explanations.
#   2. Based on this analysis, choose a suitable subset of concepts (at least two; more are welcome if appropriate) that can be naturally and meaningfully integrated into a cohesive, well-focused mathematics problem.

# B. Problem Difficulty Rules:
#   1. The problem should require deep insight and non-obvious reasoning.
#   2. It must include **at least two** of the following difficulty features:
#     a. Multi-step Reasoning: Requires multiple sequential logical steps.
#     b. Cross-topic Integration: Combines distinct mathematical topics.
#     c. Implicit or Reverse Logic: Includes hidden conditions or reverse deduction.
#     d. Distractors: Contains misleading or extraneous conditions.
#     e. Abstract Modeling: Translates complex scenarios into mathematical form.
#     f. Multiple Solution Paths: Allows various non-trivial solving methods.
#     g. Advanced Manipulation: Necessitates sophisticated algebraic or geometric transformations.
#     h. Extreme Conditions: Focuses on limits or boundary values.
#     i. Non-standard Representation: Uses unconventional presentation of familiar concepts.

# C. Problem Format Constraints:
#   1. Generate **only one single problem**, do not generate multiple questions, sub-parts, or a sequence of related questions.
#   2. The problem should only have a **single, unified solving goal** — not a series of subtasks [no numbered sub-parts (e.g., 1., 2., (a), (b)) and avoid multi-step phrasing such as “first..., then...”. ] — and should be clearly answerable through a focused line of reasoning.
#   3. The problem must admit a well-defined, verifiable solution — that is, it should have a specific and unambiguous answer such as a numerical value, algebraic expression, equation.
#   4. The problem should not be a proof-type question. Its primary goal must be to find a specific, verifiable result — not to prove a general statement.
#   5. Avoid adding any secondary tasks such as "construct an example," "justify your result," or "verify a case" unless such construction is itself the core goal of the problem. **Keep the question focused and free of auxiliary subtasks**.

# ## Output Format:
# A. Rationale  
# Provide your reasoning and thought process **Step By Step** for how you constructed the problem, including:
#   Step 1. Analyze and understand the given concepts in depth.
#   Step 2. Select a suitable combination of concepts that can be meaningfully integrated.
#   Step 3. Explain how these concepts are woven together into a unified mathematical scenario.
#   Step 4. Identify which difficulty features you incorporated and describe how they are reflected in the problem.
#   Step 5. Formulate the final problem statement clearly and concisely.
# Enclose your rationale using the following tags:
# <!-- BEGIN RATIONALE -->
# [Your construction thought process goes here]
# <!-- END RATIONALE -->

# B. Problem
# Present the final problem clearly within the following tags:
# <!-- BEGIN PROBLEM -->
# [Your final math problem goes here]
# <!-- END PROBLEM -->

# Given Concept and Explanation:
# {CONCEPT_AND_EXPLANATION}

# """

PROMPT_RATIONALE = """
Given the Concepts and Explanations along with the instructions below, develop a **single challenging mathematics problem** suitable for advanced Olympiads

## Instructions:
A. Select Relevant Concepts: 
  1. Carefully analyze the provided concepts and their explanations.
  2. Based on this analysis, choose a suitable subset of concepts (at least two; more are welcome if appropriate) that can be naturally and meaningfully integrated into a cohesive, well-focused mathematics problem.

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

C. Problem Format Constraints:
  1. Generate **only one single problem**, do not generate multiple questions, sub-parts, or a sequence of related questions.
  2. The problem should only have a **single, unified solving goal** — not a series of subtasks [no numbered sub-parts (e.g., 1., 2., (a), (b)) and avoid multi-step phrasing such as “first..., then...”. ] — and should be clearly answerable through a focused line of reasoning.
  3. The problem must admit a well-defined, verifiable solution — that is, it should have a specific and unambiguous answer such as a numerical value, algebraic expression, equation.
  4. The problem must not involve any kind of proof. Do not use phrases like "prove that", "show that", or similar. The question should ask for a specific, verifiable result, such as a number, expression, or equation.
  5. Avoid adding any secondary tasks such as "construct an example," "justify your result," or "verify a case" unless such construction is itself the core goal of the problem. **Keep the question focused and free of auxiliary subtasks**.

## Output Format:
A. Rationale  
Provide your reasoning and thought process **Step By Step** for how you constructed the problem, including:
  Step 1. Analyze and understand the given concepts in depth.
  Step 2. Select a suitable combination of concepts that can be meaningfully integrated.
  Step 3. Explain how these concepts are woven together into a unified mathematical scenario.
  Step 4. Identify which difficulty features you incorporated and describe how they are reflected in the problem.
  Step 5. Formulate the final problem statement clearly and concisely.
Enclose your rationale using the following tags:
<!-- BEGIN RATIONALE -->
[Your construction thought process goes here]
<!-- END RATIONALE -->

B. Problem
Present the final problem clearly within the following tags:
<!-- BEGIN PROBLEM -->
[Your final math problem goes here]
<!-- END PROBLEM -->

Given Concept and Explanation:
{CONCEPT_AND_EXPLANATION}

"""

API_KEY = ""
client = openai_proxy.GptProxy(api_key=API_KEY)
MODEL = "gpt-4o-2024-11-20"
SOURCE_DIR = "/home/mnt/zhanshaoxiong/pipeline/RL_math_model/collection_concept_and_detail/collect_planetmath_grouped_deduplicated.jsonl"
COLD_SFT_DATA_DIR = "/home/mnt/zhanshaoxiong/pipeline/RL_math_model/cold_data_generation/cold-sft-data"
MAX_RETRIES = 10
MAX_SFT_DATA_SIZE = 20000

def load_concept_data(source_file: str) -> list:
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def save_rationale_data(c_and_e, rationale_data, output_file: str):
    rationale_match = re.search(r"<!-- BEGIN RATIONALE -->(.*?)<!-- END RATIONALE -->", rationale_data, re.DOTALL)
    problem_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", rationale_data, re.DOTALL)
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

def sample_concept_data(concept_data, sample_size: int):
    if sample_size > len(concept_data):
        raise ValueError(f"sample_size ({sample_size}) is greater than total number of samples ({len(concept_data)})")
    sampled_lines = random.sample(concept_data, sample_size)
    sampled_data = [json.loads(line) for line in sampled_lines]
    return sampled_data

def generate_rationale(concept_and_explanation: str, client) -> str:
    user_prompt = PROMPT_RATIONALE.format(CONCEPT_AND_EXPLANATION=concept_and_explanation)
    retry_count = 0
    while True:
        try:
            # response = client.chat.completions.create(
            #     model=MODEL,
            #     messages=[
            #         {"role": "system", "content": "Given the Concepts and Explanations along with the instructions below, develop a single challenging mathematics problem suitable for advanced Olympiads"},
            #         {"role": "user", "content": user_prompt},
            #     ],
            #     stream=False
            # )
            # import pprint
            response = client.generate(
                model=MODEL,
                messages=[
                    # {"role": "system", "content": "Given the Concepts and Explanations along with the instructions below, develop a single challenging mathematics problem suitable for advanced Olympiads"},
                    {"role": "user", "content": user_prompt}
                ],
                transaction_id="lsch_test_0065"
            )
            # print(response.json())
            # reply = response.choices[0].message.content
            reply = response.json()["data"]["response_content"]["choices"][0]["message"]["content"]
            return reply
        except Exception as e:
            retry_count += 1
            if retry_count > MAX_RETRIES:
                raise e
            time.sleep(5)

if __name__ == "__main__":
    SAMPLE_NUM = 5
    concept_data = load_concept_data(SOURCE_DIR)
    for i in range(MAX_SFT_DATA_SIZE):
        logging.info(f"正在采样第 {i + 1} 次数据")
        sampled_data = sample_concept_data(concept_data, SAMPLE_NUM)
        idx = 1
        c_and_e = ""
        for idx in range(1, SAMPLE_NUM + 1):
            concept_text = sampled_data[idx - 1]["Concept"]
            explanation_text = sampled_data[idx - 1]["Explanation"]
            merged_concept = str(idx) + ". " + concept_text + ": " + explanation_text + "\n"
            c_and_e += merged_concept
        logging.info(f"采样的概念集合 {c_and_e}")
        # concept_and_explanation = sampled_data[0]["Concept"] + ": " + sampled_data[0]["Explanation"]
        rationale = generate_rationale(c_and_e, client)
        logging.info(f"生成的推理和问题: {rationale}")
        save_rationale_data(c_and_e, rationale, os.path.join(COLD_SFT_DATA_DIR, f"cold_data_gpt4o-no-prove-0519.jsonl"))