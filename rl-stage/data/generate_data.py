import pandas as pd
import json
import random
SAMPLE_NUM=5
MAX_SAMPLE = 10000

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
{CONCEPT_AND_EXPLANATION}
"""

def load_concept_data(source_file: str) -> list:
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def sample_concept_data(concept_data, sample_size: int):
    if sample_size > len(concept_data):
        raise ValueError(f"sample_size ({sample_size}) is greater than total number of samples ({len(concept_data)})")
    sampled_lines = random.sample(concept_data, sample_size)
    sampled_data = [json.loads(line) for line in sampled_lines]
    return sampled_data

def sample_inputs(n_prompts):
    concept_sample_path = "../../data_collect/sampled_concept/collect_planetmath_grouped_deduplicated.jsonl"
    concept_data = load_concept_data(concept_sample_path)
    res_prompts = []
    for i in range(n_prompts):
        sampled_data = sample_concept_data(concept_data, SAMPLE_NUM)
        idx = 1
        c_and_e = ""
        for idx in range(1, SAMPLE_NUM + 1):
            concept_text = sampled_data[idx - 1]["Concept"]
            explanation_text = sampled_data[idx - 1]["Explanation"]
            merged_concept = str(idx) + ". " + concept_text + ": " + explanation_text + "\n"
            c_and_e += merged_concept
        prompt_structured = [
            {"role": "user", "content": QUESTION_INSTRUCTION.format(CONCEPT_AND_EXPLANATION=c_and_e)}
        ]
        res_prompts.append({"prompt": prompt_structured,
                            "data_source": "MathSmith",
                            "extra_info": {
                                "sampled_concept": c_and_e,
                                "index": idx,
                            },
                        })
    return res_prompts

data = sample_inputs(MAX_SAMPLE)
df = pd.DataFrame(data)
df.to_parquet("../grpo_data/train-10k-rationale-chat.parquet", index=False)