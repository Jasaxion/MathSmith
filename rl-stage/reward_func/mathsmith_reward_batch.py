import requests
from transformers import AutoTokenizer
import re, torch
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np
import math

model_name = "Qwen/Qwen3_30B_A3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
from concurrent.futures import ThreadPoolExecutor, as_completed

# Using multiple vllm services to accelerate inference speed and complete reward computation.
API_URL_LIST = [
    "",
]
SERVER_N = len(API_URL_LIST)
REASONING_ANSWER_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}.\n"
USE_GROUP_COT_LEN = False
MAX_GEN_TOKEN = 32768 # Control the maximum generation length for vllm
TARGET_SAMPLE_N = 3 # Number of samples to generate per prompt
USE_CONSIST_R = True # Whether to use consistency reward
TARGET = 2 # Target frequency for consistency reward
R_COT_WEIGHT = 0.7 # CoT length reward weight (0.7 for default)
R_CONSIST_WEIGHT = 0.3 # Consistency reward weight (0.3 for default)
REWARD_FILTER_DATA_OUTPUT_PATH = "../data/rl_reward_collect_data/collect_reward_data.jsonl"

def split_prompts(prompts, n):
    chunks = [[] for _ in range(n)]
    for i, prompt in enumerate(prompts):
        chunks[i % n].append((i, prompt))
    return chunks

def cot_length_rw(response_list):
    cot_lengths = []
    for res in response_list:
        match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
        if match:
            content = match.group(1)
            cot_len = len(tokenizer.encode(content, add_special_tokens=False))
        else:
            content = res.replace("<think>", "")
            cot_len = len(tokenizer.encode(content, add_special_tokens=False))
        cot_lengths.append(cot_len)
    avg_cot_len = sum(cot_lengths) / len(cot_lengths)

    r_cot_len = avg_cot_len / MAX_GEN_TOKEN
    
    return r_cot_len, avg_cot_len, cot_lengths

# 8192 dynamic cot length reward for medium difficulty
# def cot_length_rw(response_list):
#     cot_lengths = []
#     for res in response_list:
#         match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
#         if match:
#             content = match.group(1)
#             cot_len = len(tokenizer.encode(content, add_special_tokens=False))
#         else:
#             content = res.replace("<think>", "")
#             cot_len = len(tokenizer.encode(content, add_special_tokens=False))
#         cot_lengths.append(cot_len)
#     optimal_length = 8192
#     avg_cot_len = sum(cot_lengths) / len(cot_lengths)
#     difference = abs(avg_cot_len - optimal_length)

#     k = math.log(2) / 1000
#     r_cot_len = math.exp(-k * difference)

#     return r_cot_len, avg_cot_len, cot_lengths

# def cot_length_rw(response_list):
#     cot_lengths = []
#     for res in response_list:
#         match = re.search(r"<think>(.*?)</think>", res, re.DOTALL)
#         if match:
#             content = match.group(1)
#             cot_len = len(tokenizer.encode(content, add_special_tokens=False))
#         else:
#             content = res.replace("<think>", "")
#             cot_len = len(tokenizer.encode(content, add_special_tokens=False))
#         cot_lengths.append(cot_len)

#     if len(cot_lengths) > 2:
#         sorted_lens = sorted(cot_lengths)
#         trimmed = sorted_lens[1:-1]  # remove min & max
#     else:
#         trimmed = cot_lengths

#     log_scaled = [np.log1p(x) for x in trimmed]
#     avg_log = sum(log_scaled) / len(log_scaled)
#     r_cot_len = avg_log / np.log1p(MAX_GEN_TOKEN)

#     return r_cot_len, avg_log, cot_lengths, log_scaled


def consistency_rw(response_list):
    extracted_answers = []
    for res in response_list:
        if "</think>" in res:
            res = res.split("</think>")[-1]

        matches = re.findall(r"\\boxed{(.*?)}", res)
        if matches:
            extracted_answers.append(matches[-1].strip())
    if not extracted_answers:
        return 0.0, 0
    ans_freq = {ans: extracted_answers.count(ans) for ans in set(extracted_answers)}
    max_freq = max(ans_freq.values())

    # r_consist = 1.0 - abs(max_freq - TARGET) / TARGET
    if(max_freq >= TARGET):
        r_consist = 1.0
    else:
        r_consist = 0.0
    return r_consist, max_freq

def rationale_step_reward(rationale: str) -> float:
    step_matches = re.findall(r"\bStep\s*\d+\b", rationale, re.IGNORECASE)
    unique_steps = set(step_matches)
    n_step = len(unique_steps)
    
    if n_step <= 5:
        return n_step / 5.0
    else:
        return max(1.0 - (n_step - 5.0) / 5.0, 0)
    
def format_rw(generated_problem):
    rationale_match = re.search(r"<rationale>(.*?)</rationale>", generated_problem, re.DOTALL)
    problem_match = re.search(r"<problem>(.*?)</problem>", generated_problem, re.DOTALL)

    if rationale_match and problem_match:
        r_format = 1.0
        r_step = rationale_step_reward(rationale_match.group(1).strip())
        r_format = r_format * 0.7 + r_step * 0.3
    else:
        r_format = 0.0

    return r_format

def save_jsonl(res_out, output_path=None):
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"output_{timestamp}.jsonl"

    if isinstance(res_out, dict):
        res_out = [res_out]

    if not isinstance(res_out, list) or not all(isinstance(item, dict) for item in res_out):
        raise ValueError("res_out 必须是 dict 或 dict 组成的 list")

    with open(output_path, 'a', encoding='utf-8') as f:
        for item in res_out:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_problem(response):
    problem_match = re.search(r"<problem>(.*?)</problem>", response, re.DOTALL)
    if not problem_match:
        return "This is a format error problem, Do not thinking and direct reply \"invalid question\". "
    problem = problem_match.group(1).strip()
    return problem

def send_batch(api_url, indexed_prompts):
    indices, prompts = zip(*indexed_prompts)
    payload = {
        "prompt": list(prompts),
        "max_tokens": MAX_GEN_TOKEN,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        # "presence_penalty": 1.2,
        "n": TARGET_SAMPLE_N
    }
    # Legacy payload settings (Unstable for results)
    # payload = {
    #     "prompt": list(prompts),
    #     "max_tokens": MAX_GEN_TOKEN,
    #     "temperature": 0.01,
    #     "top_p": 1.0,
    #     "top_k": 1,
    #     "min_p": 0,
    #     "presence_penalty": 1.2,
    #     "n": TARGET_SAMPLE_N
    # }
    try:
        response = requests.post(api_url, json=payload, timeout=6500)
        response.raise_for_status()
        result = response.json()
        choices = result.get("choices", [])
        outputs = []
        for i, idx in enumerate(indices):
            output_list = []
            for j in range(TARGET_SAMPLE_N):
                choice = choices[i * TARGET_SAMPLE_N + j]
                text = choice.get("text", "").strip()
                output_list.append(text)
            outputs.append((idx, output_list))
        return outputs
    except Exception as e:
        raise Exception(f"Error from {api_url}: {e}")

def reward_fn(prompts, responses):
    print("[✅ Custom reward_fn Batch loaded successfully for MathSmith!]")

    r_format_list = [format_rw(response) for response in responses]
    problem_list = [extract_problem(response) for response in responses]
    system_prompt = "please reason step by step, and put your final answer within \\boxed{}."

    req_prompts = []
    for problem in problem_list:
        req_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        req_prompts.append(req_prompt)

    splitted = split_prompts(req_prompts, SERVER_N)

    results = [None] * len(req_prompts)
    with ThreadPoolExecutor(max_workers=SERVER_N) as executor:
        futures = [
            executor.submit(send_batch, API_URL_LIST[i], splitted[i])
            for i in range(SERVER_N)
        ]
        for future in as_completed(futures):
            output = future.result()
            for idx, text_list in output:
                results[idx] = text_list
    rewards = []
    assert len(results) == len(req_prompts)

    # Using the average cot length of the group as a reward.
    if(USE_GROUP_COT_LEN):
        group_map = defaultdict(list)
        for idx, prompt in enumerate(prompts):
            group_map[prompt].append(idx)

        for prompt_str, idx_list in group_map.items():
            group_results = [results[i] for i in idx_list]

            print("================================")
            print("Using Group Cot Length Reward!")
            print("Total Prompts Sample: ", len(group_results))

            # Calculate the average CoT length for the group
            group_cot_list = []
            for i, output_list in enumerate(group_results):
                choices_list = []
                for j, text in enumerate(output_list):
                    choices_list.append(text)
                _, avg_cot_len = cot_length_rw(choices_list)
                group_cot_list.append(avg_cot_len)
            group_ave_cot_len =  sum(group_cot_list) / len(group_cot_list)

            for i, global_idx in enumerate(idx_list):
                choices_list = []
                for j, text in enumerate(group_results[i]):
                    choices_list.append(text)
                _, avg_cot_len = cot_length_rw(choices_list)

                if avg_cot_len < 900:
                    r_cot_len = 0.0
                else:
                    r_cot_len = avg_cot_len / group_ave_cot_len

                print("Format reward: ", r_format_list[global_idx])
                print("Ave COT Length: ", avg_cot_len)
                print("Group Ave COT Length: ", group_ave_cot_len)
                print("COT Length reward: ", r_cot_len)

                if USE_CONSIST_R:
                    r_consist, max_freq = consistency_rw(choices_list)
                    print("Max Consistance frequency: ", max_freq)
                    print("Consist reward: ", r_consist)
                    reward = r_format_list[global_idx] + (R_COT_WEIGHT * r_cot_len + R_CONSIST_WEIGHT * r_consist)
                else:
                    reward = r_format_list[global_idx] + r_cot_len

                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                # Debug Save
                sample_promblem_and_answer = {}
                sample_promblem_and_answer['prompt'] = prompts[global_idx]
                sample_promblem_and_answer['problem'] = req_prompts[global_idx]
                sample_promblem_and_answer['r_format'] = r_format_list[global_idx]
                sample_promblem_and_answer['avg_cot_len'] = avg_cot_len
                sample_promblem_and_answer['r_cot_len'] = r_cot_len
                if USE_CONSIST_R:
                    sample_promblem_and_answer['r_consist'] = r_consist
                    sample_promblem_and_answer['max_freq'] = max_freq
                sample_promblem_and_answer['answer'] = choices_list
                sample_promblem_and_answer['output'] = responses[global_idx]
                save_jsonl(sample_promblem_and_answer, REWARD_FILTER_DATA_OUTPUT_PATH)

                print("Final reward: ", reward)
                print("================================")
                rewards[global_idx] = reward
    else:
        for i, output_list in enumerate(results):
            choices_list = []
            for j, text in enumerate(output_list):
                choices_list.append(text)

            r_cot_len, avg_cot_len, cot_lengths = cot_length_rw(choices_list)
            
            print("================================")
            print("Using Default reward method!")
            print("Ave COT Length: ", avg_cot_len)
            print("All Cot Length: ", cot_lengths)
            print("Format reward: ", r_format_list[i])
            print("COT Length reward: ", r_cot_len)
            if USE_CONSIST_R:
                r_consist, max_freq = consistency_rw(choices_list)
                print("Max Consistance frequency: ", max_freq)
                print("Consist reward: ", r_consist)
                if (r_format_list[i] <= 0.0):
                    r_cot_len = 0.0
                    r_consist = 0.0
                reward = r_format_list[i] + (R_COT_WEIGHT * r_cot_len + R_CONSIST_WEIGHT * r_consist)
            else:
                if (r_format_list[i] <= 0.0):
                    r_cot_len = 0.0
                reward = r_format_list[i] + r_cot_len
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            # Debug Save
            sample_promblem_and_answer = {}
            sample_promblem_and_answer['prompt'] = prompts[i]
            sample_promblem_and_answer['problem'] = req_prompts[i]
            sample_promblem_and_answer['r_format'] = r_format_list[i]
            sample_promblem_and_answer['avg_cot_len'] = avg_cot_len
            sample_promblem_and_answer['cot_lengths'] = cot_lengths
            sample_promblem_and_answer['r_cot_len'] = r_cot_len
            if USE_CONSIST_R:
                sample_promblem_and_answer['r_consist'] = r_consist
                sample_promblem_and_answer['max_freq'] = max_freq
            sample_promblem_and_answer['answer'] = choices_list
            sample_promblem_and_answer['output'] = responses[i]
            save_jsonl(sample_promblem_and_answer, REWARD_FILTER_DATA_OUTPUT_PATH)

            print("Final reward: ", reward)
            print("================================")
            rewards.append(reward)

    print("Final Batch Reward: ", rewards)
    return rewards
