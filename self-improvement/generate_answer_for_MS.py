import os
import json
from typing import List

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --- 1. 配置项 (请在此处修改) ---

# 输入文件路径
INPUT_FILE_PATH = "/media/data2/LLM/MathSmith/data_collect/mathsmith-test/MathSmith-HC-test-collection-question-with-shortcotanswer.json" 
# 输出目录 (请确保此目录已存在)
OUTPUT_DIR = "./output_data"

# VLLM 模型配置
VLLM_MODEL_ID = "/media/data2/LLM/models/qwen/Qwen/Qwen2.5-Math-7B-Instruct"
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.9

BATCH_SIZE = 50000
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# --- 2. VLLM 初始化 ---

print(f"Initializing vLLM with model: {VLLM_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_ID, trust_remote_code=True)
llm = LLM(
    model=VLLM_MODEL_ID,
    trust_remote_code=True,
    tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
    max_model_len=4096,
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    dtype="auto"
)
print("vLLM initialized.")


# --- 3. 主执行流程 ---

if __name__ == "__main__":
    # 准备输出路径，现在只有一个输出文件
    input_filename = os.path.splitext(os.path.basename(INPUT_FILE_PATH))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{input_filename}_regenerated.json")

    # 定义采样参数，每个问题只生成一个答案 (n=1)
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        top_p=0.8,
        max_tokens=4096,
    )

    # 1. 加载整个原始数据文件，并同时提取所有问题到一个扁平列表中
    print(f"Loading data from {INPUT_FILE_PATH}...")
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # original_data = original_data[:2] 
    all_problems_text = []
    for entry in original_data:
        for qna in entry["sampled_question"]:
            all_problems_text.append(qna["problem"])
    
    total_problems = len(all_problems_text)
    print(f"Loaded {total_problems} problems. Starting generation...")

    # 2. 批量生成所有问题的新答案
    all_new_answers = []
    for i in range(0, total_problems, BATCH_SIZE):
        batch_text = all_problems_text[i : i + BATCH_SIZE]
        
        prompts = []
        for problem_text in batch_text:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": problem_text}],
                tokenize=False,
                add_generation_prompt=True,
                presence_penalty=1.2,
            )
            prompts.append(prompt_str)

        # vLLM生成答案
        request_outputs = llm.generate(prompts, sampling_params)
        
        # 收集这个批次的结果
        for output in request_outputs:
            # 因为 n=1, 所以每个 request_output 只有一个 .outputs[0]
            new_answer = output.outputs[0].text.strip()
            all_new_answers.append(new_answer)
        
        print(f"Processed {len(all_new_answers)} / {total_problems} problems...")

    # 3. 将新答案重新注入到原始数据结构中
    # 使用迭代器可以方便地按顺序取用新答案
    answer_iterator = iter(all_new_answers)
    for entry in original_data:
        for qna in entry["sampled_question"]:
            # 使用 next(answer_iterator) 获取下一个新答案并替换旧答案
            # 注意：这里的键是 "anwer"，与您的原始文件保持一致
            qna["anwer"] = next(answer_iterator)

    # 4. 将修改后的完整数据结构写入新的JSON文件
    print(f"\n--- Writing regenerated data to {output_path} ---")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # indent=4 使输出的JSON文件格式优美，易于阅读
        json.dump(original_data, outfile, indent=4, ensure_ascii=False)

    print("--- Processing Finished ---")