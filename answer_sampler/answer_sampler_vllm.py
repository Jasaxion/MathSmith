# 使用 VLLM 批量处理问题并生成答案
# 该脚本将从指定的 JSONL 文件中读取问题，使用 vLLM 模型批量生成答案，并将结果保存到指定的输出目录中，并支持多数投票
import os
import sys
import logging
import json
import re
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, Union, List, Dict
import argparse

# --- Basic ---
LOG_DIR_BASE = "./logs"
DEFAULT_SAMPLE_ANSWER = 6
VLLM_MODEL_ID_CONST = "Qwen/Qwen3_30B_A3B"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
ENABLE_THINKING = True # For the Qwen3 model, whether to enable thinking mode
VLLM_MAX_TOKEN = 33768 # vllm service maximum token length
VLLM_TP = 4 # vllm tensor parallel size
VLLM_GPU_MEMORY_UTILIZATION = 0.98 # Vllm GPU memory utilization
MODEL_MAX_TOKEN = 32768 # Maximum token length requested by the model (must be less than the maximum length of vllm)
TARGET_SAME_ANSWER_COUNT = 3 # The number of answers with the same target is less than this number, the results will be considered invalid.
DEFAULT_BATCH_SIZE = 16 # The number of problems processed per batch

# for thinking
if ENABLE_THINKING:
    MODEL_TEMPERATUR = 0.6
    MODEL_TOPP = 0.95
    MODEL_TOPK = 20
    MODEL_MINP = 0
else:
    # for no thinking
    MODEL_TEMPERATUR = 0.7
    MODEL_TOPP = 0.8
    MODEL_TOPK = 20
    MODEL_MINP = 0

# --- log configuration ---
current_time_for_log = datetime.now()
date_str_for_log = current_time_for_log.strftime("%Y%m%d")
datetime_str_for_log = current_time_for_log.strftime("%Y%m%d_%H%M%S")
script_name_for_log = os.path.splitext(os.path.basename(__file__))[0]
os.makedirs(LOG_DIR_BASE, exist_ok=True)
early_log_file = os.path.join(LOG_DIR_BASE, f"{script_name_for_log}_{date_str_for_log}_{datetime_str_for_log}.log")

root_logger_early = logging.getLogger()
if root_logger_early.hasHandlers():
    for handler in root_logger_early.handlers[:]:
        handler.close()
        root_logger_early.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(module)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(early_log_file, mode="a"),
        logging.StreamHandler()
    ]
)

# --- Version information printing ---
current_working_directory = os.getcwd()
logging.info(f"Current working directory of the Python script (Current Working Directory): {current_working_directory}")
python_executable_path = sys.executable
logging.info(f"Current Python interpreter path (Python Executable Path): {python_executable_path}")
try:
    import vllm
    vllm_version = vllm.__version__
    logging.info(f"vLLM version (vLLM Version): {vllm_version}")
    logging.info(f"vLLM Installation path (vLLM Installation Path): {vllm.__file__}")
except Exception as e:
    logging.error(f"An error occurred while retrieving the vLLM version information: {e}")
try:
    import transformers
    transformers_version = transformers.__version__
    logging.info(f"Transformers version (Transformers Version): {transformers_version}")
    logging.info(f"Transformers Installation path (Transformers Installation Path): {transformers.__file__}")
except Exception as e:
    logging.error(f"An error occurred while retrieving Transformers version information: {e}")

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    logging.critical("CRITICAL ERROR: vllm or transformers library not found. Please install them (e.g., pip install vllm transformers torch)")
    exit(1)

# --- Initializing vLLM LLM and tokenizer ---
llm_instance_global: Optional[LLM] = None
tokenizer_instance_global: Optional[AutoTokenizer] = None

try:
    logging.info(f"Initializing vLLM with model: {VLLM_MODEL_ID_CONST}")
    tokenizer_instance_global = AutoTokenizer.from_pretrained(VLLM_MODEL_ID_CONST, trust_remote_code=True)
    llm_instance_global = LLM(
        model=VLLM_MODEL_ID_CONST,
        trust_remote_code=True,
        tensor_parallel_size=VLLM_TP,
        max_model_len=VLLM_MAX_TOKEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        dtype="auto"
    )
    logging.info("vLLM LLM and Tokenizer initialized successfully.")
except Exception as e:
    logging.critical(f"Fatal Error initializing vLLM model or tokenizer: {e}", exc_info=True)
    exit(1)

def extract_boxed_answer(text: str) -> Optional[str]: 
    if not isinstance(text, str):
        return None
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    if matches:
        return matches[-1]
    return None

def load_problems(file_path: str) -> List[Dict]: 
    problems_data: List[Dict] = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line_num, line_content in enumerate(file, 1):
                line_content_stripped = line_content.strip()
                if line_content_stripped:
                    try:
                        record = json.loads(line_content_stripped)
                        if "Problem" in record and isinstance(record["Problem"], str) and \
                           "Sampled_concept" in record and isinstance(record["Sampled_concept"], (str, list)):
                            problems_data.append(record)
                        else:
                            logging.warning(
                                f"Line {line_num} in '{file_path}' is missing 'Problem' or 'Sampled_concept' key, "
                                f"or their types are incorrect: '{line_content_stripped[:200]}...'"
                            )
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON on line {line_num} in '{file_path}': {e} - Line: '{line_content_stripped[:200]}...'")
    except FileNotFoundError:
        logging.error(f"Problem file not found: {file_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading problems from {file_path}: {e}")
    return problems_data

def process_problems_batch(
    problems_batch: List[Dict],
    batch_start_index: int,
    total_problems: int,
    llm_instance: LLM,
    tokenizer_instance: AutoTokenizer,
    sampling_params_instance: SamplingParams
) -> List[Optional[Dict]]:
    
    batch_size = len(problems_batch)
    logging.info(f"Processing batch starting at problem {batch_start_index + 1} "
                 f"(size: {batch_size}, total: {total_problems}) with vLLM (n={sampling_params_instance.n} samples per problem).")

    # 1. Create prompts for each issue in the batch
    prompts_batch = []
    valid_problems_in_batch = [] # Only valid problem and their original data are retained
    for i, problem_record in enumerate(problems_batch):
        problem_text = problem_record.get("Problem", "")
        if not problem_text:
            logging.warning(f"Problem at index {batch_start_index + i} has no 'Problem' text. Skipping in this batch.")
            continue
        
        try:
            prompt_str = tokenizer_instance.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": problem_text}],
                tokenize=False,
                add_generation_prompt=True,
                presence_penalty=1.2,
                enable_thinking=ENABLE_THINKING
            )
            prompts_batch.append(prompt_str)
            valid_problems_in_batch.append(problem_record)
        except Exception as e:
            logging.error(f"Error applying chat template for problem at index {batch_start_index + i}: {e}", exc_info=True)
    
    if not prompts_batch:
        logging.warning("No valid prompts were generated for this batch.")
        return []

    # 2. Call vLLM.generate at once to handle the whole batch
    try:
        request_outputs = llm_instance.generate(prompts_batch, sampling_params_instance)
    except Exception as e:
        logging.error(f"Fatal error during vLLM batch generation: {e}", exc_info=True)
        # If the entire batch fails, generate an error result for each issue in the batch.
        error_results = []
        for problem_record in valid_problems_in_batch:
             error_results.append({
                "problem": problem_record.get("Problem", ""),
                "answer": f"Error: vLLM batch generation failed: {e}",
                "answer_dict": {},
                "highest_freq": 0,
                "sampled_concept": problem_record.get("Sampled_concept")
            })
        return error_results

    # 3. Process the returned results and match them with the original question.
    batch_results = []
    for i, single_request_output in enumerate(request_outputs):
        current_problem_record = valid_problems_in_batch[i] # 结果与有效问题一一对应
        current_problem_text = current_problem_record.get("Problem", "N/A")

        generated_samples_data = []
        num_received_samples = len(single_request_output.outputs)
        if num_received_samples != sampling_params_instance.n:
            logging.warning(f"Requested {sampling_params_instance.n} samples for problem, but received {num_received_samples}.")

        for gen_output in single_request_output.outputs:
            rationale_output = gen_output.text.strip()
            extracted_ans = extract_boxed_answer(rationale_output)
            generated_samples_data.append({"rationale": rationale_output, "boxed_answer": extracted_ans})
            
        final_highest_freq = 0
        if not generated_samples_data:
            logging.warning(f"No samples were processed for problem.")
            final_answer_rationale = "Error: No samples generated."
            output_answer_dict = {}
        else:
            rationales_by_extracted_answer = defaultdict(list)
            for sample in generated_samples_data:
                key = sample.get("boxed_answer") if sample.get("boxed_answer") is not None else "[[NO_BOXED_ANSWER]]"
                rationales_by_extracted_answer[key].append(sample["rationale"])

            final_answer_rationale = generated_samples_data[0]["rationale"]
            if rationales_by_extracted_answer:
                answer_frequencies = Counter({key: len(r_list) for key, r_list in rationales_by_extracted_answer.items()})
                if answer_frequencies:
                    best_ans_content, highest_freq_val = answer_frequencies.most_common(1)[0]
                    final_highest_freq = highest_freq_val
                    if best_ans_content in rationales_by_extracted_answer and rationales_by_extracted_answer[best_ans_content]:
                        final_answer_rationale = rationales_by_extracted_answer[best_ans_content][0]

            output_answer_dict = defaultdict(list)
            for ans_key, list_of_rationales in rationales_by_extracted_answer.items():
                count = len(list_of_rationales)
                output_answer_dict[str(count)].append(list_of_rationales)

        # Combined final result
        batch_results.append({
            "problem": current_problem_text,
            "answer": final_answer_rationale,
            "answer_dict": dict(output_answer_dict),
            "highest_freq": final_highest_freq,
            "sampled_concept": current_problem_record.get("Sampled_concept")
        })
        
    return batch_results

def main():
    parser = argparse.ArgumentParser(description="Generate answers for problems using vLLM with batch processing.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL problem file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSONL answer file.")
    parser.add_argument("--sample_answer_count", type=int, default=DEFAULT_SAMPLE_ANSWER, help=f"Number of samples per problem (default: {DEFAULT_SAMPLE_ANSWER})")
    parser.add_argument("--target_same_answer_count", type=int, default=TARGET_SAME_ANSWER_COUNT, help=f"Target count for the most frequent answer to be considered valid (default: {TARGET_SAME_ANSWER_COUNT})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Number of problems to process in a single batch (default: {DEFAULT_BATCH_SIZE})")
    args = parser.parse_args()

    current_target_same_answer_count = args.target_same_answer_count
    batch_size = args.batch_size # <--- 新增

    # --- File path and naming ---
    current_time_main = datetime.now()
    datetime_str_main = current_time_main.strftime("%Y%m%d_%H%M%S")
    safe_model_name_for_file = re.sub(r'[^\w\.-]', '_', VLLM_MODEL_ID_CONST.split("/")[-1])
    input_file_basename = os.path.basename(args.input_file)
    input_file_name_without_ext = os.path.splitext(input_file_basename)[0]
    os.makedirs(args.output_dir, exist_ok=True)
    output_valid_filename = f"Sampler_answer_with_{safe_model_name_for_file}_vLLM_{input_file_name_without_ext}_{datetime_str_main}.jsonl"
    output_valid_file_path = os.path.join(args.output_dir, output_valid_filename)
    output_invalid_filename = f"Sampler_answer_with_{safe_model_name_for_file}_vLLM_{input_file_name_without_ext}_{datetime_str_main}_invalid.jsonl"
    output_invalid_file_path = os.path.join(args.output_dir, output_invalid_filename)

    # --- Sampling Parameters ---
    current_sampling_params = SamplingParams(
        n=args.sample_answer_count,
        temperature=MODEL_TEMPERATUR,
        top_p=MODEL_TOPP,
        top_k=MODEL_TOPK,
        min_p=MODEL_MINP,
        max_tokens=MODEL_MAX_TOKEN,
    )

    if llm_instance_global is None or tokenizer_instance_global is None:
        logging.critical("LLM model or tokenizer was not initialized globally. Exiting.")
        exit(1)

    logging.info(f"--- Main Function Started ---")
    logging.info(f"Input problem file: {args.input_file}")
    logging.info(f"Valid output will be saved to: {output_valid_file_path}")
    logging.info(f"Invalid output will be saved to: {output_invalid_file_path}")
    logging.info(f"Samples per problem (n): {current_sampling_params.n}")
    logging.info(f"Target same answer count for valid data: {current_target_same_answer_count}")
    logging.info(f"Batch size: {batch_size}")

    problems_data = load_problems(args.input_file)
    if not problems_data:
        logging.error(f"No problems loaded from '{args.input_file}'. Exiting.")
        exit(1)
    
    total_problems_to_process = len(problems_data)
    logging.info(f"Loaded {total_problems_to_process} problems to process.")

    processed_valid_count = 0
    processed_invalid_count = 0

    try:
        with open(output_valid_file_path, "w", encoding="utf-8") as outfile_valid, \
             open(output_invalid_file_path, "w", encoding="utf-8") as outfile_invalid:
            
            for i in range(0, total_problems_to_process, batch_size):
                # 1. Get the problem data of the current batch
                problems_batch_data = problems_data[i : i + batch_size]
                
                # 2. Call the batch processing function
                results_batch = process_problems_batch(
                    problems_batch_data,
                    i,
                    total_problems_to_process,
                    llm_instance_global,
                    tokenizer_instance_global,
                    current_sampling_params
                )
                
                # 3. Traverse the batch results and write them to the file
                for result in results_batch:
                    if not result:
                        continue

                    highest_freq = result.pop("highest_freq")
                    
                    if highest_freq >= current_target_same_answer_count:
                        outfile_valid.write(json.dumps(result, ensure_ascii=False) + "\n")
                        processed_valid_count += 1
                    else:
                        outfile_invalid.write(json.dumps(result, ensure_ascii=False) + "\n")
                        processed_invalid_count += 1
                
                outfile_valid.flush()
                outfile_invalid.flush()

    except IOError as e:
        logging.critical(f"Could not write to output files: {e}", exc_info=True)
    except Exception as e:
        logging.critical(f"An unexpected error occurred during the main processing loop: {e}", exc_info=True)

    logging.info(f"--- All {total_problems_to_process} problems processed. Script finished. ---")
    logging.info(f"Total valid problems saved: {processed_valid_count}")
    logging.info(f"Total invalid problems saved: {processed_invalid_count}")

if __name__ == "__main__":
    main()
