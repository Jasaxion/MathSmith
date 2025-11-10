import argparse
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from str2bool import str2bool
from eval.qwen_math import math_equal, extract_answer, strip_string
from eval.juger import MathJudger
from datetime import datetime
import random
import os
import yaml
import subprocess
import logging
import sys
import gc

def create_logger(log_dir):
    today_str = datetime.now().strftime("%m-%d-%Y %H:%M:%S") 
    log_file = os.path.join(log_dir, f"SI-{today_str}.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    

def evaluate_on_mathsmith(model_path, prompts, solutions, args):
    model = LLM(
        model=model_path,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="auto",
        dtype=args.dtype,
        tensor_parallel_size=args.n_gpus,
        enforce_eager=True,
        gpu_memory_utilization=0.75,
    )
    
    with torch.no_grad():
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_len,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

    # Generate completions for remaining prompts
    batch_outputs = model.generate(prompts, sampling_params)
    completions = [completion.outputs[0].text for completion in batch_outputs]
    
    scorer = MathJudger()
    precision = 1e-4
    num_valid_question = 0
    incorrect_idxs = []
    
    for idx in range(len(completions)):
        completion = completions[idx]
        reference_solution = solutions[idx]
        if not reference_solution:
            continue
        num_valid_question += 1
        correct = scorer.judge(
            extract_answer(completion),
            strip_string(reference_solution.split("####")[1].strip()),
            precision)
        
        if not correct:
            incorrect_idxs.append(idx)
            
    del model
    del batch_outputs
    del scorer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return 1 - len(incorrect_idxs) /num_valid_question, incorrect_idxs

def sample_and_convert(idxs, train_data_path, n_sample, output_data_path):
    alpaca_data = []
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    with open(train_data_path, encoding="utf-8") as f:
        data = json.load(f)
    for idx in idxs:
        question_and_answers = data[idx]["sampled_question"]
        sampled_qna = random.sample(question_and_answers, n_sample)
        for qna in sampled_qna:
            new_entry = {
                "instruction": instruction,
                "input": qna['problem'],
                "output": qna['anwer']
            }
            alpaca_data.append(new_entry)
    
    with open(output_data_path, 'w', encoding='utf-8') as outfile:
        json.dump(alpaca_data, outfile, indent=4, ensure_ascii=False)

def generate_sft_yaml(load_model_path, save_model_path, yaml_path, yaml_template_path):
    with open(yaml_template_path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    yaml_data['model_name_or_path'] = load_model_path
    yaml_data['output_dir'] = save_model_path
    
    # Write the modified YAML to the specified path
    with open(yaml_path, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)

def sft(yaml_path):
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    command = ["llamafactory-cli", "train", yaml_path]
    
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, 
        )
        
        print("Training started. Real-time output:")
        print("-" * 50)
        
        while True:
            output = process.stdout.readline()
            if not output and process.poll() is not None:
                break
            if output:
                print(output.strip()) 
        
        exit_code = process.poll()
        print("-" * 50)
        if exit_code == 0:
            print("Training completed successfully!")
        else:
            print(f"Training failed with exit code: {exit_code}")
            sys.exit(exit_code)
            
    except FileNotFoundError:
        print("Error: llamafactory-cli not found! Check installation.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(2)

def main():
    parser = argparse.ArgumentParser(description="Evaluate large language models on critical datasets.")
    parser.add_argument("--eval_data_path", type=str, default="../data_collect/mathsmith-test/MathSmith-HC-test.jsonl", help="Path to the dataset file.")
    parser.add_argument("--train_data_path", type=str, default="../data_collect/mathsmith-test/MathSmith-HC-test-collection-question-with-shortcotanswer.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the pretrained model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use for the model (e.g., fp16, bf16, etc.).")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=32768, help="Maximum number of tokens to generate.")
    parser.add_argument("--use_chat_template", type=str2bool, default=True)
    parser.add_argument("--use_concat", type=str2bool, default=False, help="Whether to use concatenation for prompts or System/User format.")
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--yaml_template_path", default='./sft/MathSmith_Evaluator_Qwen_Math_1_5B.yaml')
    parser.add_argument("--temp_train_data_path", default='./data/Self_Improvement_Data.json')
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--max_practice_num", type=int, default=10)
    parser.add_argument("--expected_acc", type=float, default=0.75)

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    output_model_dir = os.path.join(args.output_dir, 'model')
    output_log_dir = os.path.join(args.output_dir, 'log')
    output_config_dir = os.path.join(args.output_dir, 'config')
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_log_dir, exist_ok=True)
    os.makedirs(output_config_dir, exist_ok=True)
    
    create_logger(output_log_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print("thinking mode", args.thinking)
    
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    ms_prompts = []
    ms_solutions = []
    with open(args.eval_data_path, encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            prompt = item["prompt"]
            if args.use_chat_template:
                if args.use_concat:
                    prompt = prompt + " " + system_prompt
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.thinking)
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }

                if 'qwen3' in args.model_path.lower():
                    template_kwargs['enable_thinking'] = args.thinking

                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            ms_prompts.append(prompt)
            ms_solutions.append(item.get("reference_solution", item.get("solution")))
    
    for i in range(args.max_practice_num):
        logging.info(f'run {i}')
        load_model_path = args.model_path if i == 0 else os.path.join(output_model_dir, f'practice_{i-1}')
        save_model_path = os.path.join(output_model_dir, f'practice_{i}')
        yaml_path = os.path.join(output_config_dir, f'practice_{i}.yaml') 
        
        acc, incorrect_idx = evaluate_on_mathsmith(load_model_path, ms_prompts, ms_solutions, args)
        logging.info(f'acc: {acc}')
        if acc > args.expected_acc:
            logging.info(f' acc > expected_acc, done')
            break
        sample_and_convert(incorrect_idx, args.train_data_path, args.n_sample, args.temp_train_data_path)
        generate_sft_yaml(load_model_path, save_model_path, yaml_path, args.yaml_template_path)
        sft(yaml_path)


if __name__ == "__main__":
    main()