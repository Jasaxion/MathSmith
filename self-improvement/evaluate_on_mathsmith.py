import argparse
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from str2bool import str2bool
from eval.math_equivalence import is_equiv_minerva as is_equiv
from eval.util import last_boxed_only_string, first_boxed_only_string, remove_boxed
from eval.qwen_math import math_equal, extract_answer, strip_string
import os
import re


def main():
    parser = argparse.ArgumentParser(description="Evaluate large language models on critical datasets.")
    parser.add_argument("--data_path", type=str, default="../data_collect/mathsmith-test/MathSmith-HC-test.jsonl", help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
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
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=8)

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # Load the tokenizer for LLaMA or any model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print("thinking mode", args.thinking)
    print("epoch", args.n)
    # Load inference framework
    model = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="auto",
        dtype=args.dtype,
        tensor_parallel_size=args.n_gpus,
        enforce_eager=True,
        gpu_memory_utilization=0.75,
    )

    items = []
    completions = []
    seed = 0
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    prompts = []
    # idx = 0
    with open(args.data_path, encoding="utf-8") as f:
        for line in f.readlines():
            # idx +=1 
            # if idx > 5:
            #     break
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
            
            prompts.append(prompt)
            items.append(item)

    with torch.no_grad():
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_len,
            repetition_penalty=args.repetition_penalty,
            seed=seed,
        )

    # Generate completions for remaining prompts
    batch_outputs = model.generate(prompts, sampling_params)
    completions = [completion.outputs[0].text for completion in batch_outputs]
    
    idx = 0
    for item, completion in zip(items, completions):
        print(f"------{idx}-----")
        idx += 1
        item["completion"] = completion
        reference_solution = item.get("reference_solution", item.get("solution"))
        
        if not reference_solution:
            item["correctness"] = -1
            continue
        
        correct = math_equal(
            extract_answer(completion),
            strip_string(reference_solution.split("####")[1].strip()),
            timeout=False,
        ) or is_equiv(
            remove_boxed(last_boxed_only_string(completion)),
            reference_solution.split("####")[-1].strip() if "####" in reference_solution else (
                remove_boxed(last_boxed_only_string(reference_solution)),
            )
        )
        item["correctness"] = int(correct)
        
        
    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()