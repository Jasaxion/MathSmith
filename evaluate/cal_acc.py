import argparse
import json
from tqdm import tqdm
import numpy as np
import re
from collections import defaultdict

from eval.math_equivalence import is_equiv_minerva as is_equiv
from eval.util import last_boxed_only_string, first_boxed_only_string, remove_boxed
from eval.qwen_math import math_equal, extract_answer, strip_string


def main():
    parser = argparse.ArgumentParser(description="Evaluate large language models")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")

    args = parser.parse_args()

    completions_by_prompt = defaultdict(lambda: defaultdict(list))

    all_items = []
    with open(args.output_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            all_items.append(item)

            if "prompt" not in item:
                raise ValueError(f"Item is missing required 'prompt' field")

            source = item["source"]
            prompt = item["prompt"]
            completions_by_prompt[source][prompt].append(item)

    results = defaultdict(lambda: defaultdict(dict))

    prompts_by_source = defaultdict(set)

    for source in completions_by_prompt:
        for prompt in tqdm(completions_by_prompt[source]):
            completions = completions_by_prompt[source][prompt]

            prompts_by_source[source].add(prompt)

            for run_idx, item in enumerate(completions):
                completion = item["completion"]
                reference_solution = item.get("reference_solution", item.get("solution"))

                if source in ["gsm8k"]:
                    correct = math_equal(
                        extract_answer(completion),
                        reference_solution.split("####")[-1].strip()
                    )
                elif source in ["math", "aime2024", "aime2025"]:
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
                else:
                    raise NotImplementedError(f"Source '{source}' is not implemented")

                results[source][run_idx][prompt] = int(correct)

    print("\nRESULTS BY SOURCE:")
    print("-" * 80)
    print(f"{'Source':<15} {'Accuracy':<20} {'Num Prompts':<15} {'Runs':<10}")
    print("-" * 80)

    for source in sorted(results.keys()):
        # We expect 8 runs (0-7)
        expected_runs = 8
        prompts = sorted(prompts_by_source[source])

        run_accuracies = []
        run_details = []

        for run_idx in range(expected_runs):
            if run_idx not in results[source]:
                print(f"Warning: Source '{source}' is missing run index {run_idx}")
                continue

            correct_count = 0
            total_count = 0

            for prompt in prompts:
                if prompt in results[source][run_idx]:
                    correct_count += results[source][run_idx][prompt]
                    total_count += 1

            if total_count > 0:
                run_accuracy = correct_count / total_count
                run_accuracies.append(run_accuracy)
                run_details.append(f"Run {run_idx}: {run_accuracy:.4f} ({correct_count}/{total_count})")

        if run_accuracies:
            mean_accuracy = np.mean(run_accuracies)
            std_dev = np.std(run_accuracies, ddof=1)

            mean_pct = round(mean_accuracy * 100, 1)
            std_dev_pct = round(std_dev * 100, 1)

            accuracy_str = f"{mean_pct:.1f}% ± {std_dev_pct:.1f}%"

            print(f"{source:<15} {accuracy_str:<20} {len(prompts):<15} {len(run_accuracies)}")

            for detail in run_details:
                print(f"  {detail}")

            print("-" * 80)


if __name__ == "__main__":
    main()