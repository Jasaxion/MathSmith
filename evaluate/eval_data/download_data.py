import os
import json
from datasets import load_dataset

def download_math_benchmarks_separately():
    datasets_to_download = [
        ("Maxwell-Jia/AIME_2024", None, "train", "AIME2024"),
        ("opencompass/AIME2025", "AIME2025-I", "test", "AIME2025-I"),
        ("opencompass/AIME2025", "AIME2025-II", "test", "AIME2025-II"),
        ("openai/gsm8k", "main", "test", "GSM8k"),
        ("HuggingFaceH4/MATH-500", None, "test", "MATH"),
        # new eval dataset
        ("reasoning-machines/gsm-hard", None, "train", "GSM-Hard"),
        ("di-zhang-fdu/College_Math_Test", None, "test", "CollegeMath"),
    ]
    output_dir = "./"
    created_files = []
    for path, name, split, benchmark_name in datasets_to_download:
        output_file = os.path.join(output_dir, f"{benchmark_name}.jsonl")
        try:
            trust_remote = path == "openai/gsm8k"
            dataset = load_dataset(path, name, split=split, trust_remote_code=trust_remote)
            
            count = 0
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for item in dataset:
                    item['benchmark'] = benchmark_name
                    f_out.write(json.dumps(item) + '\n')
                    count += 1
            
            print(f"write {count} data to {output_file}")
            created_files.append(output_file)

        except Exception as e:
            print(f"processing dataset {path} (sub: {name}) occurred error: {e}")

    for file_path in created_files:
        print(f"- {file_path}")

if __name__ == "__main__":
    download_math_benchmarks_separately()