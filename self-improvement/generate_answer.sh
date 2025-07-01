model_name='/media/data2/LLM/models/qwen/Qwen/Qwen2___5-Math-1___5B'
output_dir='output/qwen'
n_gpus=1
n=1

python generate_answer.py \
  --input_file ./output/problems/MathSmith-Qwen3-8B-add_30_percent_consist-Step100_generated_problems.jsonl \
  --output_dir ./output/answers \
  --sample_answer_count 1 \
  --target_same_answer_count 1 \
  --batch_size 1024
