# AIME 2024
python infer_longcot.py \
  --data_path data/qwen3/aime2024_test.jsonl \
  --output_path output/qwen3/nothinking/aime2024_predictions_thinking.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \

python cal_acc.py --output_path output/qwen3/nothinking/aime2024_predictions_thinking.jsonl

# AIME 2025
python infer_longcot.py \
  --data_path data/qwen3/aime2025_test.jsonl \
  --output_path output/qwen3/nothinking/aime2025_predictions_thinking.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \

python cal_acc.py --output_path output/qwen3/nothinking/aime2025_predictions_thinking.jsonl

# MATH500
python infer_longcot.py \
  --data_path data/qwen3/math500_test.jsonl \
  --output_path output/qwen3/nothinking/math500_predictions_thinking.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \

python cal_acc.py --output_path output/qwen3/nothinking/math500_predictions_thinking.jsonl

# Gsm8k
python infer_longcot.py \
  --data_path data/qwen3/gsm8k_test.jsonl \
  --output_path output/qwen3/nothinking/gsm8k_predictions_thinking.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \

python cal_acc.py --output_path output/qwen3/nothinking/gsm8k_predictions_thinking.jsonl