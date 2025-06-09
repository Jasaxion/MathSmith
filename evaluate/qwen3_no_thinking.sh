python infer_longcot.py \
  --data_path data/qwen3/aime2024_test.jsonl \
  --output_path output/qwen3/aime2024_predictions.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \

python cal_acc.py --output_path output/qwen3/aime2024_predictions.jsonl