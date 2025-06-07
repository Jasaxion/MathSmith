python infer_longcot.py \
  --data_path data/qwq/qwq_aime2024_test.jsonl \
  --output_path data/qwen3/aime2024_predictions.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \
  --thinking

python cal_acc.py --output_path data/qwen3/aime2024_predictions.jsonl