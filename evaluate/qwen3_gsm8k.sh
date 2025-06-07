python infer_longcot.py \
  --data_path data/qwq/qwq_gsm8k_test.jsonl \
  --output_path data/qwen3/gsm8k_predictions.jsonl \
  --model_path 'Qwen/Qwen3-8B' \
  --n_gpus 1 \
  --n 1 \
  --thinking

python cal_acc.py --output_path data/qwen3/gsm8k_predictions.jsonl