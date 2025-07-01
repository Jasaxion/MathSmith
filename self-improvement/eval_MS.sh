model_name='/media/data2/LLM/models/qwen/Qwen/Qwen2___5-Math-1___5B'
output_dir='output/qwen'
n_gpus=1
n=1

python evaluate_on_mathsmith.py \
  --output_path "${output_dir}/MS_prediction.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}" \
  --max_len 4096