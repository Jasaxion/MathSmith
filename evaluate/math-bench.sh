#!/bin/bash
set -e

model_name='xxxx'
output_dir='output/xxx'
n_gpus=2
n=1

mkdir -p "${output_dir}"

# --- AIME 2024 Start ---
echo "Processing AIME 2024..."
python infer_longcot.py \
  --data_path data/qwen3/aime2024_test.jsonl \
  --output_path "${output_dir}/aime2024_predictions_thinking.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2024_predictions_thinking.jsonl"
echo "AIME 2024 processed completed"
echo ""
# --- AIME 2024 End ---

# --- AIME 2025 Start ---
echo "Processing AIME 2025..."
python infer_longcot.py \
  --data_path data/qwen3/aime2025_test.jsonl \
  --output_path "${output_dir}/aime2025_predictions_thinking.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2025_predictions_thinking.jsonl"
echo "AIME 2025 processed completed"
echo ""
# --- AIME 2025 End ---

# --- MATH500 Start---
echo "Processing MATH500..."
python infer_longcot.py \
  --data_path data/qwen3/math500_test.jsonl \
  --output_path "${output_dir}/math500_predictions_thinking.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/math500_predictions_thinking.jsonl"
echo "MATH500 processed completed"
echo ""
# --- MATH500 End---

# --- Gsm8k Start---
echo "Processing Gsm8k..."
python infer_longcot.py \
  --data_path data/qwen3/gsm8k_test.jsonl \
  --output_path "${output_dir}/gsm8k_predictions_thinking.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking.jsonl"
echo "Gsm8k processed completed"
echo ""
# --- Gsm8k End---

echo "✅ All tasks have been completed!"