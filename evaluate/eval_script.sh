#!/bin/bash
set -e

# Run 3 reasoning tests in Zero-shot mode and check the accuracy of the results.

model_name='xxxx'
output_dir='output/xxx'
n_gpus=2
n=1

mkdir -p "${output_dir}"

# --- AIME 2024 Start ---
# --- AIME 2024 x1 ---
echo "Processing AIME 2024...(x1)"
python infer_longcot.py \
  --data_path data/qwen3/aime2024_test.jsonl \
  --output_path "${output_dir}/aime2024_predictions_thinking_x1.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2024_predictions_thinking_x1.jsonl"
echo "AIME 2024 processed completed"
echo ""

# --- AIME 2024 x2 ---
echo "Processing AIME 2024...(x2)"
python infer_longcot.py \
  --data_path data/qwen3/aime2024_test.jsonl \
  --output_path "${output_dir}/aime2024_predictions_thinking_x2.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2024_predictions_thinking_x2.jsonl"
echo "AIME 2024 processed completed"
echo ""

# --- AIME 2024 x3 ---
echo "Processing AIME 2024...(x3)"
python infer_longcot.py \
  --data_path data/qwen3/aime2024_test.jsonl \
  --output_path "${output_dir}/aime2024_predictions_thinking_x3.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2024_predictions_thinking_x3.jsonl"
echo "AIME 2024 processed completed"
echo ""
# --- AIME 2024 End ---

# --- AIME 2025 Start ---
# --- AIME 2025 x1---
echo "Processing AIME 2025(x1)..."
python infer_longcot.py \
  --data_path data/qwen3/aime2025_test.jsonl \
  --output_path "${output_dir}/aime2025_predictions_thinking_x1.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2025_predictions_thinking_x1.jsonl"
echo "AIME 2025 processed completed"
echo ""

echo "Processing AIME 2025(x2)..."
python infer_longcot.py \
  --data_path data/qwen3/aime2025_test.jsonl \
  --output_path "${output_dir}/aime2025_predictions_thinking_x2.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2025_predictions_thinking_x2.jsonl"
echo "AIME 2025 processed completed"
echo ""

echo "Processing AIME 2025(x3)..."
python infer_longcot.py \
  --data_path data/qwen3/aime2025_test.jsonl \
  --output_path "${output_dir}/aime2025_predictions_thinking_x3.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/aime2025_predictions_thinking_x3.jsonl"
echo "AIME 2025 processed completed"
echo ""
# --- AIME 2025 End ---

# --- MATH500 Start---
# --- MATH500 ---
echo "Processing MATH500(x1)..."
python infer_longcot.py \
  --data_path data/qwen3/math500_test.jsonl \
  --output_path "${output_dir}/math500_predictions_thinking_x1.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/math500_predictions_thinking_x1.jsonl"
echo "MATH500 processed completed"
echo ""

echo "Processing MATH500(x2)..."
python infer_longcot.py \
  --data_path data/qwen3/math500_test.jsonl \
  --output_path "${output_dir}/math500_predictions_thinking_x2.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/math500_predictions_thinking_x2.jsonl"
echo "MATH500 processed completed"
echo ""

echo "Processing MATH500(x3)..."
python infer_longcot.py \
  --data_path data/qwen3/math500_test.jsonl \
  --output_path "${output_dir}/math500_predictions_thinking_x3.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/math500_predictions_thinking_x3.jsonl"
echo "MATH500 processed completed"
echo ""
# --- MATH500 End---

# --- Gsm8k Start---
# --- Gsm8k ---
echo "Processing Gsm8k(x1)..."
python infer_longcot.py \
  --data_path data/qwen3/gsm8k_test.jsonl \
  --output_path "${output_dir}/gsm8k_predictions_thinking_x1.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking_x1.jsonl"
echo "Gsm8k processed completed"
echo ""

echo "Processing Gsm8k(x2)..."
python infer_longcot.py \
  --data_path data/qwen3/gsm8k_test.jsonl \
  --output_path "${output_dir}/gsm8k_predictions_thinking_x2.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking_x2.jsonl"
echo "Gsm8k processed completed"
echo ""

echo "Processing Gsm8k(x3)..."
python infer_longcot.py \
  --data_path data/qwen3/gsm8k_test.jsonl \
  --output_path "${output_dir}/gsm8k_predictions_thinking_x3.jsonl" \
  --model_path "${model_name}" \
  --n_gpus "${n_gpus}" \
  --n "${n}"

python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking_x3.jsonl"
echo "Gsm8k processed completed"
echo ""
# --- Gsm8k End---

echo "✅ All tasks have been completed!"