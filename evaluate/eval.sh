#!/bin/bash
set -e

# 输入应该为Model_name + Customized suffix + use_thinking
declare -a model_list=(
    "qwen/qwen3,"
    "meta/llama,"
    "/home/qwen/qwen_model,Thinking,True"
    # "meta-llama/Llama-2-7b-chat-hf,_finetuned"
)
n_gpus=2
n=1

for item in "${model_list[@]}"; do
    IFS=',' read -r model_name suffix use_thinking <<< "$item"

    model_basename=$(basename "${model_name}")
    output_dir="output/${model_basename}${suffix}"
    mkdir -p "${output_dir}"

    thinking_arg=""
    if [[ "${use_thinking}" == "True" ]]; then
        thinking_arg="--thinking"
    fi

    # --- AIME 2024 Start ---
    # --- AIME 2024 x1 ---
    echo "Processing AIME 2024...(x1)"
    python infer_longcot.py \
    --data_path data/qwen3/aime2024_test.jsonl \
    --output_path "${output_dir}/aime2024_predictions_thinking_x1.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

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
    --n "${n}" \
    ${thinking_arg}

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
    --n "${n}" \
    ${thinking_arg}

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
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/aime2025_predictions_thinking_x1.jsonl"
    echo "AIME 2025 processed completed"
    echo ""

    echo "Processing AIME 2025(x2)..."
    python infer_longcot.py \
    --data_path data/qwen3/aime2025_test.jsonl \
    --output_path "${output_dir}/aime2025_predictions_thinking_x2.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/aime2025_predictions_thinking_x2.jsonl"
    echo "AIME 2025 processed completed"
    echo ""

    echo "Processing AIME 2025(x3)..."
    python infer_longcot.py \
    --data_path data/qwen3/aime2025_test.jsonl \
    --output_path "${output_dir}/aime2025_predictions_thinking_x3.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

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
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/math500_predictions_thinking_x1.jsonl"
    echo "MATH500 processed completed"
    echo ""

    echo "Processing MATH500(x2)..."
    python infer_longcot.py \
    --data_path data/qwen3/math500_test.jsonl \
    --output_path "${output_dir}/math500_predictions_thinking_x2.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/math500_predictions_thinking_x2.jsonl"
    echo "MATH500 processed completed"
    echo ""

    echo "Processing MATH500(x3)..."
    python infer_longcot.py \
    --data_path data/qwen3/math500_test.jsonl \
    --output_path "${output_dir}/math500_predictions_thinking_x3.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

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
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking_x1.jsonl"
    echo "Gsm8k processed completed"
    echo ""

    echo "Processing Gsm8k(x2)..."
    python infer_longcot.py \
    --data_path data/qwen3/gsm8k_test.jsonl \
    --output_path "${output_dir}/gsm8k_predictions_thinking_x2.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking_x2.jsonl"
    echo "Gsm8k processed completed"
    echo ""

    echo "Processing Gsm8k(x3)..."
    python infer_longcot.py \
    --data_path data/qwen3/gsm8k_test.jsonl \
    --output_path "${output_dir}/gsm8k_predictions_thinking_x3.jsonl" \
    --model_path "${model_name}" \
    --n_gpus "${n_gpus}" \
    --n "${n}" \
    ${thinking_arg}

    python cal_acc.py --output_path "${output_dir}/gsm8k_predictions_thinking_x3.jsonl"
    echo "Gsm8k processed completed"
    echo ""
    # --- Gsm8k End---
done

echo "✅ All tasks have been completed!"