model_name='/media/data2/LLM/models/qwen/Qwen/Qwen2___5-Math-1___5B'
n_gpus=1

python self_improve.py \
  --output_dir "./output_test/" \
  --model_path "${model_name}" \
  --eval_data_path "/media/data2/LLM/MathSmith/data_collect/mathsmith-test/MathSmith-HC-test.jsonl" \
  --train_data_path "/media/data2/LLM/MathSmith/data_collect/mathsmith-test/MathSmith-HC-test-collection-question-with-shortcotanswer.json" \
  --yaml_template_path "./sft/MathSmith_Evaluator_Qwen_Math_1_5B.yaml" \
  --n_sample 1 \
  --max_practice_num 3 \
  --expected_acc 0.75 \
  --n_gpus "${n_gpus}" \
  --max_len 4096