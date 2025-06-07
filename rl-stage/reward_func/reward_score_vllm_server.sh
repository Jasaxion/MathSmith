export VLLM_USE_V1=1
vllm serve Qwen/Qwen3_30B_A3B \
    --host 0.0.0.0 \
    --port 59876 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
    --max-model-len 65536 \
    --dtype bfloat16 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.98