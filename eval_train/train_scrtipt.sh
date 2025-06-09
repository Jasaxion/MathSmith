set -x

export WANDB_API_KEY=
export WANDB_PROJECT="MathSmith-thinking-Qwen3-8B-sft"

llamafactory-cli train ./MathSmith_w_thinking_sft.yaml