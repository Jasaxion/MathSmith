from collections import defaultdict
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

class PromptBatchRewardManager:
    """RewardManager that Batch uses prompt and response for reward computation."""

    def __init__(self, tokenizer, num_examine=5, compute_score=None, reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.kwargs = kwargs

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
        
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        prompts_str = []
        for i in range(len(data)):
            valid_len = valid_prompt_lengths[i]
            valid_prompt_ids = prompt_ids[i][-valid_len:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            prompts_str.append(prompt_str)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        scores = self.compute_score(
            prompts=prompts_str,
            responses=responses_str,
            **self.kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        # data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = "MathSmith"
            if already_printed.get(data_source, 0) < self.num_examine:
                response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor