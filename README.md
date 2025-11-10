# MathSmith

**MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy**

[![Paper](https://img.shields.io/badge/arXiv-2508.05592-b31b1b.svg)](https://arxiv.org/abs/2508.05592)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()

## Overview

MathSmith is a framework for enhancing mathematical reasoning capabilities of large language models by generating challenging synthetic problems from scratch. Unlike methods that modify existing problems, MathSmith creates novel problems through a reinforced policy, ensuring diversity and scalability.

## Pipeline

The MathSmith framework consists of four main stages:

1. **Concept Collection**: Randomly sample concept–explanation pairs from PlanetMath to ensure data independence.

2. **Supervised Fine-tuning (SFT)**: Train the model on collected concept–explanation pairs to establish foundational understanding.

3. **Reinforcement Learning (RL)**: Optimize the model using PPO/GRPO with rewards based on:
   - Structural validity
   - Reasoning complexity  
   - Answer consistency

4. **Weakness-Focused Self-Improvement**: Iteratively identify and address model weaknesses by generating targeted problem variants.

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/MathSmith.git
cd MathSmith
pip install -r requirements.txt
```

### Data Collection

Collect concept–explanation pairs from PlanetMath:

```bash
cd data_collect/planetmath_process
# Follow instructions to process PlanetMath data
```

### Training

**SFT Stage:**
```bash
cd sft-stage
# Configure MathSmith_Questioner-Qwen3-8B.yaml
# Run SFT training
```

**RL Stage:**
```bash
cd rl-stage/train_script
bash rl_mathsmith.sh
```

### Problem Generation

Generate mathematical problems using the trained model:

```bash
python QM_sampler.py
```

### Evaluation

Evaluate on benchmarks (GSM8K, MATH-500, AIME2024, AIME2025, OlympiadBench):

```bash
cd evaluate
bash eval.sh
```

### Self-Improvement

Run the weakness-focused improvement pipeline:

```bash
cd self-improvement
bash self_improve.sh
```

## Repository Structure

```
MathSmith/
├── data_collect/          # Concept collection and data processing
├── sft-stage/             # Supervised fine-tuning scripts
├── rl-stage/              # Reinforcement learning training
│   ├── train_script/      # RL training scripts
│   └── reward_func/       # Reward function implementations
├── answer_sampler/        # Answer generation for problems
├── evaluate/              # Evaluation scripts and benchmarks
├── self-improvement/      # Weakness-focused improvement pipeline
├── utils/                 # Utility functions
└── QM_sampler.py         # Problem generation script
```

## Results

MathSmith consistently outperforms baselines across five benchmarks under both short and long chain-of-thought settings:
- **Easy & Medium**: GSM8K, MATH-500
- **Hard**: AIME2024, AIME2025, OlympiadBench

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhan2025mathsmith,
  title={MathSmith: Towards Extremely Hard Mathematical Reasoning by Forging Synthetic Problems with a Reinforced Policy},
  author={Zhan, Shaoxiong and Lai, Yanlin and Lu, Ziyu and Lin, Dahua and Yang, Ziqing and Tan, Fei},
  journal={arXiv preprint arXiv:2508.05592},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
