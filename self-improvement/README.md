# Self-Improvement Pipeline

## Quick Start

Execute the self-improvement process with:
```bash
bash self_improve.sh
```

## Script Arguments

- `yaml_template_path`: Template file for SFT hyperparameter configuration
- `n_sample`: Number of samples to resample for each incorrect concept
- `max_practice_num`: Maximum number of self-improvement iterations
- `expected_acc`: Target accuracy threshold (loop terminates when achieved)
- `eval_data_path`: Path to evaluation dataset
- `train_data_path`: Path to training dataset

## Pre-run Configuration

### 1. YAML Template
Adjust the SFT template according to your requirements. The provided template serves as a reference configuration.

### 2. Dataset Configuration
Update `data/dataset_info.json`:
- Modify the `file_name` entry to point to your generated dataset

## Training Data Preparation

The script `self_improve.sh` uses pre-generated training data:
```
"../data_collect/mathsmith-test/MathSmith-HC-test-collection-question-with-shortcotanswer.json"
```

### About the Variant Collection
This dataset contains pre-sampled question variants to accelerate the self-improvement pipeline:

- Generation Method: Use `QM_sampler` and `answer_sampler` to create question and reasoning chain variants
- Pre-processed Dataset: Download our pre-sampled variant collection from Hugging Face:
  - [MathSmith-Self-Improvement-VariantSet](https://huggingface.co/datasets/Jasaxion/MathSmith-Self-Improvement-VarientSet)
- Note: The provided dataset contains 10 samples per variant. You can regenerate the collection with more samples as needed.

### Custom Generation
To create your own variant library:
1. Use the sampling tools to generate question and answer chain variants
2. Update the `train_data_path` to point to your custom dataset
3. Ensure the dataset format matches the expected structure