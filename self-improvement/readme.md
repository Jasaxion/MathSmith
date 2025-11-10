How to run:
`bash self_improve.sh`

Arguments:
- `yaml_template_path`: Template used for SFT hyperparameters.
- `n_sample`: Number of samples to resample for each incorrect concept.
- `max_practice_num`: Maximum number of self-improvement iterations.
- `expected_acc`: Accuracy threshold that stops the loop once met.
- `eval_data_path`: Evaluation dataset path.
- `train_data_path`: Training dataset path.

Before running, also check:
- `yaml_template`: Adjust the SFT template as needed, using the provided one for reference.
- `data/dataset_info.json`: Update the `file_name` entry to point to the generated dataset.