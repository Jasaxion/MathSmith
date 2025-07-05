运行：
bash self_imporve.sh

参数说明：
- yaml_template_path：sft的参数模板
- n_sample：每个错误concept 重采样多少sample
- max_practice_num：最大循环次数
- expect_acc：期望acc，到达后停止循环
- eval_data_path：测试数据路径
- train_data_path：训练数据库路径

除了运行参数以外需要修改的地方：
- yaml_template: sft的参数模板，可参考现有的修改
- dataset_info.json: 修改file_name