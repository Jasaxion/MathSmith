import json

input_path = '/media/data2/LLM/MathSmith/evaluate/eval_data/olympiad.jsonl'
ouput_path = '/media/data2/LLM/MathSmith/evaluate/data/qwen3/olympiad_test.jsonl'

def transform_dataset(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for idx, line in enumerate(infile):  # idx从0开始自动编号
            data = json.loads(line)
            print(f"\\boxed{ '{' + data['final_answer'][0] + '}'}\n#### {data['final_answer'][0]}")
            # input()
            new_entry = {
                "idx": idx,  # 从0开始的自增编号
                "prompt": data["question"],  # 直接使用原始question
                "reference_solution": f"\\boxed{ '{' + data['final_answer'][0] + '}'}\n#### {data['final_answer'][0]}",  # 保留原始answer格式
                "source": "olympiad"  # 固定source
            }
            
            outfile.write(json.dumps(new_entry, ensure_ascii=False) + '\n')

# 使用示例
transform_dataset(input_path, ouput_path)