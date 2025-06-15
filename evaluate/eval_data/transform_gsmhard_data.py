import json

# 假设输入文件名为input.jsonl，输出文件名为output.jsonl
with open('/media/data2/LLM/MathSmith/evaluate/eval_data/GSM-Hard.jsonl', 'r', encoding='utf-8') as infile, \
     open('/media/data2/LLM/MathSmith/evaluate/data/qwen3/gsmhard_test.jsonl', 'w', encoding='utf-8') as outfile:
    
    for idx, line in enumerate(infile, start=0):
        data = json.loads(line)
        
        # 构建新格式
        new_data = {
            "idx": idx,
            "prompt": data["input"],
            "reference_solution": f"#### {data['target']}",
            "source": "gsmhard"
        }
        
        # 写入新文件
        outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print("转换完成！")