import json
import argparse

def convert_jsonl_to_alpaca(input_file_path, output_file_path):
    """
    将包含 "prompt" 和 "output" 的 JSONL 文件转换为 Alpaca 格式的 JSON 文件。

    Args:
        input_file_path (str): 输入的 JSONL 文件路径。
        output_file_path (str): 输出的 JSON 文件路径。
    """
    alpaca_data = []
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                # 解析每一行的 JSON 对象
                original_data = json.loads(line.strip())

                # 提取 prompt 和 output
                prompt = original_data.get("prompt", "") or original_data.get("problem", "")
                output = original_data.get("output", "") or original_data.get("answer", "")

                # 构建新的 Alpaca 格式字典
                new_entry = {
                    "instruction": instruction,
                    "input": prompt,
                    "output": output
                }
                alpaca_data.append(new_entry)

        # 将转换后的数据写入新的 JSON 文件
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(alpaca_data, outfile, indent=4, ensure_ascii=False)

        print(f"✅ 转换成功！文件已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"❌ 错误: 输入文件未找到 -> {input_file_path}")
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 解析 JSONL 文件时出错。请检查文件格式。错误信息: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="将 JSONL 文件转换为 Alpaca 格式的 JSON 文件。")
    parser.add_argument("input_file", help="输入的 JSONL 文件路径 (例如: data.jsonl)")
    parser.add_argument("output_file", help="输出的 JSON 文件路径 (例如: alpaca_data.json)")

    args = parser.parse_args()

    # 执行转换函数
    convert_jsonl_to_alpaca(args.input_file, args.output_file)