import json
import argparse

def convert_jsonl_to_alpaca(input_file_path, output_file_path):
    """
    Convert a JSONL file containing "prompt" and "output" into a JSON file in Alpaca format.

    Args:
        input_file_path (str): Input JSONL file path.
        output_file_path (str): Output JSON file path.

    python convert_alpace.py data.jsonl alpaca_data.json
    """
    alpaca_data = []
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                original_data = json.loads(line.strip())

                prompt = original_data.get("prompt", "") or original_data.get("problem", "")
                output = original_data.get("output", "") or original_data.get("answer", "")

                new_entry = {
                    "instruction": instruction,
                    "input": prompt,
                    "output": output
                }
                alpaca_data.append(new_entry)

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(alpaca_data, outfile, indent=4, ensure_ascii=False)

        print(f"Conversion successful! The file has been saved to {output_file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found -> {input_file_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSONL file. Please check the file format. Error message: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a JSONL file into a JSON file in Alpaca format.")
    parser.add_argument("input_file", help="Input JSONL file path (e.g., data.jsonl)")
    parser.add_argument("output_file", help="Output JSON file path (e.g., alpaca_data.json)")

    args = parser.parse_args()
    convert_jsonl_to_alpaca(args.input_file, args.output_file)