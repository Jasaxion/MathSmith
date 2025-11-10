import os
import json
import time
import openai

# Using GPT-4o to extract math concepts and explanation from PlanetMath markdown files

openai.api_key = "sk-xxx"
MODEL = "gpt-4o-mini"
# crawled markdown from planetmath
MD_FILE_PATH = "./planetmath/article-md-clean"
OUTPUT_JSONL = "./planetmath/collect_planetmath_grouped.jsonl"

def extract_concepts_from_text(md_content: str) -> str:
    prompt = (
        "You are a mathematical assistant.\n"
        "Given the following markdown content from a mathematical article, extract key concepts and their explanations.\n\n"
        "Instructions:\n"
        "1. If the content does NOT define or discuss a math concept, respond with exactly: Not a math concept\n"
        "2. Otherwise, identify all important math concepts mentioned.\n"
        "   - For each concept, ensure it is explicitly mentioned or defined in the given markdown article, and provide a brief explanation based strictly on the article content.\n"
        "   - The explanation should be concise and relevant to the concept.\n"
        "   - Concept and Explanation must be written in LaTeX-compatible format, using proper mathematical notation where appropriate.\n"
        "   - listing 1–3 broad mathematical categories related to the content.\n"
        "   - For each concept, output a JSON format:\n"
        "     [{\"Concept\": \"...\", \"Explanation\": \"...\", \"Categories\": [\"...\"]}, ...]}\n"
        f"Markdown Math Content:\n{md_content}"
    )

    while True:
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API call failed — {e}")
            print(f"Retrying in 5 seconds...")
            time.sleep(5)

import re

def parse_json_block(response_text: str):
    if "Not a math concept" in response_text:
        return [], False

    try:
        json_start = response_text.find("[")
        json_end = response_text.rfind("]") + 1
        json_block = response_text[json_start:json_end]

        # Try escaping \ to \\ (excludes ones that are already double slashes)
        json_block_safe = re.sub(r'(?<!\\)\\(?![\\nt"])', r'\\\\', json_block)

        items = json.loads(json_block_safe)
        concepts = []
        for item in items:
            if (
                isinstance(item, dict)
                and "Concept" in item
                and "Explanation" in item
                and "Categories" in item
            ):
                concepts.append({
                    "Concept": item["Concept"],
                    "Explanation": item["Explanation"],
                    "Categories": item["Categories"]
                })
            else:
                print(f"Missing expected fields in: {item}")
        return concepts, True
    except Exception as e:
        print(f"JSON block parsing failed: {e}")
        return [], False



def write_output(concepts, output_path, source_file):
    if not concepts:
        print(f"No concepts extracted from {source_file}")
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for concept in concepts:
            concept["source"] = source_file
            f.write(json.dumps(concept, ensure_ascii=False) + "\n")
    print(f"Saved concepts from {source_file} ({len(concepts)} entries)")

if __name__ == "__main__":
    md_files = [f for f in os.listdir(MD_FILE_PATH) if f.endswith(".md")]

    if not md_files:
        raise FileNotFoundError(f"No .md files in: {MD_FILE_PATH}")

    for md_file in md_files:
        full_path = os.path.join(MD_FILE_PATH, md_file)
        with open(full_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        result_text = extract_concepts_from_text(md_content)
        concepts, valid = parse_json_block(result_text)

        if not valid:
            print(f"Skipped non-math file: {md_file}")
            continue

        write_output(concepts, OUTPUT_JSONL, md_file)
