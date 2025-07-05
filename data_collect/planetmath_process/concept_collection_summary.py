import os
import time
import json
import openai

# === 配置部分 ===
openai.api_key = "sk-xxx"
MODEL = "gpt-4"
MD_FILE_PATH = "/home/mnt/zhanshaoxiong/pipeline/RL_math_model/open_corpus/planetmath/article-md-clean"
OUTPUT_JSONL = "/home/mnt/zhanshaoxiong/pipeline/RL_math_model/collection_concept_and_detail/collect_planetmath_grouped.jsonl"

def upload_md_file(file_path: str) -> str:
    with open(file_path, "rb") as f:
        response = openai.files.create(file=f, purpose="assistants")
    print(f"✅ Uploaded: {file_path} → file_id = {response.id}")
    return response.id

def create_assistant() -> str:
    assistant = openai.beta.assistants.create(
        name="MathConceptExtractor",
        instructions=(
            "You are a mathematical assistant. Given a Markdown (.md) file, your task is as follows:\n\n"
            "1. Determine whether the file defines or discusses mathematical concepts.\n"
            "   - If not, reply with exactly: 'Not a math concept'.\n\n"
            "2. If it does, extract all key mathematical **concepts**, **definitions**, or **theorems** mentioned.\n"
            "   - A single file may contain multiple such entries.\n"
            "   - For each entry, output one-line JSON like:\n"
            '     {"theorem": "...", "concept": "..."}\n'
            "   - The `concept` field should be concise and essential.\n\n"
            "3. At the end, append a line starting with:\n"
            "     Categories: [\"...\"]\n"
            "   which lists 1–3 high-level mathematical categories related to the content."
        ),
        model=MODEL,
        tools=[{"type": "file_search"}],
    )
    print(f"✅ Assistant created: {assistant.id}")
    return assistant.id

def run_extraction(file_id: str, assistant_id: str) -> str:
    thread = openai.beta.threads.create()
    openai.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="Please analyze this markdown file and extract its mathematical concepts and categories.",
        # file_ids=[file_id],
        attachments=[{
            "file_id": file_id,
            "tools": [{"type": "file_search"}]
        }]
    )

    run = openai.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    print("⏳ Running assistant...")

    while True:
        status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id).status
        if status == "completed":
            break
        elif status in ["failed", "cancelled", "expired"]:
            raise RuntimeError(f"❌ Run failed with status: {status}")
        time.sleep(2)

    messages = openai.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(messages.data):
        if msg.role == "assistant":
            return msg.content[0].text.value.strip()
    return ""

def parse_extracted_text(text: str):
    lines = text.splitlines()
    concepts = []
    categories = []

    for line in lines:
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                concepts.append(json.loads(line))
            except Exception:
                print(f"⚠️ Skipped invalid JSON: {line}")
        elif line.lower().startswith("categories:"):
            cat_part = line[len("categories:"):].strip()
            try:
                categories = json.loads(cat_part.replace("'", '"'))  # handle single quotes
            except Exception:
                print(f"⚠️ Failed to parse category list: {cat_part}")
    return concepts, categories

def write_grouped_output(concepts, categories, output_path, source_file):
    if not concepts:
        print(f"⚠️ No concepts extracted from {source_file}")
        return
    with open(output_path, "a", encoding="utf-8") as f:
        for concept in concepts:
            concept["source"] = source_file
            concept["categories"] = categories if categories else ["Uncategorized"]
            f.write(json.dumps(concept, ensure_ascii=False) + "\n")
    print(f"✅ Saved concepts from {source_file} with categories: {categories}")

if __name__ == "__main__":
    assistant_id = create_assistant()
    md_files = [f for f in os.listdir(MD_FILE_PATH) if f.endswith(".md")]

    if not md_files:
        raise FileNotFoundError(f"❌ No .md files in: {MD_FILE_PATH}")

    for md_file in md_files:
        full_path = os.path.join(MD_FILE_PATH, md_file)
        file_id = upload_md_file(full_path)
        result_text = run_extraction(file_id, assistant_id)

        if "Not a math concept" in result_text:
            print(f"ℹ️ Skipped non-math file: {md_file}")
            continue

        concepts, categories = parse_extracted_text(result_text)
        write_grouped_output(concepts, categories, OUTPUT_JSONL, md_file)
