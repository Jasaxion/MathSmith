import os
import time
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Filter out Markdown files from PlanetMath that do not contain mathematical content.

MY_KEY = "sk-xxx"
DEEPSEEK_MODEL = "gpt-4o-mini"
SOURCE_DIR = "./planetmath/article-md-clean"
NO_MATH_DIR = "./planetmath/a_clean_no_math"
MAX_CONTENT_LEN = 2000
MAX_WORKERS = 4

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "math_file_filter.log")
PROCESSED_FILE_LIST = os.path.join(LOG_DIR, "processed_files.txt")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

def load_processed_files():
    if not os.path.exists(PROCESSED_FILE_LIST):
        return set()
    with open(PROCESSED_FILE_LIST, "r") as f:
        return set(line.strip() for line in f.readlines())

def append_to_processed(fname):
    with open(PROCESSED_FILE_LIST, "a") as f:
        f.write(fname + "\n")

def check_math_presence(md_content: str, fname: str, client) -> bool:
    sys_prompt = (
        "You are an expert classifier. Given the content of a Markdown file, "
        "respond with 'Yes' if it includes mathematical knowledge (definitions, theorems, formal logic, etc.), "
        "and 'No' if it does not. Respond only with 'Yes' or 'No'.\n\n"
    )
    user_prompt = f"Content:\n{md_content.strip()[:MAX_CONTENT_LEN]}"
    retry_count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False
            )
            reply = response.choices[0].message.content.strip()
            logging.info(f"[{fname}] response: {reply}")
            return reply.lower().startswith("yes")
        except Exception as e:
            retry_count += 1
            logging.warning(f"[{fname}] API failed ({retry_count}th tries): {e}")
            time.sleep(5)

# process a single file
def process_file(fname, processed_files):
    if fname in processed_files:
        logging.info(f"Skipping: {fname}")
        return

    fpath = os.path.join(SOURCE_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        logging.warning(f"[{fname}] Skipping unreadable file: {e}")
        return

    if not content:
        logging.warning(f"[{fname}] Skipping empty file")
        return

    logging.info(f"Processing file: {fname}")
    client = OpenAI(api_key=MY_KEY)
    has_math = check_math_presence(content, fname, client)

    if not has_math:
        os.makedirs(NO_MATH_DIR, exist_ok=True)
        dest_path = os.path.join(NO_MATH_DIR, fname)
        shutil.move(fpath, dest_path)
        logging.info(f"[{fname}] Non-math-related files have been moved.")
    else:
        logging.info(f"[{fname}] Preserve math-related files")

    append_to_processed(fname)

def filter_markdown_files_parallel():
    processed_files = load_processed_files()
    md_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".md")]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_file, fname, processed_files) for fname in md_files]
        for future in as_completed(futures):
            _ = future.result()

if __name__ == "__main__":
    logging.info("Starting math content filtering...")
    filter_markdown_files_parallel()
    logging.info("All files have been processed.")
