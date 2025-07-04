import os
import time
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ========== 配置 ==========
MY_KEY = "sk-xxx"
# client = OpenAI(api_key=MY_KEY, base_url="https://api.deepseek.com")
DEEPSEEK_MODEL = "deepseek-chat"
SOURCE_DIR = "/home/mnt/zhanshaoxiong/pipeline/RL_math_model/open_corpus/planetmath/article-md-clean"
NO_MATH_DIR = "/home/mnt/zhanshaoxiong/pipeline/RL_math_model/open_corpus/planetmath/a_clean_no_math"
MAX_CONTENT_LEN = 2000
MAX_WORKERS = 4

LOG_DIR = "./logs"
LOG_FILE = os.path.join(LOG_DIR, "math_file_filter.log")
PROCESSED_FILE_LIST = os.path.join(LOG_DIR, "processed_files.txt")

# ========== 日志设置 ==========
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

# ========== 加载已处理文件 ==========
def load_processed_files():
    if not os.path.exists(PROCESSED_FILE_LIST):
        return set()
    with open(PROCESSED_FILE_LIST, "r") as f:
        return set(line.strip() for line in f.readlines())

def append_to_processed(fname):
    with open(PROCESSED_FILE_LIST, "a") as f:
        f.write(fname + "\n")

# ========== API 判断函数 ==========
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
            logging.info(f"[{fname}] 模型回应: {reply}")
            return reply.lower().startswith("yes")
        except Exception as e:
            retry_count += 1
            logging.warning(f"[{fname}] DeepSeek API 调用失败 (第 {retry_count} 次): {e}")
            time.sleep(5)

# ========== 单个文件处理 ==========
def process_file(fname, processed_files):
    if fname in processed_files:
        logging.info(f"🟡 已处理过的文件跳过: {fname}")
        return

    fpath = os.path.join(SOURCE_DIR, fname)
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as e:
        logging.warning(f"[{fname}] 跳过无法读取的文件: {e}")
        return

    if not content:
        logging.warning(f"[{fname}] 空文件跳过")
        return

    logging.info(f"🔍 正在检测文件: {fname}")
    client = OpenAI(api_key=MY_KEY, base_url="https://api.deepseek.com")
    has_math = check_math_presence(content, fname, client)

    if not has_math:
        os.makedirs(NO_MATH_DIR, exist_ok=True)
        dest_path = os.path.join(NO_MATH_DIR, fname)
        shutil.move(fpath, dest_path)
        logging.info(f"➡️ [{fname}] 非数学文件已移动")
    else:
        logging.info(f"✅ [{fname}] 保留数学文件")

    append_to_processed(fname)

# ========== 主函数 ==========
def filter_markdown_files_parallel():
    processed_files = load_processed_files()
    md_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".md")]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_file, fname, processed_files) for fname in md_files]
        for future in as_completed(futures):
            _ = future.result()

if __name__ == "__main__":
    logging.info("🚀 启动数学文件过滤（并发+去重）")
    filter_markdown_files_parallel()
    logging.info("🏁 所有文件检测完成")
