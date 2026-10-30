# Generate_legal.py
"""
Legal-elements generation script.

Run:
    python Generate_legal.py --dataset lecard
    python Generate_legal.py --dataset lecardv2
"""
import os
import sys
import json
import time
import logging
import asyncio
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import urllib3

import config
from llm import LLMEngine

# suppress urllib3 warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ===========================
# Logging configuration
# ===========================

class HTTPRequestFilter(logging.Filter):
    def filter(self, record):
        http_keywords = ["HTTP", "http", "GET", "POST", "Request", "Response", "connection"]
        return not any(k in record.getMessage() for k in http_keywords)

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("GLIER_Extraction")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(HTTPRequestFilter())

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(HTTPRequestFilter())

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


# ===========================
# Prompt builders
# ===========================

def build_lecard_prompt(content_text: str, case_name: str) -> str:
    truncated_text = content_text[:20000]
    prompt = f"""
You are a senior legal-text analysis expert. Carefully read the following criminal judgment and extract the "charge" and "legal elements".

**Input information**:
Case name: {case_name}
Case content: {truncated_text}

**Extraction requirements**:
1. **Charge**: extract all convicted charges, formatted as a list.
2. **Legal elements**:
   - extract **4 to 6** key phrases describing the core characteristics of the criminal act.
   - MUST use professional legal terminology (e.g., "consuming methamphetamine", "abusing official position").
   - STRICTLY PROHIBITED from including specific sentencing/outcome results.
   - STRICTLY PROHIBITED from directly repeating the charge itself.
   - do not duplicate elements.

**Output format**:
Return a standard JSON object directly:
{{
  "charge": ["charge1"],
  "legal_elements": ["element1", "element2", "element3", "element4"]
}}
"""
    return prompt

def build_lecardv2_prompt(content_text: str, gt_charges: List[str]) -> str:
    truncated_text = content_text[:20000]
    charge_str = "、".join(gt_charges) if gt_charges else "unknown"
    prompt = f"""
You are a senior legal-text analysis expert. Based on the provided criminal case content, extract the core "legal elements" for the case's charge.

**Case information**:
Convicted charge: [{charge_str}]
Case content: {truncated_text}

**Extraction requirements**:
1. **Goal**: extract **4 to 6** key legal elements supporting the conviction, targeting only the above [convicted charge].
2. **Content requirements**:
   - must use professional legal terminology.
   - STRICTLY PROHIBITED from including specific sentencing or case-outcome results.
   - do NOT simply repeat the charge itself.

**Output format**:
Return a standard JSON object directly:
{{
  "legal_elements": ["element1", "element2", "element3", "element4"]
}}
"""
    return prompt


# ===========================
# File processing logic
# ===========================

def process_lecard_file(file_path: Path, llm: LLMEngine) -> Dict[str, Any]:
    file_id = file_path.stem
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        content = data.get('qw', '')
        if not content or len(content) < 50:
            content = f"{data.get('ajjbqk', '')} {data.get('cpfxgc', '')}"
        case_name = data.get('ajName', '')

        if len(content) < 10:
            return {"id": file_id, "error": "empty content"}

        prompt = build_lecard_prompt(content, case_name)
        raw_res = llm.call_model_sync(prompt)
        parsed_res = llm.parse_json_output(raw_res)

        charge = parsed_res.get("charge", [])
        elements = parsed_res.get("legal_elements", [])

        if not isinstance(charge, list):
            charge = [str(charge)]
        if not isinstance(elements, list):
            elements = [str(elements)]

        return {"id": file_id, "charge": charge, "legal_elements": elements}
    except Exception as e:
        return {"id": file_id, "error": str(e)}

def process_lecardv2_file(file_path: Path, llm: LLMEngine) -> Dict[str, Any]:
    file_id = file_path.stem
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        fact = data.get('fact', '') or ""
        reason = data.get('reason', '') or ""
        content = f"Case facts: {fact}\nCourt holds: {reason}"

        gt_charge = data.get('charge', [])
        if isinstance(gt_charge, str):
            gt_charge = [gt_charge]

        if len(content) < 15:
            return {"id": file_id, "error": "Fact and Reason are empty"}

        prompt = build_lecardv2_prompt(content, gt_charge)
        raw_res = llm.call_model_sync(prompt)
        parsed_res = llm.parse_json_output(raw_res)

        elements = parsed_res.get("legal_elements", [])
        if not isinstance(elements, list):
            elements = [str(elements)]

        return {"id": file_id, "charge": gt_charge, "legal_elements": elements}
    except Exception as e:
        return {"id": file_id, "error": str(e)}


# ===========================
# Async scheduler
# ===========================

async def run_extraction_task(dataset_type: str, llm: LLMEngine):
    # configure paths
    if dataset_type == 'lecard':
        INPUT_ROOT = config.LECARD_INPUT_ROOT
        OUTPUT_DIR = config.LECARD_OUTPUT_DIR
        process_func = process_lecard_file
    else:
        INPUT_ROOT = config.LECARDV2_INPUT_ROOT
        OUTPUT_DIR = config.LECARDV2_OUTPUT_DIR
        process_func = process_lecardv2_file

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_JSON = OUTPUT_DIR / "legal_elements_all.json"
    PROCESSED_LOG = OUTPUT_DIR / "processed_ids.txt"
    FAILED_LOG = OUTPUT_DIR / "failed_ids.txt"
    LOG_FILE = OUTPUT_DIR / "extraction.log"

    logger = setup_logger(LOG_FILE)
    logger.info(f"====== Starting {dataset_type.upper()} extraction task ======")
    logger.info(f"Input dir: {INPUT_ROOT}")

    all_files = [f for f in INPUT_ROOT.rglob("*.json") if f.is_file()]
    logger.info(f"Found {len(all_files)} files to process")

    processed_files = set()
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f if line.strip())

    results = []
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} historical results")
        except:
            logger.warning("Failed to parse historical results; will start recording from scratch")

    files_to_process = [f for f in all_files if str(f) not in processed_files]
    logger.info(f"{len(files_to_process)} files remaining to process")

    executor = ThreadPoolExecutor(max_workers=config.CONCURRENCY)
    semaphore = asyncio.Semaphore(config.CONCURRENCY)

    success_count = 0
    fail_count = 0

    async def worker(file_path: Path):
        nonlocal success_count, fail_count
        str_path = str(file_path)

        async with semaphore:
            loop = asyncio.get_running_loop()
            for attempt in range(config.MAX_RETRIES):
                try:
                    res = await loop.run_in_executor(executor, process_func, file_path, llm)

                    if "error" in res:
                        raise ValueError(res["error"])

                    results.append(res)
                    success_count += 1

                    with open(PROCESSED_LOG, 'a', encoding='utf-8') as f:
                        f.write(str_path + "\n")

                    if success_count % 10 == 0:
                        logger.info(f"Progress: success {success_count} | fail {fail_count}")
                    return

                except Exception as e:
                    if attempt == config.MAX_RETRIES - 1:
                        fail_count += 1
                        logger.error(f"Failed to process {file_path.name}: {e}")
                        with open(FAILED_LOG, 'a', encoding='utf-8') as f:
                            f.write(f"{str_path}\t{str(e)}\n")
                    else:
                        await asyncio.sleep(2)

    tasks = [asyncio.create_task(worker(f)) for f in files_to_process]

    if tasks:
        total_tasks = len(tasks)
        chunks = [tasks[i:i + config.PROGRESS_EVERY] for i in range(0, total_tasks, config.PROGRESS_EVERY)]

        for i, chunk in enumerate(chunks):
            await asyncio.gather(*chunk)
            logger.info(f"Saving progress... (batch {i+1}/{len(chunks)})")
            with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        logger.info("No new tasks to execute.")

    # final full save
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 30)
    logger.info(f"Task finished. Total results: {len(results)}")
    logger.info(f"Output path: {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description="GLIER Legal Elements Extraction")
    parser.add_argument('--dataset', type=str, choices=['lecard', 'lecardv2'], required=True,
                        help="Dataset to process (lecard or lecardv2)")
    args = parser.parse_args()

    llm = LLMEngine()
    if not llm.client:
        print("API client not initialized; please check SDK and key settings.")
        return

    start = time.time()
    try:
        asyncio.run(run_extraction_task(args.dataset, llm))
    except KeyboardInterrupt:
        print("\nExtraction task stopped by user.")
    finally:
        print(f"Total time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()
