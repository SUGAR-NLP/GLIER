"""
build_bm25_index.py — Build the BM25 index (bm25_index.pkl) for GLIER.

- Indexes LeCaRD candidate documents.
- Tokenizes with jieba (same as inference-time query_tokens) so BM25 matches correctly.
- Output {"bm25_object": BM25Okapi, "pid2idx": {pid: idx}} is compatible with
  genlcr_core.BM25Processor.

Run:
    python build_bm25_index.py
Data root can be overridden via the GLIER_DATA_ROOT env var (see config.py).

"""
import os
import sys
import json
import glob
import pickle
import jieba
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# make config.py (same dir) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

# candidate dir and output path follow config (consistent with GLIER data layout)
CANDIDATES_ROOT = config.LECARD_CANDIDATES_ROOT
SAVE_PATH = config.LECARD_BM25_INDEX


def build_index():
    print(f"[1] Scanning candidate documents: {CANDIDATES_ROOT}")
    all_files = glob.glob(os.path.join(str(CANDIDATES_ROOT), "**", "*.json"), recursive=True)

    corpus_pids = []
    tokenized_corpus = []

    for fp in tqdm(all_files, desc="Tokenizing Docs"):
        try:
            base = os.path.basename(fp)
            pid = int(os.path.splitext(base)[0])

            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            text = str(obj.get("ajjbqk", "")).strip()

            if text:
                corpus_pids.append(pid)
                tokenized_corpus.append(list(jieba.cut(text)))
        except Exception:
            continue

    if not corpus_pids:
        print(f"[WARN] No documents scanned. Check that CANDIDATES_ROOT points to the candidate dir: {CANDIDATES_ROOT}")
        return

    print(f"[2] Building BM25 object (docs = {len(corpus_pids)}) ...")
    bm25 = BM25Okapi(tokenized_corpus)

    pid2idx = {pid: i for i, pid in enumerate(corpus_pids)}

    os.makedirs(os.path.dirname(str(SAVE_PATH)), exist_ok=True)
    data_to_save = {
        "bm25_object": bm25,
        "pid2idx": pid2idx,
    }
    print(f"[3] Saving to {SAVE_PATH} ...")
    with open(str(SAVE_PATH), "wb") as f:
        pickle.dump(data_to_save, f)
    print("Done!")


if __name__ == "__main__":
    build_index()
