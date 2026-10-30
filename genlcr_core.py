# genlcr_core.py
import os
import json
import logging
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    pass

# ============================================================
# 1. Logging & basic utilities
# ============================================================

def init_logging(output_dir: str, log_name: str = "process.log"):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, log_name), mode='w', encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

def load_queries(query_path: str) -> Dict[int, str]:
    """
    Robust query loader (compatible with both LeCaRD and LeCaRDv2).
    Supports:
    1. Standard JSON list: [ {...}, {...} ]
    2. Stacked JSON (JSONL): { ... }{ ... }
    3. Field compatibility: ridx/id, q/query
    """
    if not os.path.exists(query_path):
        logging.error(f"File not found: {query_path}")
        return {}

    logging.info(f"[QUERY] Loading: {query_path}")
    qid2query = {}

    with open(query_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return {}

    def parse_obj(obj):
        # prefer ridx, then id
        ridx = obj.get("ridx") if obj.get("ridx") is not None else obj.get("id")
        # prefer q, then query
        q_text = obj.get("q") if obj.get("q") is not None else obj.get("query")
        if ridx is not None:
            qid2query[int(ridx)] = str(q_text or "").strip()

    # Attempt 1: standard JSON list or dict
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for obj in data:
                parse_obj(obj)
            logging.info(f"[QUERY] (List Mode) loaded {len(qid2query)} queries.")
            return qid2query
        elif isinstance(data, dict):
            parse_obj(data)
            return qid2query
    except json.JSONDecodeError:
        pass

    # Attempt 2: stacked JSON (stream parsing) - handles LeCaRD's special format
    decoder = json.JSONDecoder()
    pos = 0
    total_len = len(content)

    while pos < total_len:
        # skip whitespace
        while pos < total_len and content[pos].isspace():
            pos += 1
        if pos >= total_len:
            break

        try:
            obj, end_pos = decoder.raw_decode(content, idx=pos)
            pos = end_pos
            if isinstance(obj, dict):
                parse_obj(obj)
        except json.JSONDecodeError:
            logging.warning(f"[QUERY] JSON decode error at pos {pos}, skipping rest.")
            break

    logging.info(f"[QUERY] (Stream Mode) loaded {len(qid2query)} queries.")
    return qid2query

def load_legal_elements_one(path: str) -> Dict[int, Tuple[List[str], List[str]]]:
    logging.info(f"[DATA] Loading document library: {path}")
    if not os.path.exists(path):
        logging.warning(f"Path does not exist: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pid2struct = {}
    iterable = data if isinstance(data, list) else data.values()

    for obj in iterable:
        try:
            pid = int(str(obj.get("id")).strip())
        except:
            continue

        charges = [str(c).strip() for c in obj.get("charge", []) if str(c).strip()]
        elements = [str(e).strip() for e in obj.get("legal_elements", []) if str(e).strip()]

        if charges or elements:
            pid2struct[pid] = (charges, elements)
    return pid2struct

def build_inverted_indexes(*paths: str):
    all_data_sources = []
    for p in paths:
        if p:
            all_data_sources.append(load_legal_elements_one(p))

    charge_index = defaultdict(set)
    element_index = defaultdict(set)
    pid2struct_full = {}

    for data_source in all_data_sources:
        for pid, (charges, elements) in data_source.items():
            if pid not in pid2struct_full:
                pid2struct_full[pid] = {'charges': set(), 'elements': set()}

            pid2struct_full[pid]['charges'].update(charges)
            pid2struct_full[pid]['elements'].update(elements)

            for c in charges:
                charge_index[c].add(pid)
            for e in elements:
                element_index[e].add(pid)

    logging.info(f"[INDEX] Index built. Unique PIDs (documents): {len(pid2struct_full)}")
    return charge_index, element_index, pid2struct_full

# ============================================================
# 2. Model definitions (InferenceEngine & MLP)
# ============================================================

class InferenceEngine:
    def __init__(self, model_path, max_source_length=512, max_target_length=128, num_beams=3, no_cuda=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.num_beams = num_beams

        logging.info(f"[MODEL] Loading mT5 from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # ensure [SEP] is in the vocabulary (added during training, idempotent here)
        if "[SEP]" not in self.tokenizer.vocab:
            self.tokenizer.add_tokens(["[SEP]"])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def generate_batch(self, input_texts: List[str], batch_size: int = 32) -> Tuple[List[str], List[float], List[float]]:
        """
        Joint generation + decoupled confidence extraction (charge / element scored separately).

        Generation uses model.generate; confidence uses a teacher-forcing forward pass to obtain
        the per-token log-prob, then splits the sequence at [SEP] into a "charge segment" and an
        "element segment", averaging and exponentiating each. Returns
        (texts, charge_confs, element_confs), consistent with the chunked-version interface.
        """
        all_texts, all_charge_confs, all_element_confs = [], [], []
        sep_token_id = self.tokenizer.convert_tokens_to_ids("[SEP]")

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_source_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=self.max_target_length,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                seqs = outputs.sequences
                decoded_batch = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)
                all_texts.extend(decoded_batch)

                # ---- teacher-forcing: per-token conditional log-prob ----
                target_seqs = seqs[:, 1:].clone()            # drop first token to align with forward
                labels = target_seqs.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100

                forward_outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels
                )
                log_probs = F.log_softmax(forward_outputs.logits, dim=-1)
                gathered = log_probs.gather(dim=-1, index=target_seqs.unsqueeze(-1)).squeeze(-1)

            # ---- split into two tracks by [SEP] ----
            for j in range(target_seqs.size(0)):
                token_ids = target_seqs[j].cpu().numpy()
                logprobs_j = gathered[j].cpu().numpy()

                charge_lp, elem_lp = [], []
                found_sep = False

                for t_id, logp in zip(token_ids, logprobs_j):
                    if t_id in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                        continue
                    if t_id == sep_token_id:
                        found_sep = True
                        continue
                    if not found_sep:
                        charge_lp.append(logp)
                    else:
                        elem_lp.append(logp)

                all_charge_confs.append(float(np.exp(np.mean(charge_lp))) if charge_lp else 0.001)
                all_element_confs.append(float(np.exp(np.mean(elem_lp))) if elem_lp else 0.001)

        return all_texts, all_charge_confs, all_element_confs

class LCRScorer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(LCRScorer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# 3. BM25 processor & cleaner
# ============================================================

class BM25Processor:
    def __init__(self, index_path: str):
        logging.info(f"[BM25] Loading index: {index_path}")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"BM25 Index not found: {index_path}")

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data.get("bm25") or data.get("bm25_object")
        self.pid2idx = data["pid2idx"]
        logging.info(f"[BM25] Loaded, document count: {len(self.pid2idx)}")

    def get_batch_scores(self, query_tokens: list, target_pids: list) -> dict:
        scores = {}
        for pid in target_pids:
            idx = -1
            # compatible with multiple pid types (int, str, numeric str)
            if pid in self.pid2idx:
                idx = self.pid2idx[pid]
            elif isinstance(pid, str) and pid.isdigit():
                try:
                    p_int = int(pid)
                    if p_int in self.pid2idx:
                        idx = self.pid2idx[p_int]
                except:
                    pass
            elif isinstance(pid, int):
                p_str = str(pid)
                if p_str in self.pid2idx:
                    idx = self.pid2idx[p_str]

            if idx != -1:
                score = 0.0
                doc_freqs = self.bm25.doc_freqs[idx]
                doc_len = self.bm25.doc_len[idx]
                for token in query_tokens:
                    freq = doc_freqs.get(token, 0)
                    if freq > 0:
                        numerator = freq * (self.bm25.k1 + 1)
                        denominator = freq + self.bm25.k1 * (1 - self.bm25.b + self.bm25.b * doc_len / self.bm25.avgdl)
                        score += self.bm25.idf.get(token, 0) * (numerator / denominator)
                scores[pid] = score
            else:
                scores[pid] = 0.0
        return scores

class TermCleaner:
    def __init__(self, valid_charges: set, valid_elements: set):
        self.valid_charges = valid_charges
        self.valid_elements = valid_elements

    def clean(self, raw_text: str, mode: str) -> list:
        """Core cleaning logic: split -> strip -> validate -> deduplicate"""
        if not raw_text:
            return []

        # compatible with " | " and "|" separators; strip whitespace on each term
        items = [x.strip() for x in raw_text.split('|') if x.strip()]

        valid_set = self.valid_charges if mode == 'charge' else self.valid_elements

        cleaned_list = []
        seen = set()

        for item in items:
            if item in valid_set:
                if item not in seen:
                    cleaned_list.append(item)
                    seen.add(item)

        return cleaned_list
