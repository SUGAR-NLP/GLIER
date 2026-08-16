#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_scorer_LeCaRD.py
Train the MLP Scorer (LeCaRD V1) -- GLIER paper Stage C② (feature engineering) + Stage C③ (MLP training).
Paper mapping: 5-dim features [charge_conf, element_conf, hit_charge, hit_elem_ratio, norm_bm25]
               -> LCRScorer(input_dim=5) -> re-rank candidate documents.
Features:
  - data source adapted to LeCaRD V1 (label_top30_dict.json contains both labels and the candidate pool)
  - optional Hard Negative Mining (prefer high-BM25 negatives via --hard_ratio)
  - Per-Query Normalization (norm_bm25 divided by the max within the current query's candidate pool)
Paths and hyperparameters are centralized in config.py, consistent with the rest of the project.
"""

import os
import json
import pickle
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import config
import genlcr_core as core

try:
    import jieba
except ImportError:
    print("Error: pip install jieba")
    exit(1)


# ============================================================
# 1. Data loading helpers (adapted to LeCaRD V1)
# ============================================================

def load_lecard_labels_and_pool(path: str):
    """
    Load LeCaRD V1's label_top30_dict.json.
    Returns:
      labels: {qid: {docid: grade}}
      pool:   {qid: [docid, ...]}
    """
    logging.info(f"[DATA] Loading LeCaRD labels from {path}")
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return {}, {}

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    labels = defaultdict(dict)
    pool = defaultdict(list)

    for qid_str, doc_dict in data.items():
        try:
            qid = int(qid_str)
        except ValueError:
            continue

        current_pool = []
        for docid_str, grade in doc_dict.items():
            try:
                docid = int(docid_str)
                grade = int(grade)
                labels[qid][docid] = grade
                current_pool.append(docid)
            except ValueError:
                continue

        if current_pool:
            pool[qid] = current_pool

    logging.info(f"[DATA] Loaded pool for {len(pool)} queries.")
    return labels, pool


def load_inference_cache(path: str) -> dict:
    logging.info(f"[CACHE] Loading inference cache from {path}")
    if not os.path.exists(path):
        logging.warning(f"Cache file not found: {path} (skipped).")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    cache = {}
    for item in data:
        # LeCaRD V1 usually uses id or ridx
        ridx = item.get('ridx') or item.get('id')
        if ridx is not None:
            cache[int(ridx)] = item
    logging.info(f"[CACHE] Loaded {len(cache)} cached items.")
    return cache


# ============================================================
# 2. Dataset & feature extraction
# ============================================================

class ScorerDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def extract_training_data(
    queries: dict,
    ranking_pool: dict,
    labels_full: dict,
    inference_cache: dict,
    pid2struct: dict,
    bm25_processor: core.BM25Processor,
    cleaner: core.TermCleaner = None,
    threshold: int = 3,
    desc: str = "Extracting",
    cache_file: str = None,
):
    """Feature extraction pipeline (Per-Query Normalization)."""
    if cache_file and os.path.exists(cache_file):
        logging.info(f"[{desc}] Found feature cache, loading quickly: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['features'], cached['labels']

    logging.info(f"[{desc}] Generating features (Thr={threshold}, Norm=PerQuery)...")

    features = []
    labels = []
    query_items = list(queries.items())

    for ridx, query_text in tqdm(query_items, desc=desc, mininterval=1.0):
        candidate_pids = ranking_pool.get(ridx)
        cache_item = inference_cache.get(ridx)
        if not candidate_pids or not cache_item:
            continue

        # decoupled confidence from the generative model (run_inference.py already produced distinct values)
        charge_conf = float(cache_item.get('conf_charge', 0.0))
        element_conf = float(cache_item.get('conf_element', 0.0))

        pred_charges = cache_item.get('cleaned_charges')
        pred_elements = cache_item.get('cleaned_elements')

        # fallback cleaning
        if pred_charges is None:
            raw_c = cache_item.get('gen_charge_raw') or ""
            pred_charges = cleaner.clean(raw_c, mode='charge') if cleaner else []
        if pred_elements is None:
            raw_e = cache_item.get('gen_element_raw') or ""
            pred_elements = cleaner.clean(raw_e, mode='element') if cleaner else []

        # BM25 computation & local normalization (Per-Query Normalization)
        query_tokens = list(jieba.cut(query_text))
        score_dict = bm25_processor.get_batch_scores(query_tokens, candidate_pids)
        bm25_vals = [score_dict.get(pid, 0.0) for pid in candidate_pids]

        local_max = max(bm25_vals) if bm25_vals else 1.0
        if local_max <= 1e-6:
            local_max = 1.0

        q_grades = labels_full.get(ridx, {})

        for pid in candidate_pids:
            doc_struct = pid2struct.get(pid)
            if not doc_struct:
                continue

            # Feature: Hit Charge
            hit_c = 1.0 if any(pc in doc_struct['charges'] for pc in pred_charges) else 0.0

            # Feature: Hit Element Ratio
            doc_elems_set = set(doc_struct['elements'])
            hit_e_count = sum(1 for pe in pred_elements if pe in doc_elems_set)
            hit_e_ratio = hit_e_count / len(pred_elements) if pred_elements else 0.0

            # Feature: Norm BM25 (divided by local max)
            norm_bm25 = score_dict.get(pid, 0.0) / local_max

            # [5-dim features] in the paper's order
            features.append([charge_conf, element_conf, hit_c, hit_e_ratio, norm_bm25])

            # Label: LeCaRD V1 positive = grade >= threshold (=3)
            grade = int(q_grades.get(pid, 0))
            labels.append(1.0 if grade >= threshold else 0.0)

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels, 'normalization': 'per_query'}, f)

    return features, labels


def balance_training_data_hard_mining(features, labels, neg_ratio=3.0, hard_ratio=0.0):
    """
    [Optional] Hard Negative Mining: prefer negatives with higher norm_bm25 (feature[4]).
    hard_ratio=0.0 falls back to pure random negative sampling.
    """
    logging.info(f"Negative sampling (Ratio={neg_ratio}, HardMining={hard_ratio})...")

    labels_np = np.array(labels)
    pos_indices = np.where(labels_np > 0.5)[0]
    neg_indices = np.where(labels_np < 0.5)[0]

    if len(pos_indices) == 0:
        logging.warning("No positive samples found in training data!")
        return features, labels

    n_pos = len(pos_indices)
    n_neg_total_keep = int(n_pos * neg_ratio)

    if n_neg_total_keep > len(neg_indices):
        n_neg_total_keep = len(neg_indices)

    # === Hard Negative Mining logic ===
    logging.info("Sorting negative samples by BM25 score...")
    neg_scores = []
    for idx in neg_indices:
        neg_scores.append((idx, features[idx][4]))

    neg_scores.sort(key=lambda x: x[1], reverse=True)
    sorted_neg_indices = [x[0] for x in neg_scores]

    n_hard = int(n_neg_total_keep * hard_ratio)
    n_random = n_neg_total_keep - n_hard

    hard_negs = sorted_neg_indices[:n_hard]
    remaining_pool = sorted_neg_indices[n_hard:]

    if len(remaining_pool) < n_random:
        random_negs = remaining_pool
    else:
        random_negs = random.sample(remaining_pool, n_random)

    final_neg_indices = list(hard_negs) + list(random_negs)

    final_indices = list(pos_indices) + final_neg_indices
    random.shuffle(final_indices)

    logging.info(f"Balanced Data: Pos {len(pos_indices)} | Neg {len(final_neg_indices)} "
                 f"(Hard: {len(hard_negs)}, Random: {len(random_negs)})")

    return [features[i] for i in final_indices], [labels[i] for i in final_indices]


# ============================================================
# Main loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0")

    # paths (default from config, overridable via command line)
    parser.add_argument("--bm25_index_path", type=str, default=str(config.LECARD_BM25_INDEX))
    parser.add_argument("--train_query_path", type=str, default=str(config.LECARD_QUERY_TRAIN))
    parser.add_argument("--test_query_path", type=str, default=str(config.LECARD_QUERY_TEST))
    parser.add_argument("--label_path", type=str, default=str(config.LECARD_LABEL_PATH))
    parser.add_argument("--legal_elements_path", type=str, default=str(config.LECARD_LEGAL_ELEMENTS_PATH))
    parser.add_argument("--inference_cache_path", type=str, default=str(config.LECARD_INFERENCE_CACHE_TRAIN))
    parser.add_argument("--inference_cache_path_test", type=str, default=str(config.LECARD_INFERENCE_CACHE_TEST))
    parser.add_argument("--scorer_save_path", type=str, default=str(config.LECARD_SCORER_DIR / "scorer_best.pth"))
    parser.add_argument("--cache_dir", type=str, default=str(config.LECARD_SCORER_DIR / "cache_features"))

    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=config.SCORER_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.SCORER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.SCORER_LEARNING_RATE)
    parser.add_argument("--neg_ratio", type=float, default=config.SCORER_NEG_RATIO)
    parser.add_argument("--hard_ratio", type=float, default=config.SCORER_HARD_RATIO,
                        help="Fraction of high-BM25 negatives (0.0=pure random, 1.0=all hard)")
    parser.add_argument("--pos_threshold", type=int, default=config.LECARD_RELEVANCE_THRESHOLD)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    core.init_logging(os.path.dirname(args.scorer_save_path), log_name="train_scorer_lecard.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(">>> LeCaRD V1 Scorer Training (One-step + Optional Hard Negative Mining)")
    logging.info(f"    Train Query : {args.train_query_path}")
    logging.info(f"    Label      : {args.label_path}")
    logging.info(f"    Hard Ratio : {args.hard_ratio}")
    jieba.initialize()

    # 1. load components
    bm25_proc = core.BM25Processor(args.bm25_index_path)
    train_qs = core.load_queries(args.train_query_path)
    test_qs = core.load_queries(args.test_query_path)  # used as validation set

    logging.info(f"Loaded {len(train_qs)} train queries, {len(test_qs)} test queries.")

    charge_idx, elem_idx, pid2struct = core.build_inverted_indexes(args.legal_elements_path)
    cleaner = core.TermCleaner(set(charge_idx.keys()), set(elem_idx.keys()))

    labels_full, ranking_pool = load_lecard_labels_and_pool(args.label_path)

    # inference cache: merge train and test (skip if test is missing)
    inference_cache = load_inference_cache(args.inference_cache_path)
    test_cache = load_inference_cache(args.inference_cache_path_test)
    inference_cache.update(test_cache)
    logging.info(f"Merged inference cache size: {len(inference_cache)}")

    if not ranking_pool or not inference_cache:
        logging.error("Missing pool or inference cache.")
        return

    # 2. filter valid queries (must be in pool and have inference results, and contain positives)
    def filter_valid(qs):
        valid = {}
        for qid, qtext in qs.items():
            if qid in ranking_pool and qid in inference_cache:
                pool_pids = ranking_pool[qid]
                q_grades = labels_full.get(qid, {})
                if any(int(q_grades.get(pid, 0)) >= args.pos_threshold for pid in pool_pids):
                    valid[qid] = qtext
        return valid

    train_qs_valid = filter_valid(train_qs)
    test_qs_valid = filter_valid(test_qs)

    logging.info(f"Valid Queries (with pos>={args.pos_threshold} in pool): "
                 f"Train {len(train_qs_valid)}, Val {len(test_qs_valid)}")

    # 3. feature extraction
    os.makedirs(args.cache_dir, exist_ok=True)

    train_feats_raw, train_labels_raw = extract_training_data(
        queries=train_qs_valid,
        ranking_pool=ranking_pool,
        labels_full=labels_full,
        inference_cache=inference_cache,
        pid2struct=pid2struct,
        bm25_processor=bm25_proc,
        cleaner=cleaner,
        threshold=args.pos_threshold,
        desc="TrainFeat",
        cache_file=os.path.join(args.cache_dir, "train_feats.pkl"),
    )

    train_feats, train_labels = balance_training_data_hard_mining(
        train_feats_raw,
        train_labels_raw,
        neg_ratio=args.neg_ratio,
        hard_ratio=args.hard_ratio,
    )

    val_feats, val_labels = extract_training_data(
        queries=test_qs_valid,
        ranking_pool=ranking_pool,
        labels_full=labels_full,
        inference_cache=inference_cache,
        pid2struct=pid2struct,
        bm25_processor=bm25_proc,
        cleaner=cleaner,
        threshold=args.pos_threshold,
        desc="ValFeat",
        cache_file=os.path.join(args.cache_dir, "test_feats.pkl"),
    )

    logging.info(f"Feature dims sanity check -> first sample: {train_feats[0] if train_feats else 'EMPTY'}")

    # 4. training loop
    logging.info(f">>> Start Training MLP. BatchSize={args.batch_size}")
    train_loader = DataLoader(ScorerDataset(train_feats, train_labels), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ScorerDataset(val_feats, val_labels), batch_size=args.batch_size, shuffle=False)

    model = core.LCRScorer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        all_probs, all_labels_list = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                probs = torch.sigmoid(model(bx)).cpu().numpy()
                all_probs.extend(probs)
                all_labels_list.extend(by.numpy())

        try:
            val_auc = roc_auc_score(all_labels_list, all_probs)
        except Exception:
            val_auc = 0.5

        logging.info(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

        scheduler.step(val_auc)

        # checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            epoch_path = args.scorer_save_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_path)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), args.scorer_save_path)
            logging.info(f"  [+] Saved Best Model (AUC={best_auc:.4f})")


if __name__ == "__main__":
    main()
