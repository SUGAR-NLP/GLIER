#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_lecard.py
LeCaRD V1 final evaluation (GLIER paper Stage C③ re-ranking) -- pure ranking-pool re-rank.
Loads the trained LCRScorer, builds 5-dim features for each query's candidate pool and re-ranks,
then computes NDCG / P / MAP / MRR. Field names are aligned with run_inference.py output.
Paths and hyperparameters come from config.py, consistent with the rest of the project.
"""

import os
import json
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import config
import genlcr_core as core
from genlcr_core import LCRScorer

try:
    import jieba
except ImportError:
    print("Missing dependencies. pip install jieba")
    exit(1)

try:
    import Metrics
except ImportError:
    logging.error("Metrics.py not found. Please ensure it is in the same directory.")
    exit(1)


# ============================================================
# 1. Data loading helpers (adapted to LeCaRD V1)
# ============================================================

def load_lecard_labels_and_pool(path: str):
    logging.info(f"[DATA] Loading labels from {path}")
    if not os.path.exists(path):
        return {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = defaultdict(dict)
    pool = defaultdict(list)
    for qid_str, doc_dict in data.items():
        qid = str(qid_str).strip()
        current_pool = []
        for docid_str, grade in doc_dict.items():
            docid = str(docid_str).strip()
            try:
                g_val = int(grade)
                labels[qid][docid] = g_val
                current_pool.append(docid)
            except ValueError:
                continue
        if current_pool:
            pool[qid] = current_pool
    return labels, pool


def load_inference_cache(path: str) -> dict:
    logging.info(f"[CACHE] Loading Inference Cache from {path}")
    cache = {}
    if not os.path.exists(path):
        logging.error(f"Cache file not found: {path}")
        return cache
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        raw_id = item.get('ridx') or item.get('id')
        if raw_id is not None:
            cache[str(raw_id).strip()] = item
    logging.info(f"[CACHE] Loaded {len(cache)} items.")
    return cache


def load_docs_struct(path: str):
    pid2struct = {}
    if not os.path.exists(path):
        return pid2struct
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        iterable = data if isinstance(data, list) else data.values()
        for obj in iterable:
            pid = str(obj.get("id", "")).strip()
            if not pid:
                continue
            c = set([str(x).strip() for x in obj.get("charge", [])])
            e = set([str(x).strip() for x in obj.get("legal_elements", [])])
            pid2struct[pid] = {"charges": c, "elements": e}
    return pid2struct


# ============================================================
# 2. Ranking logic
# ============================================================

class RerankLogic:
    def __init__(self, pid2struct, bm25_processor, scorer_model=None, device="cpu"):
        self.pid2struct = pid2struct
        self.bm25_processor = bm25_processor
        self.scorer = scorer_model
        self.device = device
        jieba.initialize()

    def run_ranking(self, query_text, candidate_pids, pred_charges, pred_elements, charge_conf, elem_conf):
        if not candidate_pids:
            return {}

        # 1. BM25
        query_tokens = list(jieba.cut(query_text))
        score_dict = self.bm25_processor.get_batch_scores(query_tokens, candidate_pids)
        bm25_scores = [score_dict.get(pid, 0.0) for pid in candidate_pids]
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 1e-6 else 1.0

        # 2. Features (same order as the training script and the paper)
        features_list = []
        for i, pid in enumerate(candidate_pids):
            doc = self.pid2struct.get(pid, {"charges": set(), "elements": set()})
            hit_c = 1.0 if any(pc in doc["charges"] for pc in pred_charges) else 0.0
            hit_e_cnt = sum(1 for pe in pred_elements if pe in doc["elements"])
            hit_e_ratio = hit_e_cnt / len(pred_elements) if pred_elements else 0.0
            norm_bm25 = bm25_scores[i] / max_bm25
            features_list.append([charge_conf, elem_conf, hit_c, hit_e_ratio, norm_bm25])

        # 3. Scoring
        final_results = {}
        if self.scorer:
            inputs = torch.tensor(features_list, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.scorer(inputs)).cpu().numpy()
            for i, pid in enumerate(candidate_pids):
                final_results[pid] = {"score": float(probs[i]), "features": features_list[i]}
        else:
            for i, pid in enumerate(candidate_pids):
                final_results[pid] = {"score": features_list[i][4], "features": features_list[i]}
        return final_results


# ============================================================
# Main
# ============================================================

def init_logging_local(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "eval_lecard_metrics.log"), mode='w', encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main():
    parser = argparse.ArgumentParser()

    # paths (default from config, overridable via command line)
    parser.add_argument("--query_path", type=str, default=str(config.LECARD_QUERY_TEST))
    parser.add_argument("--document_path", type=str, default=str(config.LECARD_LEGAL_ELEMENTS_PATH))
    parser.add_argument("--label_path", type=str, default=str(config.LECARD_LABEL_PATH))
    parser.add_argument("--bm25_index_path", type=str, default=str(config.LECARD_BM25_INDEX))
    parser.add_argument("--scorer_path", type=str, default=str(config.LECARD_SCORER_DIR / "scorer_best.pth"))
    parser.add_argument("--inference_cache_path", type=str, default=str(config.LECARD_INFERENCE_CACHE_TEST))
    parser.add_argument("--output_dir", type=str, default=str(config.LECARD_SCORER_DIR / "eval"))
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--relevance_threshold", type=int, default=config.LECARD_RELEVANCE_THRESHOLD,
                        help="LeCaRD V1 relevance label threshold (positives are grade==threshold)")

    args = parser.parse_args()

    init_logging_local(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(">>> LeCaRD V1 Evaluation (Pure Ranking Pool Re-rank)")

    # 1. load data
    qid2query_raw = core.load_queries(args.query_path)
    qid2query = {str(k).strip(): v for k, v in qid2query_raw.items()}

    pid2struct = load_docs_struct(args.document_path)
    inference_cache = load_inference_cache(args.inference_cache_path)
    labels_full, ranking_pool = load_lecard_labels_and_pool(args.label_path)

    try:
        bm25_proc = core.BM25Processor(args.bm25_index_path)
    except Exception as e:
        logging.error(f"BM25 Index failed to load: {e}")
        return

    # 2. load the scorer and inspect the first-layer weights (interpretability)
    scorer = None
    if os.path.exists(args.scorer_path):
        sd = torch.load(args.scorer_path, map_location="cpu")
        dim = sd['net.0.weight'].shape[1] if 'net.0.weight' in sd else 5
        scorer = LCRScorer(input_dim=dim)
        scorer.load_state_dict(sd)
        scorer.to(device)
        scorer.eval()
        logging.info(f"MLP Scorer Loaded (Input Dim: {dim}).")

        w1 = scorer.net[0].weight.data.cpu().numpy()
        feature_names = ["Charge_Conf", "Elem_Conf", "Hit_Charge", "Hit_Elem_Ratio", "Norm_BM25"]
        avg_weights = np.mean(np.abs(w1), axis=0)
        logging.info(f"Feature Names: {feature_names}")
        logging.info(f"Avg Absolute Weight: {np.round(avg_weights, 4)}")
    else:
        logging.warning("No Scorer found. Using Norm_BM25 only.")

    # 3. evaluation
    logic = RerankLogic(pid2struct, bm25_proc, scorer, device)

    target_qids = [q for q in ranking_pool.keys() if q in qid2query and q in inference_cache]
    logging.info(f"Evaluating {len(target_qids)} Queries...")

    accum_metrics = defaultdict(list)
    detailed_logs = []

    for qid in tqdm(target_qids, desc="Evaluating"):
        pool_pids = ranking_pool[qid]
        q_text = qid2query[qid]
        q_labels = labels_full.get(qid, {})

        cache_item = inference_cache[qid]
        cleaned_charges = cache_item.get("cleaned_charges", [])
        if not cleaned_charges and "gen_charge_raw" in cache_item:
            cleaned_charges = [x.strip() for x in cache_item["gen_charge_raw"].split('|') if x.strip()]
        cleaned_elements = cache_item.get("cleaned_elements", [])
        if not cleaned_elements and "gen_element_raw" in cache_item:
            cleaned_elements = [x.strip() for x in cache_item["gen_element_raw"].split('|') if x.strip()]

        conf_c = float(cache_item.get("conf_charge", 0.0))
        conf_e = float(cache_item.get("conf_element", 0.0))

        results = logic.run_ranking(q_text, pool_pids, cleaned_charges, cleaned_elements, conf_c, conf_e)
        ranked_pids = sorted(results.keys(), key=lambda x: results[x]['score'], reverse=True)

        # LeCaRD V1 strict binary definition: grade == relevance_threshold is relevant
        golden_set = {pid for pid, grade in q_labels.items() if grade == args.relevance_threshold}

        # only the metrics reported in the paper (GLIER)
        q_metrics = Metrics.compute_paper_metrics(ranked_pids, golden_set)

        for k, v in q_metrics.items():
            accum_metrics[k].append(v)

        top_details = []
        limit = min(30, len(ranked_pids))
        for rank_idx in range(limit):
            pid = ranked_pids[rank_idx]
            res_item = results[pid]
            grade = q_labels.get(pid, 0)
            top_details.append({
                "rank": rank_idx + 1,
                "pid": pid,
                "score": res_item["score"],
                "features": {
                    "charge_conf": round(res_item['features'][0], 4),
                    "elem_conf": round(res_item['features'][1], 4),
                    "hit_charge": int(res_item['features'][2]),
                    "hit_elem_ratio": round(res_item['features'][3], 4),
                    "norm_bm25": round(res_item['features'][4], 4),
                },
                "grade": grade,
                "is_relevant": True if grade == args.relevance_threshold else False,
            })

        detailed_logs.append({
            "qid": qid,
            "gen_charge": " | ".join(cleaned_charges),
            "gen_elements": " | ".join(cleaned_elements),
            "metrics": q_metrics,
            "top30_details": top_details,
        })

    # 4. results summary -- print only the paper metrics, one line each
    logging.info("Evaluation metrics (LeCaRD V1, GLIER paper 4.1.1; NDCG@5 from Table 5):")
    if accum_metrics:
        avg = {k: np.mean(v) for k, v in accum_metrics.items()}
        for key in Metrics.PAPER_METRIC_ORDER:
            if key in avg:
                logging.info(f"  {key}: {avg[key] * 100:.2f}")
    else:
        logging.warning("No metrics computed.")

    logging.info("-" * 40)

    out_file = os.path.join(args.output_dir, "eval_metrics_details.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    logging.info(f"Details saved to {out_file}")


if __name__ == "__main__":
    main()
