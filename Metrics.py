# Metrics.py
# Minimal IR metric implementation. Interface is kept consistent with eval_lecard.py:
#   compute_all_metrics(ranked_pids, relevant_set, k_list, map_k_list, hits_k_list, mrr_k_list)
#   compute_paper_metrics(ranked_pids, relevant_set)  -> only the metrics reported in the GLIER paper
# This file can be replaced wholesale with a stricter implementation without touching the eval script.
import numpy as np
from typing import List, Set


def _precision_at_k(ranked, relevant, k):
    if k <= 0:
        return 0.0
    hits = sum(1 for pid in ranked[:k] if pid in relevant)
    return hits / k


def _recall_at_k(ranked, relevant, k):
    if k <= 0 or not relevant:
        return 0.0
    hits = sum(1 for pid in ranked[:k] if pid in relevant)
    return hits / len(relevant)


def _ndcg_at_k(ranked, relevant, k):
    dcg = 0.0
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            dcg += 1.0 / np.log2(i + 1)
    n = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(ranked, relevant, k):
    if not relevant:
        return 0.0
    hits = 0
    sum_p = 0.0
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            hits += 1
            sum_p += hits / i
    return sum_p / len(relevant)


def _mrr_at_k(ranked, relevant, k):
    for i, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def _hits_at_k(ranked, relevant, k):
    return 1.0 if any(pid in relevant for pid in ranked[:k]) else 0.0


def compute_all_metrics(ranked_pids: List, relevant_set: Set,
                        k_list=None, map_k_list=None, hits_k_list=None, mrr_k_list=None):
    k_list = k_list or [1, 3, 5, 10]
    map_k_list = map_k_list or [3, 5, 10]
    hits_k_list = hits_k_list or [1, 3, 5, 10]
    mrr_k_list = mrr_k_list or [1, 3, 5, 10]

    metrics = {}
    for k in k_list:
        metrics[f"NDCG@{k}"] = _ndcg_at_k(ranked_pids, relevant_set, k)
        metrics[f"P@{k}"] = _precision_at_k(ranked_pids, relevant_set, k)
        metrics[f"R@{k}"] = _recall_at_k(ranked_pids, relevant_set, k)
        metrics[f"Hits@{k}"] = _hits_at_k(ranked_pids, relevant_set, k)
    for k in map_k_list:
        metrics[f"MAP@{k}"] = _average_precision(ranked_pids, relevant_set, k)
    for k in mrr_k_list:
        metrics[f"MRR@{k}"] = _mrr_at_k(ranked_pids, relevant_set, k)
    return metrics


# ============================================================
# Metrics reported in the GLIER paper
#   - 4.1.1 main evaluation: MAP, P@3, R@3, R@5, Hits@3, Hits@5, MRR@5
#   - Table 5 feature ablation additionally reports: NDCG@5
# Only these 8 metrics are computed, strictly aligned with the paper tables.
# MAP is computed over the full candidate pool ranking (standard MAP, no cutoff);
# the rest use the cut-offs given in the paper.
# ============================================================
PAPER_METRIC_ORDER = [
    "MAP", "NDCG@5", "P@3", "R@3", "R@5", "Hits@3", "Hits@5", "MRR@5",
]


def compute_paper_metrics(ranked_pids: List, relevant_set: Set):
    k_full = max(len(ranked_pids), 1)
    return {
        # Standard MAP: average precision over the full pool ranking, no cutoff
        "MAP": _average_precision(ranked_pids, relevant_set, k_full),
        # reported in Table 5
        "NDCG@5": _ndcg_at_k(ranked_pids, relevant_set, 5),
        # main evaluation metrics from 4.1.1
        "P@3": _precision_at_k(ranked_pids, relevant_set, 3),
        "R@3": _recall_at_k(ranked_pids, relevant_set, 3),
        "R@5": _recall_at_k(ranked_pids, relevant_set, 5),
        "Hits@3": _hits_at_k(ranked_pids, relevant_set, 3),
        "Hits@5": _hits_at_k(ranked_pids, relevant_set, 5),
        "MRR@5": _mrr_at_k(ranked_pids, relevant_set, 5),
    }
