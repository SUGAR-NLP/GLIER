#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_genlcr.py (GLIER Framework - Joint Generative Inference)

Single-step generation training script based on mT5 / T5, for both LeCaRD and LeCaRDv2.
Notes:
1. Unifies the data-reading logic for LeCaRD and LeCaRDv2.
2. Introduces config.py for centralized path and default-parameter management.
3. Input prompt: "predict charge and elements: ..."
4. Target: "Charge [SEP] Elements"

Run:
    python train_genlcr.py --dataset lecard --gpus 0
    python train_genlcr.py --dataset lecardv2 --gpus 1
"""

import os
import re
import json
import random
import argparse
import logging
import glob
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

# config file
import config

# ============================================================
# Utility functions
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_genlcr.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='a', encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Log written to: {log_path}")

# ============================================================
# Data loading
# ============================================================

def load_legal_elements(path: str) -> Dict[int, List[Tuple[str, str]]]:
    """
    Load a single legal-element label file.
    Returns: {pid: [(charge_str, elements_str), ...]}
    """
    assert os.path.exists(path), f"File not found: {path}"
    logging.info(f"[LEGAL] Loading label file: {path}")

    raw_data_list = []

    # try standard JSON first; on failure fall back to tolerant stream parsing (mainly for merge artifacts)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data_list = json.load(f)
    except json.JSONDecodeError as e:
        logging.warning(f"[LEGAL] JSON format error ({e}), trying stream/regex recovery mode...")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        decoder = json.JSONDecoder()
        idx, length = 0, len(content)
        while idx < length and content[idx] != '{':
            idx += 1
        while idx < length:
            try:
                obj, end_idx = decoder.raw_decode(content, idx)
                raw_data_list.append(obj)
                idx = end_idx
                while idx < length and content[idx] != '{':
                    idx += 1
            except json.JSONDecodeError:
                break

    pid2struct: Dict[int, List[Tuple[str, str]]] = {}
    skipped = 0

    for obj in raw_data_list:
        try:
            # compatible with id (LeCaRD) and pid (LeCaRDv2)
            raw_id = obj.get("id", obj.get("pid"))
            if raw_id is None:
                skipped += 1
                continue
            pid = int(str(raw_id).strip())
        except:
            skipped += 1
            continue

        charges = obj.get("charge", [])
        elements = obj.get("legal_elements", [])

        if not charges or not elements:
            skipped += 1
            continue

        charge_str = " | ".join([str(c).strip() for c in charges if str(c).strip()])
        elements_str = " | ".join([str(e).strip() for e in elements if str(e).strip()])

        if not charge_str or not elements_str:
            skipped += 1
            continue

        pid2struct.setdefault(pid, []).append((charge_str, elements_str))

    logging.info(f"[LEGAL] Valid PIDs: {len(pid2struct)}, skipped entries: {skipped}")
    return pid2struct


def collect_doc_texts(candidates_root: str, dataset_type: str) -> Dict[int, str]:
    """
    Recursively read all json under candidates_root.
    Selects different field-parsing logic based on dataset_type.
    """
    assert os.path.exists(candidates_root), f"candidates_root does not exist: {candidates_root}"
    logging.info(f"[DOC] Scanning candidate set: {candidates_root} | mode: {dataset_type}")

    all_files = glob.glob(os.path.join(candidates_root, "**", "*.json"), recursive=True)

    pid2text: Dict[int, str] = {}
    skipped = 0
    for fp in all_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            skipped += 1
            continue

        # get PID (prefer from json, otherwise from filename)
        pid = obj.get("pid", obj.get("id"))
        if pid is None:
            stem, _ = os.path.splitext(os.path.basename(fp))
            try:
                pid = int(stem)
            except:
                skipped += 1
                continue
        else:
            pid = int(pid)

        # get text (separate LeCaRD vs LeCaRDv2 logic)
        if dataset_type == "lecard":
            txt = str(obj.get("ajjbqk", "")).strip()
        elif dataset_type == "lecardv2":
            fact = str(obj.get("fact", "")).strip()
            reason = str(obj.get("reason", "")).strip()
            txt = f"{fact} {reason}".strip()
        else:
            txt = ""

        if not txt:
            skipped += 1
            continue

        pid2text[pid] = txt

    logging.info(f"[DOC] Valid texts: {len(pid2text)}, skipped: {skipped}")
    return pid2text


# ============================================================
# Dataset
# ============================================================

class HierarchicalGenByDocDataset(Dataset):
    def __init__(self, pid2text: Dict[int, str], pid2labels: Dict[int, List[Tuple[str, str]]]):
        self.instances = []
        logging.info("[DATASET] Building Dataset ...")
        skipped = 0
        for pid, text in pid2text.items():
            if pid not in pid2labels:
                skipped += 1
                continue
            for li, (charge_str, elements_str) in enumerate(pid2labels[pid]):
                self.instances.append({
                    "pid": pid,
                    "text": text,
                    "charge": charge_str,
                    "elements": elements_str,
                    "label_source_idx": li
                })
        logging.info(f"[DATASET] Instances: {len(self.instances)} (skipped without labels: {skipped})")

    def __len__(self): return len(self.instances)
    def __getitem__(self, idx): return self.instances[idx]


# ============================================================
# Training & checkpoint logic
# ============================================================

def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = glob.glob(os.path.join(output_dir, "epoch-*"))
    if not checkpoints:
        return None

    valid_checkpoints = []
    for cp in checkpoints:
        match = re.match(r"epoch-(\d+)", os.path.basename(cp))
        if match:
            valid_checkpoints.append((int(match.group(1)), cp))

    if not valid_checkpoints:
        return None
    valid_checkpoints.sort(key=lambda x: x[0], reverse=True)
    return valid_checkpoints[0]


def train(args, model, tokenizer, train_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if n_gpu > 1 and not args.no_cuda:
        logging.info(f"[TRAIN] Using {n_gpu} GPUs for DataParallel training!")
        model = nn.DataParallel(model)
    model.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=lambda x: x
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    t_total = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * args.warmup_ratio), t_total)

    start_epoch, global_step = 0, 0
    last_ckpt_info = get_last_checkpoint(args.output_dir)

    if last_ckpt_info:
        saved_epoch, ckpt_path = last_ckpt_info
        logging.info(f"[RESUME] Detected checkpoint: {ckpt_path}, restoring model weights...")
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).state_dict())
        else:
            model.load_state_dict(AutoModelForSeq2SeqLM.from_pretrained(ckpt_path).state_dict())

        training_state_path = os.path.join(ckpt_path, "training_state.pt")
        if os.path.exists(training_state_path):
            state_dict = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(state_dict["optimizer"])
            scheduler.load_state_dict(state_dict["scheduler"])
            global_step = state_dict["global_step"]
            start_epoch = saved_epoch
            logging.info(f"[RESUME] Success! Resuming from Epoch {start_epoch + 1}")
        else:
            start_epoch = saved_epoch
            logging.warning(f"[RESUME] training_state.pt not found; only model weights restored.")
    else:
        logging.info("[TRAIN] No checkpoint detected, training from scratch.")

    model.train()

    def compute_gen_loss(inputs, targets):
        enc = tokenizer(inputs, padding=True, truncation=True, max_length=args.max_source_length, return_tensors="pt").to(device)
        labels = tokenizer(targets, padding=True, truncation=True, max_length=args.max_target_length, return_tensors="pt")["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        labels = labels.to(device)

        outputs = model(**enc, labels=labels)
        loss = outputs.loss
        return loss.mean() if n_gpu > 1 else loss

    for epoch in range(start_epoch, args.num_train_epochs):
        epoch_loss = 0.0
        logging.info(f"--- Epoch {epoch + 1} / {args.num_train_epochs} (Joint Generation) ---")

        for step, batch in enumerate(train_loader):
            texts = [b["text"] for b in batch]
            charges = [b["charge"] for b in batch]
            elements = [b["elements"] for b in batch]

            # joint generation prompt and target
            inputs = [f"predict charge and elements: {t}" for t in texts]
            targets = [f"{c} [SEP] {e}" for c, e in zip(charges, elements)]

            final_loss = compute_gen_loss(inputs, targets)

            optimizer.zero_grad()
            final_loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += final_loss.item()

            if global_step % args.logging_steps == 0:
                logging.info(f"Epoch {epoch + 1} | Step {global_step} | Loss: {final_loss.item():.4f}")

        # save checkpoint
        save_dir = os.path.join(args.output_dir, f"epoch-{epoch + 1}")
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        torch.save({
            "epoch": epoch + 1,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }, os.path.join(save_dir, "training_state.pt"))
        logging.info(f"Saved checkpoint to {save_dir}")

    logging.info("Training Finished.")

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # core dataset selection
    parser.add_argument("--dataset", type=str, choices=["lecard", "lecardv2"], required=True,
                        help="Target dataset to train on")

    # GPU control
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use, e.g. '0,1'")
    parser.add_argument("--no_cuda", action="store_true")

    # other training parameters (allow overriding config via command line)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # dynamically attach config parameters
    args.model_name_or_path = config.DEFAULT_MODEL_PATH
    args.num_train_epochs = config.TRAIN_EPOCHS
    args.batch_size = config.TRAIN_BATCH_SIZE
    args.learning_rate = config.LEARNING_RATE
    args.warmup_ratio = 0.1
    args.max_grad_norm = 1.0
    args.max_target_length = config.MAX_TARGET_LENGTH

    # dispatch paths and truncation lengths by dataset
    if args.dataset == "lecard":
        args.candidates_root = str(config.LECARD_CANDIDATES_ROOT)
        args.legal_elements_path = str(config.LECARD_LEGAL_ELEMENTS_PATH)
        args.output_dir = str(config.CHECKPOINT_DIR_LECARD)
        args.max_source_length = config.MAX_SOURCE_LENGTH_LECARD
    else:
        args.candidates_root = str(config.LECARDV2_CANDIDATES_ROOT)
        args.legal_elements_path = str(config.LECARDV2_LEGAL_ELEMENTS_PATH)
        args.output_dir = str(config.CHECKPOINT_DIR_LECARDV2)
        args.max_source_length = config.MAX_SOURCE_LENGTH_LECARDV2

    init_logging(args.output_dir)
    set_seed(42)

    if not args.no_cuda and torch.cuda.is_available():
        logging.info(f"CUDA: {os.environ.get('CUDA_VISIBLE_DEVICES')} | GPUs: {torch.cuda.device_count()}")
    else:
        logging.info("Using CPU.")

    # load and start training pipeline
    pid2labels = load_legal_elements(args.legal_elements_path)
    pid2text = collect_doc_texts(args.candidates_root, dataset_type=args.dataset)
    train_dataset = HierarchicalGenByDocDataset(pid2text, pid2labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if "[SEP]" not in tokenizer.vocab:
        logging.info("Adding [SEP] token to tokenizer...")
        tokenizer.add_tokens(["[SEP]"])

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    train(args, model, tokenizer, train_dataset)

if __name__ == "__main__":
    main()
