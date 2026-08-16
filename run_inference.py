# run_inference.py
# Offline mT5 inference (GLIER joint generation + constrained cleaning)
#
# Aligned with the paper / train_genlcr.py: uses the joint prompt
#   "predict charge and elements: {query}"  ->  "{charge} [SEP] {elements}"
# instead of the old two-step prompt ("predict charge:" / "predict elements:").
# The first two of the 5 features (charge_conf, element_conf) produced here are
# consumed by the scorer stage.
#
# Note: this file only changes the inference logic; genlcr_core.py is untouched.
#   - base genlcr_core.InferenceEngine.generate_batch returns a 2-tuple (texts, confs),
#     in which case conf_charge and conf_element both take the joint-sequence confidence.
#   - if the Fine-GLIER genlcr_core_chunk.py is dropped into this directory and imported,
#     generate_batch returns a 3-tuple (texts, charge_confs, elem_confs) for decoupled confidence.
import os
import json
import argparse
import logging
import torch
from tqdm import tqdm

import genlcr_core as core
import config  # your config file


def main():
    parser = argparse.ArgumentParser(
        description="Batch mT5 joint-generation inference with constrained cleaning (config-driven, aligned with paper/training)"
    )

    # core switches
    parser.add_argument("--dataset", type=str, choices=["LeCaRD", "LeCaRDv2"], required=True)
    parser.add_argument("--split", type=str, choices=["train", "test", "dev"], required=True)

    # query path is not hard-coded in config, so it is passed in
    parser.add_argument("--query_path", type=str, required=True, help="Path to the specific query file")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the inference cache")

    # hardware & hyperparameters (overridable)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=32)
    # allow overriding the model path from config via command line
    parser.add_argument("--model_path", type=str, default=None, help="Override config default checkpoint path if needed")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ========================================================
    # dynamically map config paths
    # ========================================================
    if args.dataset == "LeCaRD":
        legal_elements_path = config.LECARD_LEGAL_ELEMENTS_PATH
        model_path = args.model_path or (config.CHECKPOINT_DIR_LECARD / "epoch-16").as_posix()
    else:  # LeCaRDv2
        legal_elements_path = config.LECARDV2_LEGAL_ELEMENTS_PATH
        model_path = args.model_path or (config.CHECKPOINT_DIR_LECARDV2 / "epoch-20").as_posix()

    # ========================================================

    log_filename = f"inference_{args.dataset}_{args.split}_data.log"
    output_dir = os.path.dirname(os.path.abspath(args.output_file))
    core.init_logging(output_dir, log_name=log_filename)

    logging.info(f"========== Start inference pipeline (joint generation, config-driven) ==========")
    logging.info(f"Dataset   : {args.dataset} ({args.split})")
    logging.info(f"Elements  : {legal_elements_path}")
    logging.info(f"Model     : {model_path}")
    logging.info(f"Query     : {args.query_path}")
    logging.info(f"Output    : {args.output_file}")
    logging.info(f"=======================================================")

    # 1. load queries
    qid2query = core.load_queries(args.query_path)
    if not qid2query:
        return

    sorted_qids = sorted(list(qid2query.keys()))
    q_texts = [qid2query[qid] for qid in sorted_qids]

    # 2. build the constrained whitelist
    logging.info(f"[INIT] Loading legal-elements whitelist...")
    charge_idx, elem_idx, _ = core.build_inverted_indexes(legal_elements_path)
    if not charge_idx or not elem_idx:
        return
    cleaner = core.TermCleaner(valid_charges=set(charge_idx.keys()), valid_elements=set(elem_idx.keys()))

    # 3. load model
    engine = core.InferenceEngine(model_path)

    # safety check: [SEP] is a special token added during training; the checkpoint must contain it
    if "[SEP]" not in engine.tokenizer.vocab:
        logging.error(
            "[SEP] is not in the tokenizer vocabulary! Make sure model_path points to a checkpoint "
            "trained by train_genlcr.py (which calls add_tokens(['[SEP]']) during training)."
        )
        return

    # 4. batch joint generation & cleaning
    results = []
    total_batches = (len(sorted_qids) + args.batch_size - 1) // args.batch_size

    for i in tqdm(range(0, len(sorted_qids), args.batch_size), total=total_batches, desc=f"Inferencing {args.split}"):
        batch_qids = sorted_qids[i: i + args.batch_size]
        batch_queries = q_texts[i: i + args.batch_size]

        # --- joint generation (identical to the training prompt) ---
        prompts = [f"predict charge and elements: {q}" for q in batch_queries]
        out = engine.generate_batch(prompts, batch_size=len(batch_queries))

        # compatible with base genlcr_core (2-tuple: texts, confs)
        # and chunked genlcr_core_chunk (3-tuple: texts, charge_confs, elem_confs)
        if len(out) == 3:
            raw_outputs, conf_charges, conf_elements = out
        else:
            raw_outputs, confs = out
            conf_charges = confs
            conf_elements = confs

        # --- parse & constrained filtering (cleaning) ---
        for j in range(len(batch_qids)):
            gen_text = raw_outputs[j]

            # split by the training target format "{charge} [SEP] {elements}"
            parts = gen_text.split("[SEP]")
            raw_charge = parts[0].strip() if len(parts) > 0 else ""
            raw_element = parts[1].strip() if len(parts) > 1 else ""

            # clean / validate (whitelist filters out hallucinations)
            cleaned_c = cleaner.clean(raw_charge, mode='charge')
            cleaned_e = cleaner.clean(raw_element, mode='element')

            results.append({
                "ridx": batch_qids[j],
                "query": batch_queries[j],

                "gen_charge_raw": raw_charge,
                "gen_element_raw": raw_element,
                "full_output_raw": gen_text,

                "cleaned_charges": cleaned_c,
                "cleaned_elements": cleaned_e,

                # --- confidence (joint for base version, decoupled for chunk version) ---
                "conf_charge": conf_charges[j],
                "conf_element": conf_elements[j],
            })

    # 5. save results
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f">>> {args.dataset} {args.split} inference finished! {len(results)} records.")


if __name__ == "__main__":
    main()
