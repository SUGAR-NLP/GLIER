#!/bin/bash
# run_feature_generation.sh
# Offline mT5 inference for later evaluation/testing.

# exit on error
set -e

GPU_ID="2"
DATASET="LeCaRD"

# Root directory holding your LeCaRD / LeCaRDv2 datasets.
# Override with:  GLIER_DATA_ROOT=/path/to/data ./run_feature_generation.sh
DATA_ROOT="${GLIER_DATA_ROOT:-./data/raw}"

echo "==================================================="
echo "Step 1: Generating features for MLP TRAIN split"
echo "==================================================="
python run_inference.py \
    --dataset ${DATASET} \
    --split train \
    --query_path ${DATA_ROOT}/${DATASET}/query/train_query.json \
    --output_file ./checkpoints/${DATASET}/MLP/inference_cache_train.json \
    --gpu_id ${GPU_ID} \
    --batch_size 32

echo "==================================================="
echo "Step 2: Generating features for MLP TEST split"
echo "==================================================="
python run_inference.py \
    --dataset ${DATASET} \
    --split test \
    --query_path ${DATA_ROOT}/${DATASET}/query/test_query.json \
    --output_file ./checkpoints/${DATASET}/MLP/inference_cache_test.json \
    --gpu_id ${GPU_ID} \
    --batch_size 32

echo ">>> Feature generation completed for both train and test splits!"
