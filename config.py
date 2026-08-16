# config.py
import os
from pathlib import Path

# ==========================================
# 1. API & LLM configuration (mask before pushing to GitHub)
# ==========================================
API_KEY = "YOUR_API_KEY_HERE"  # replace with your own API key
MODEL_NAME = "YOUR_MODEL_NAME_HERE"  # model to use, e.g. "gpt-4" or "mt5-base"

# ==========================================
# 2. Data paths
#    DATA_ROOT points to the directory that holds your LeCaRD /
#    LeCaRDv2 datasets. Override it with the GLIER_DATA_ROOT env var,
#    or edit the sub-paths below to match your own layout.
#    No absolute server paths are committed to the repository.
# ==========================================
DATA_ROOT = Path(os.environ.get("GLIER_DATA_ROOT", "./data/raw"))

# LeCaRD v1 data (adjust sub-paths to your dataset layout)
LECARD_CANDIDATES_ROOT = DATA_ROOT / "LeCaRD" / "candidates"
LECARD_LEGAL_ELEMENTS_PATH = DATA_ROOT / "LeCaRD" / "legal_elements_merged_all.json"  # charge + legal elements per document
LECARD_OUTPUT_DIR = Path("./data/LeCaRD/document")
LECARD_INPUT_ROOT = LECARD_CANDIDATES_ROOT  # raw case documents consumed by Generate_legal.py

LECARD_QUERY_TRAIN = DATA_ROOT / "LeCaRD" / "query" / "train_query.json"
LECARD_QUERY_TEST = DATA_ROOT / "LeCaRD" / "query" / "test_query.json"
LECARD_BM25_INDEX = DATA_ROOT / "LeCaRD" / "bm25_index.pkl"  # BM25 index, used as the 5th MLP feature

# LeCaRD v2 data (adjust sub-paths to your dataset layout)
LECARDV2_CANDIDATES_ROOT = DATA_ROOT / "LeCaRDv2" / "candidates"
LECARDV2_LEGAL_ELEMENTS_PATH = DATA_ROOT / "LeCaRDv2" / "legal_elements_merged.json"  # charge + legal elements per document
LECARDV2_OUTPUT_DIR = Path("./data/LeCaRDv2/document")
LECARDV2_INPUT_ROOT = LECARDV2_CANDIDATES_ROOT  # raw case documents consumed by Generate_legal.py

LECARDV2_QUERY_TRAIN = DATA_ROOT / "LeCaRDv2" / "query" / "train_query.json"
LECARDV2_QUERY_TEST = DATA_ROOT / "LeCaRDv2" / "query" / "test_query.json"
LECARDV2_BM25_INDEX = DATA_ROOT / "LeCaRDv2" / "bm25_index.pkl"  # BM25 index, used as the 5th MLP feature

# Async concurrency & scheduling
CONCURRENCY = 100
PROGRESS_EVERY = 50
MAX_RETRIES = 3
TIMEOUT_SECONDS = 120

# ==========================================
# 3. Model training paths & configuration
# ==========================================
# Default mT5 checkpoint path (override with GLIER_MODEL_PATH, or pass --model_path)
DEFAULT_MODEL_PATH = os.environ.get("GLIER_MODEL_PATH", "./models/mt5-base")

# Checkpoint save paths
CHECKPOINT_DIR_LECARD = Path("./checkpoints/LeCaRD/mt5_genlcr_onestep")
CHECKPOINT_DIR_LECARDV2 = Path("./checkpoints/LeCaRDv2/mt5_genlcr_onestep")

# Default hyperparameters (adjust as needed)
TRAIN_BATCH_SIZE = 2
LEARNING_RATE = 3e-5
MAX_SOURCE_LENGTH_LECARD = 512   # max length, adjust as needed
MAX_SOURCE_LENGTH_LECARDV2 = 512  # max length, adjust as needed
MAX_TARGET_LENGTH = 256
TRAIN_EPOCHS = 20

# ==========================================
# 4. MLP Scorer configuration
# ==========================================
# LeCaRD v1 relevance annotations (label_top30_dict.json contains both labels and the candidate pool)
LECARD_LABEL_PATH = DATA_ROOT / "LeCaRD" / "label" / "label_top30_dict.json"

# Generative inference cache (produced by run_inference.py; consumed by the MLP for conf_charge/conf_element and cleaned_*)
# train / test are kept in two separate files and merged during training
LECARD_INFERENCE_CACHE_TRAIN = Path("./checkpoints/LeCaRD/MLP/inference_cache_train.json")
LECARD_INFERENCE_CACHE_TEST = Path("./checkpoints/LeCaRD/MLP/inference_cache_test.json")

# MLP scoring model output directory
LECARD_SCORER_DIR = Path("./checkpoints/LeCaRD/MLP/scorer")

# Relevance threshold (LeCaRD v1 uses grades 1~3; positives are grades >= 3, evaluation uses grade == 3)
LECARD_RELEVANCE_THRESHOLD = 3

# MLP training hyperparameters
SCORER_BATCH_SIZE = 64
SCORER_LEARNING_RATE = 1e-4
SCORER_EPOCHS = 100
SCORER_NEG_RATIO = 3.0        # negative : positive ratio
SCORER_HARD_RATIO = 0.0       # fraction of high-BM25 negatives (0.0 = pure random negative sampling)
