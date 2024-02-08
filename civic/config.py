import os


MODULE_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CIVIC_DIR = os.path.dirname(MODULE_ROOT_DIR)
DATA_DIR = os.path.join(CIVIC_DIR, "data")
DATA_RESULTS_DIR = os.path.join(DATA_DIR, "04_results")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "02_processed")
DATA_RAW_DIR = os.path.join(DATA_DIR, "01_raw")
DATA_HF_DIR = os.path.join(DATA_DIR, "03_huggingface")
MODEL_STORAGE_DIR = os.path.join(CIVIC_DIR, "models")
MODEL_CHECKPOINT_DIR = os.environ.get(
    "MODEL_CHECKPOINT_DIR", os.path.join(MODEL_STORAGE_DIR, "model_checkpoints")
)
HF_DATA_CACHE_DIR = os.environ.get(
    "HF_DATA_CACHE_DIR", os.path.join(DATA_DIR, "huggingface_cache")
)
