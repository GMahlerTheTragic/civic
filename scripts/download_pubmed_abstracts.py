import os

from civic.config import HF_DATA_CACHE_DIR
from civic.utils.filesystem_utils import create_folder_if_not_exists

from datasets import load_dataset


if __name__ == "__main__":
    DOWNLOAD_CACHE_PATH = os.path.join(HF_DATA_CACHE_DIR, "pubmed-central-cache")
    create_folder_if_not_exists(DOWNLOAD_CACHE_PATH)
    df = load_dataset("pmc/open_access", cache_dir=DOWNLOAD_CACHE_PATH, num_proc=8)
    print(f"Success. Dataset cashed at {DOWNLOAD_CACHE_PATH}")
