import os
from tqdm import tqdm
from datasets import Dataset

from civic.config import HF_DATA_CACHE_DIR

from datasets import load_dataset


def has_abstract_more_than_1024(sample):
    abstract_text = sample["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
    return len(abstract_text.split(" ")) > 1024


def filter_and_process(dataset):
    filtered_abstracts = []
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if has_abstract_more_than_1024(sample):
            filtered_abstracts.append(
                sample["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
            )
        print(f"{total} / {len(dataset)} processed")
    print(f"{len(filtered_abstracts)/total:.2%} of data after filtering.")
    return Dataset.from_list(filtered_abstracts)


if __name__ == "__main__":
    DOWNLOAD_CACHE_PATH = os.path.join(HF_DATA_CACHE_DIR, "pubmed-central-cache")
    ds = load_dataset("pmc/open_access", cache_dir=DOWNLOAD_CACHE_PATH)
    ds["train"].map(lambda x: x["text"])
    print(f"Success. Dataset loaded from {DOWNLOAD_CACHE_PATH}")
    filtered_data = filter_and_process(ds)
