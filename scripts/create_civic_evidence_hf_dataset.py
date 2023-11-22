import os
from civic import config

from datasets import DatasetDict

from civic.config import DATA_HF_DIR
from civic.utils.filesystem_utils import create_folder_if_not_exists

from datasets import load_dataset


if __name__ == "__main__":
    create_folder_if_not_exists(DATA_HF_DIR)
    train_dataset = (
        load_dataset(
            "csv",
            data_files=os.path.join(
                config.DATA_PROCESSED_DIR,
                "civic_evidence_train.csv",
            ),
        )
        .remove_columns("Unnamed: 0")
        .sort("id")
    )

    test_dataset = (
        load_dataset(
            "csv",
            data_files=os.path.join(
                config.DATA_PROCESSED_DIR, "civic_evidence_test.csv"
            ),
        )
        .remove_columns("Unnamed: 0")
        .sort("id")
    )

    combined_dataset = DatasetDict(
        {"train": train_dataset["train"], "test": test_dataset["train"]}
    )

    print(combined_dataset)

    combined_dataset.save_to_disk(os.path.join(DATA_HF_DIR, "civic-evidence"))
