import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from civic.utils.filesystem_utils import check_file_exists
from civic.config import DATA_PROCESSED_DIR

FILE_NOT_FOUND_ERROR_MESSAGE = "Please first run the script process_civic_evidence_data"

EVIDENCE_LEVEL_TO_NUMBER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


class CivicEvidenceDataSet(Dataset):
    @staticmethod
    def full_train_dataset(tokenizer, tokenizer_max_length):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv")
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, tokenizer_max_length)
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def full_test_dataset(tokenizer, tokenizer_max_length):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, tokenizer_max_length)
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def accepted_only_train_dataset(tokenizer, tokenizer_max_length):
        path_to_file = os.path.join(
            DATA_PROCESSED_DIR, "civic_evidence_train_accepted_only.csv"
        )
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, tokenizer_max_length)
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def accepted_only_test_dataset(tokenizer, tokenizer_max_length):
        path_to_file = os.path.join(
            DATA_PROCESSED_DIR, "civic_evidence_test_accepted_only.csv"
        )
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, tokenizer_max_length)
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def _map_to_numerical(evidence_level):
        return EVIDENCE_LEVEL_TO_NUMBER.get(evidence_level, np.nan)

    def __init__(self, path_to_data, tokenizer, tokenizer_max_length):
        df = pd.read_csv(path_to_data)
        self.evidence_levels = df["evidenceLevel"]
        self.labels = df["evidenceLevel"].map(self._map_to_numerical)
        self.abstracts = df["sourceAbstract"]
        self.prepend_string = df["prependString"]
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        abstract = str(self.abstracts[idx])
        prepend_string = str(self.prepend_string[idx])
        input_string = "Metadata:\n" + prepend_string + "\n" + "Abstract:\n" + abstract
        label = self.labels[idx]
        evidence_level = self.evidence_levels[idx]

        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # Tokenize the abstract and convert it into input features
        encoding = self.tokenizer.encode_plus(
            input_string,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        return {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "label": torch.tensor(label),
            "evidence_level": evidence_level,
        }
