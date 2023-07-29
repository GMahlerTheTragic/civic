import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from civic.utils.filesystem_utils import check_file_exists
from config import PROJECT_ROOT


class CivicEvidenceDataSet(Dataset):
    @staticmethod
    def full_train_dataset(tokenizer):
        path_to_file = os.path.join(
            PROJECT_ROOT, "data/02_processed/civic_evidence_train.csv"
        )
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, None)
        raise FileNotFoundError(
            "Please first run the script process_civic_evidence_data"
        )

    @staticmethod
    def full_test_dataset(tokenizer):
        path_to_file = os.path.join(
            PROJECT_ROOT, "data/02_processed/civic_evidence_test.csv"
        )
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, None)
        raise FileNotFoundError(
            "Please first run the script process_civic_evidence_data"
        )

    @staticmethod
    def accepted_only_train_dataset(tokenizer):
        path_to_file = os.path.join(
            PROJECT_ROOT, "data/02_processed/civic_evidence_train_accepted_only.csv"
        )
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, None)
        raise FileNotFoundError(
            "Please first run the script process_civic_evidence_data"
        )

    @staticmethod
    def accepted_only_test_dataset(tokenizer):
        path_to_file = os.path.join(
            PROJECT_ROOT, "data/02_processed/civic_evidence_test_accepted_only.csv"
        )
        if check_file_exists(path_to_file):
            return CivicEvidenceDataSet(path_to_file, tokenizer, None)
        raise FileNotFoundError(
            "Please first run the script process_civic_evidence_data"
        )

    def __init__(self, path_to_data, tokenizer, tokenizer_max_length):
        df = pd.read_csv(path_to_data)
        self.evidence_levels = df["evidenceLevel"]
        self.labels = pd.factorize(df["evidenceLevel"])[0]
        self.abstracts = df["sourceAbstract"]
        self.prepend_string = df["prependString"]
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        abstract = str(self.abstracts[idx])
        prepend_string = str(self.prepend_string[idx])
        input_string = abstract + prepend_string
        label = self.labels[idx]
        evidence_level = self.evidence_levels[idx]

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
