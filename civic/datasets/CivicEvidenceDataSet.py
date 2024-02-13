import os
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer

from civic.utils.filesystem_utils import check_file_exists
from civic.config import DATA_PROCESSED_DIR

FILE_NOT_FOUND_ERROR_MESSAGE = "Please first run the script process_civic_evidence_data"

EVIDENCE_LEVEL_TO_NUMBER = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


class CivicEvidenceDataSet(Dataset):
    @staticmethod
    def full_train_dataset(tokenizer, tokenizer_max_length, use_prepend_string=False):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df, tokenizer, tokenizer_max_length, use_prepend_string
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def train_dataset_unique_abstracts(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train_ua.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df, tokenizer, tokenizer_max_length, use_prepend_string
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def train_val_dataset_unique_abstracts(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file_train = os.path.join(
            DATA_PROCESSED_DIR, "civic_evidence_train_ua.csv"
        )
        path_to_file_val = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val_ua.csv")
        if check_file_exists(path_to_file_train) and check_file_exists(
            path_to_file_val
        ):
            df_train = pd.read_csv(path_to_file_train)
            df_val = pd.read_csv(path_to_file_val)
            return CivicEvidenceDataSet(
                pd.concat([df_train, df_val], axis=0).reset_index(),
                tokenizer,
                tokenizer_max_length,
                use_prepend_string,
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def train_val_dataset(tokenizer, tokenizer_max_length, use_prepend_string=False):
        path_to_file_train = os.path.join(
            DATA_PROCESSED_DIR, "civic_evidence_train.csv"
        )
        path_to_file_val = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv")
        if check_file_exists(path_to_file_train) and check_file_exists(
            path_to_file_val
        ):
            df_train = pd.read_csv(path_to_file_train)
            df_val = pd.read_csv(path_to_file_val)
            return CivicEvidenceDataSet(
                pd.concat([df_train, df_val], axis=0).reset_index(),
                tokenizer,
                tokenizer_max_length,
                use_prepend_string,
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def full_validation_dataset(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df, tokenizer, tokenizer_max_length, use_prepend_string
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def validation_dataset_unique_abstracts(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val_ua.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df, tokenizer, tokenizer_max_length, use_prepend_string
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def full_test_dataset(tokenizer, tokenizer_max_length, use_prepend_string=False):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df, tokenizer, tokenizer_max_length, use_prepend_string
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def test_dataset_unique_abstracts(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_ua.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df, tokenizer, tokenizer_max_length, use_prepend_string
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def test_dataset_gpt4_unique_abstracts(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file = os.path.join(
            DATA_PROCESSED_DIR, "civic_evidence_test_gpt4_ua.csv"
        )
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df,
                tokenizer,
                tokenizer_max_length,
                use_prepend_string,
                return_ref_tokens_for_ig=True,
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def full_test_dataset_gpt4(
        tokenizer, tokenizer_max_length, use_prepend_string=False
    ):
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            return CivicEvidenceDataSet(
                df,
                tokenizer,
                tokenizer_max_length,
                use_prepend_string,
                return_ref_tokens_for_ig=True,
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def full_test_dataset_long_only(
        tokenizer, tokenizer_max_length, longer_than=512, use_prepend_string=False
    ):
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            df["input"] = (
                "Metadata:\n"
                + df["prependString"]
                + "\n"
                + "Abstract:\n"
                + df["sourceAbstract"]
            )
            df["input_len"] = df["input"].map(lambda x: len(tokenizer_bert.encode(x)))
            df = (
                df.loc[df["input_len"] > longer_than]
                .reset_index(drop=True)
                .drop(columns=["input", "input_len"], axis=1)
            )
            return CivicEvidenceDataSet(
                df,
                tokenizer,
                tokenizer_max_length,
                use_prepend_string,
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def test_dataset_long_only_unique_abstracts(
        tokenizer, tokenizer_max_length, longer_than=512, use_prepend_string=False
    ):
        tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        path_to_file = os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_ua.csv")
        if check_file_exists(path_to_file):
            df = pd.read_csv(path_to_file)
            df["input_len"] = df["sourceAbstract"].map(
                lambda x: len(tokenizer_bert.encode(x))
            )
            df = (
                df.loc[df["input_len"] > longer_than]
                .reset_index(drop=True)
                .drop(columns=["input_len"], axis=1)
            )
            return CivicEvidenceDataSet(
                df,
                tokenizer,
                tokenizer_max_length,
                use_prepend_string,
            )
        raise FileNotFoundError(FILE_NOT_FOUND_ERROR_MESSAGE)

    @staticmethod
    def _map_to_numerical(evidence_level):
        return EVIDENCE_LEVEL_TO_NUMBER.get(evidence_level, np.nan)

    def __init__(
        self,
        df,
        tokenizer,
        tokenizer_max_length,
        use_prepend_string=False,
        return_ref_tokens_for_ig=False,
    ):
        self.evidence_levels = df["evidenceLevel"].astype(
            pd.CategoricalDtype(["A", "B", "C", "D", "E"])
        )
        self.labels = (
            df["evidenceLevel"]
            .map(self._map_to_numerical)
            .astype(pd.CategoricalDtype([0, 1, 2, 3, 4]))
        )
        self.abstracts = df["sourceAbstract"]
        self.evidence_item_id = df["id"]
        self.prepend_string = df["prependString"]
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.return_ref_tokens_for_ig = return_ref_tokens_for_ig
        self.use_prepend_string = use_prepend_string

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        abstract = str(self.abstracts[idx])
        prepend_string = str(self.prepend_string[idx])
        if self.use_prepend_string:
            input_string = (
                "Metadata:\n" + prepend_string + "\n" + "Abstract:\n" + abstract
            )
        else:
            input_string = abstract
        label = self.labels[idx]
        evidence_level = self.evidence_levels[idx]

        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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

        outputs = {
            "evidence_item_id": self.evidence_item_id[idx],
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
            "label": torch.tensor(label),
            "evidence_level": evidence_level,
        }

        if self.return_ref_tokens_for_ig:
            input_ref_ids = torch.clone(outputs["input_ids"])
            keep_values_mask = (input_ref_ids != self.tokenizer.cls_token_id) & (
                input_ref_ids != self.tokenizer.sep_token_id
            )
            input_ref_ids[keep_values_mask] = self.tokenizer.pad_token_id
            outputs["input_ref_ids"] = input_ref_ids
            outputs["input_text"] = input_string

        return outputs

    def __eq__(self, other):
        if self.__dict__.keys() != other.__dict__.keys():
            return False
        return all(
            [
                v1.equals(v2) if isinstance(v1, pd.Series) else v1 == v2
                for v1, v2 in zip(self.__dict__.values(), other.__dict__.values())
            ]
        )

    @property
    def class_probabilities(self):
        value_counts = self.labels.value_counts(normalize=True)
        return torch.tensor([value_counts.loc[i] for i in range(5)])

    @property
    def inverse_class_prob_weights(self):
        probs = self.class_probabilities
        weights = 1 / probs
        inf_mask = torch.isinf(weights)
        weights = torch.where(inf_mask, torch.tensor(0.0), weights)
        return weights / weights.sum()
