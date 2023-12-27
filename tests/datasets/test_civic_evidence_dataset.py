import os

import pytest
from pandas import DataFrame
from transformers import BertTokenizerFast
from unittest.mock import Mock, patch

import civic
from civic.config import DATA_PROCESSED_DIR
from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.utils.filesystem_utils import check_file_exists


class TestCivicEvidenceDataset:
    @pytest.fixture
    def mock_ce_df(self):
        return DataFrame(
            {
                "evidenceLevel": [
                    "A",
                    "B",
                    "C",
                ],
                "sourceAbstract": ["source1", "source2", "source3"],
                "prependString": ["pp1", "pp2", "pp3"],
            }
        )

    @pytest.fixture
    def check_file_exists_mock_false(self, mocker):
        return mocker.patch(
            "civic.datasets.CivicEvidenceDataSet.check_file_exists", return_value=False
        )

    @pytest.fixture
    def check_file_exists_mock_true(self, mocker):
        return mocker.patch(
            "civic.datasets.CivicEvidenceDataSet.check_file_exists", return_value=True
        )

    @pytest.fixture
    def check_read_csv_mock(self, mocker, mock_ce_df):
        return mocker.patch(
            "civic.datasets.CivicEvidenceDataSet.pd.read_csv", return_value=mock_ce_df
        )

    @pytest.fixture
    def mock_tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    def test_full_train_dataset_error(
        self, check_file_exists_mock_false, mock_tokenizer
    ):
        with pytest.raises(FileNotFoundError):
            CivicEvidenceDataSet.full_train_dataset(mock_tokenizer, 512)

        check_file_exists_mock_false.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv")
        )

    def test_full_train_dataset_success(
        self,
        mock_ce_df,
        check_file_exists_mock_true,
        check_read_csv_mock,
        mock_tokenizer,
    ):
        ced = CivicEvidenceDataSet.full_train_dataset(mock_tokenizer, 512)
        expected = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        assert ced == expected
        check_file_exists_mock_true.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_train.csv")
        )

    def test_full_validation_dataset_error(
        self, check_file_exists_mock_false, mock_tokenizer
    ):
        with pytest.raises(FileNotFoundError):
            CivicEvidenceDataSet.full_validation_dataset(mock_tokenizer, 512)

        check_file_exists_mock_false.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv")
        )

    def test_full_validation_dataset_success(
        self,
        mock_ce_df,
        check_file_exists_mock_true,
        check_read_csv_mock,
        mock_tokenizer,
    ):
        ced = CivicEvidenceDataSet.full_validation_dataset(mock_tokenizer, 512)
        expected = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        assert ced == expected
        check_file_exists_mock_true.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_val.csv")
        )

    def test_full_test_dataset_error(
        self, check_file_exists_mock_false, mock_tokenizer
    ):
        with pytest.raises(FileNotFoundError):
            CivicEvidenceDataSet.full_test_dataset(mock_tokenizer, 512)

        check_file_exists_mock_false.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        )

    def test_full_test_dataset_success(
        self,
        mock_ce_df,
        check_file_exists_mock_true,
        check_read_csv_mock,
        mock_tokenizer,
    ):
        ced = CivicEvidenceDataSet.full_test_dataset(mock_tokenizer, 512)
        expected = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        assert ced == expected
        check_file_exists_mock_true.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        )

    def test_full_gpt4_dataset_error(
        self, check_file_exists_mock_false, mock_tokenizer
    ):
        with pytest.raises(FileNotFoundError):
            CivicEvidenceDataSet.full_test_dataset_gpt4(mock_tokenizer, 512)

        check_file_exists_mock_false.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4.csv")
        )

    def test_full_gpt4_dataset_success(
        self,
        mock_ce_df,
        check_file_exists_mock_true,
        check_read_csv_mock,
        mock_tokenizer,
    ):
        ced = CivicEvidenceDataSet.full_test_dataset_gpt4(mock_tokenizer, 512)
        expected = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512, True)
        assert ced == expected
        check_file_exists_mock_true.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test_gpt4.csv")
        )
