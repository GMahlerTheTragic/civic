import os

import pytest
import torch
from pandas import DataFrame
from transformers import BertTokenizerFast

from civic.config import DATA_PROCESSED_DIR
from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet


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
    def mock_ce_df_long(self):
        return DataFrame(
            {
                "evidenceLevel": [
                    "A",
                    "B",
                    "C",
                ],
                "sourceAbstract": ["source " * 100, "source2 " * 500, "source3 " * 500],
                "prependString": ["pp1", "pp2", "pp3"],
            }
        )

    @pytest.fixture
    def mock_ce_df_long_filtered(self):
        return DataFrame(
            {
                "evidenceLevel": [
                    "B",
                    "C",
                ],
                "sourceAbstract": ["source2 " * 500, "source3 " * 500],
                "prependString": ["pp2", "pp3"],
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
    def check_read_csv_mock_long(self, mocker, mock_ce_df_long):
        return mocker.patch(
            "civic.datasets.CivicEvidenceDataSet.pd.read_csv",
            return_value=mock_ce_df_long,
        )

    @pytest.fixture
    def mock_tokenizer(self):
        return BertTokenizerFast.from_pretrained("bert-base-uncased")

    @pytest.fixture(
        scope="module",
        params=[
            (
                DataFrame(
                    {
                        "evidenceLevel": [],
                        "sourceAbstract": [],
                        "prependString": [],
                    }
                ),
                0,
            ),
            (
                DataFrame(
                    {
                        "evidenceLevel": [
                            "A",
                            "B",
                            "C",
                        ],
                        "sourceAbstract": [
                            "source " * 100,
                            "source2 " * 500,
                            "source3 " * 500,
                        ],
                        "prependString": ["pp1", "pp2", "pp3"],
                    }
                ),
                3,
            ),
            (
                DataFrame(
                    {
                        "evidenceLevel": [
                            "B",
                            "C",
                        ],
                        "sourceAbstract": ["source2 " * 500, "source3 " * 500],
                        "prependString": ["pp2", "pp3"],
                    }
                ),
                2,
            ),
        ],
    )
    def __len__test_cases(self, request):
        yield request.param

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

    def test_full_test_dataset_long_only_error(
        self, check_file_exists_mock_false, mock_tokenizer
    ):
        with pytest.raises(FileNotFoundError):
            CivicEvidenceDataSet.full_test_dataset_long_only(mock_tokenizer, 512)

        check_file_exists_mock_false.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        )

    def test_full_test_dataset_long_only_success(
        self,
        mock_ce_df_long,
        mock_ce_df_long_filtered,
        check_file_exists_mock_true,
        check_read_csv_mock_long,
        mock_tokenizer,
    ):
        ced = CivicEvidenceDataSet.full_test_dataset_long_only(mock_tokenizer, 512)
        expected = CivicEvidenceDataSet(mock_ce_df_long_filtered, mock_tokenizer, 512)
        assert ced == expected
        check_file_exists_mock_true.assert_called_once_with(
            os.path.join(DATA_PROCESSED_DIR, "civic_evidence_test.csv")
        )

    def test__len__(
        self,
        __len__test_cases,
        mock_tokenizer,
    ):
        df, length = __len__test_cases
        ced = CivicEvidenceDataSet(df, mock_tokenizer, 512)
        assert ced.__len__() == length

    def test___getitem__(self, mock_ce_df, mock_tokenizer):
        ced = CivicEvidenceDataSet(
            mock_ce_df, mock_tokenizer, 12, return_ref_tokens_for_ig=True
        )
        sample = ced.__getitem__(0)
        expected_output = {
            "input_ids": torch.tensor(
                [101, 27425, 1024, 4903, 2487, 10061, 1024, 3120, 2487, 102, 0, 0]
            ),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
            "label": torch.tensor(0),
            "evidence_level": "A",
            "input_ref_ids": torch.tensor([101, 0, 0, 0, 0, 0, 0, 0, 0, 102, 0, 0]),
            "input_text": "Metadata:\npp1\nAbstract:\nsource1",
        }
        print(expected_output)
        print(sample)

        assert all(
            [
                torch.equal(v, sample[k])
                if isinstance(v, torch.Tensor)
                else v == sample[k]
                for k, v in expected_output.items()
            ]
        )

    def test___eq__1(self, mock_ce_df, mock_tokenizer):
        ced1 = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        ced2 = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        assert ced1 == ced2

    def test___eq__2(self, mock_ce_df, mock_tokenizer):
        ced1 = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        ced3 = CivicEvidenceDataSet(mock_ce_df.iloc[:2], mock_tokenizer, 512)
        assert ced1 != ced3

    def test_class_probabilities(self, mock_ce_df, mock_tokenizer):
        ced = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        expected_probabilities = torch.DoubleTensor([1 / 3] * 3 + [0.0] * 2)
        assert torch.allclose(
            ced.class_probabilities, expected_probabilities, atol=1e-5
        )

    def test_inverse_class_prob_weights(self, mock_ce_df, mock_tokenizer):
        ced = CivicEvidenceDataSet(mock_ce_df, mock_tokenizer, 512)
        expected_weights = torch.DoubleTensor([1 / 3] * 3 + [0.0] * 2)

        assert torch.allclose(
            ced.inverse_class_prob_weights, expected_weights, atol=1e-5
        )
