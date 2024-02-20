import torch.cuda
from torch.utils.data import DataLoader

from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.datasets.CivicEvidenceMultiClassDataSet import CivicEvidenceMultiClassDataSet
from civic.evaluation.ClassifierModelEvaluator import ClassifierModelEvaluator
from civic.evaluation.IntegratedGradientWrapper import (
    IntegratedGradientWrapper,
    IntegratedGradientsModelType,
)
from civic.evaluation.ModelEvaluator import ModelEvaluator
from civic.evaluation.MultiLabelClassifierModelEvaluator import (
    MultiLabelClassifierModelEvaluator,
)
from civic.models.bert.BertForCivicEvidenceClassification import (
    BertForCivicEvidenceClassification,
)
from civic.models.roberta.RobertaForCivicEvidenceClassification import (
    RobertaForCivicEvidenceClassification,
)
from civic.training.ModelTrainerFactory import CivicModelTrainingMode


class ModelEvaluatorFactory:

    @staticmethod
    def _get_problem_type_from_mode(mode: CivicModelTrainingMode):
        problem_type = (
            "multi_label_classification"
            if mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
            else "single_label_classification"
        )
        return problem_type

    @staticmethod
    def _get_dataloaders_from_mode(mode, tokenizer, tokenizer_max_length, batch_size):
        if mode == CivicModelTrainingMode.ABSTRACTS_PLUS_PREPEND_METADATA:
            test_dataloader = CivicEvidenceDataSet.full_test_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
            test_dataloader_long = CivicEvidenceDataSet.full_test_dataset_long_only(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
            test_dataloader_gpt4 = CivicEvidenceDataSet.full_test_dataset_gpt4(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
            val_dataloader = CivicEvidenceDataSet.full_validation_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
        elif mode == CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY:
            test_dataloader = CivicEvidenceDataSet.test_dataset_unique_abstracts(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
            test_dataloader_long = (
                CivicEvidenceDataSet.test_dataset_long_only_unique_abstracts(
                    tokenizer, tokenizer_max_length, use_prepend_string=True
                )
            )
            test_dataloader_gpt4 = (
                CivicEvidenceDataSet.test_dataset_gpt4_unique_abstracts(
                    tokenizer, tokenizer_max_length, use_prepend_string=True
                )
            )
            val_dataloader = CivicEvidenceDataSet.validation_dataset_unique_abstracts(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
        elif mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL:
            test_dataloader = CivicEvidenceMultiClassDataSet.full_test_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
            test_dataloader_long = (
                CivicEvidenceMultiClassDataSet.test_dataset_long_only(
                    tokenizer, tokenizer_max_length, use_prepend_string=False
                )
            )
            test_dataloader_gpt4 = (
                CivicEvidenceMultiClassDataSet.full_test_dataset_gpt4(
                    tokenizer, tokenizer_max_length, use_prepend_string=False
                )
            )
            val_dataloader = CivicEvidenceMultiClassDataSet.full_val_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
        else:
            raise RuntimeError("Enum option is not covered")
        return (
            DataLoader(
                test_dataloader,
                batch_size=batch_size,
                shuffle=False,
            ),
            DataLoader(
                test_dataloader_long,
                batch_size=batch_size,
                shuffle=False,
            ),
            DataLoader(
                test_dataloader_gpt4,
                batch_size=batch_size,
                shuffle=False,
            ),
            DataLoader(
                val_dataloader,
                batch_size=batch_size,
                shuffle=False,
            ),
        )

    def _get_evaluator_from_model(
        self,
        tokenizer,
        model,
        tokenizer_max_length,
        batch_size,
        integrated_gradient_wrapper,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY,
    ):

        tdl, tdll, tdlg, vdl = self._get_dataloaders_from_mode(
            mode, tokenizer, tokenizer_max_length, batch_size
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_evaluator = (
            MultiLabelClassifierModelEvaluator(
                tdl,
                model,
                device,
                integrated_gradient_wrapper,
                tdll,
                tdlg,
                val_data_loader=vdl,
            )
            if mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
            else ClassifierModelEvaluator(
                tdl,
                model,
                device,
                integrated_gradient_wrapper,
                tdll,
                tdlg,
            )
        )
        return model_evaluator

    def create_biomed_roberta_model_evaluator(
        self,
        snapshot,
        batch_size,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelEvaluator:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_biomed_roberta_base(
            snapshot, problem_type=problem_type
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Roberta
        )
        return self._get_evaluator_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper, mode=mode
        )

    def create_roberta_model_evaluator(
        self,
        snapshot,
        batch_size,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelEvaluator:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_roberta_base(
            snapshot, problem_type=problem_type
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Roberta
        )
        return self._get_evaluator_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper, mode=mode
        )

    def create_biomed_roberta_long_model_evaluator(
        self,
        snapshot,
        batch_size,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelEvaluator:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_long_biomed_roberta_pretrained(
            snapshot, problem_type=problem_type
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Roberta
        )
        return self._get_evaluator_from_model(
            tokenizer, model, 1024, batch_size, integrated_gradient_wrapper, mode=mode
        )

    def create_bert_model_evaluator(
        self,
        snapshot,
        batch_size,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelEvaluator:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bert_base_uncased(
            snapshot, problem_type=problem_type
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Bert
        )
        return self._get_evaluator_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper, mode=mode
        )

    def create_pubmed_bert_model_evaluator(
        self,
        snapshot,
        batch_size,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelEvaluator:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_pubmed_bert(
            snapshot, problem_type=problem_type
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Bert
        )
        return self._get_evaluator_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper, mode=mode
        )

    def create_bio_link_bert_model_evaluator(
        self,
        snapshot,
        batch_size,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelEvaluator:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert(
            snapshot, problem_type=problem_type
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Bert
        )
        return self._get_evaluator_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper, mode=mode
        )
