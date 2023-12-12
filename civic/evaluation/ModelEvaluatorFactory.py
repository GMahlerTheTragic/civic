import torch.cuda
from torch.utils.data import DataLoader

from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.evaluation.ClassifierModelEvaluator import ClassifierModelEvaluator
from civic.evaluation.IntegratedGradientWrapper import (
    IntegratedGradientWrapper,
    IntegratedGradientsModelType,
)
from civic.evaluation.ModelEvaluator import ModelEvaluator
from civic.models.bert.BertForCivicEvidenceClassification import (
    BertForCivicEvidenceClassification,
)
from civic.models.roberta.RobertaForCivicEvidenceClassification import (
    RobertaForCivicEvidenceClassification,
)


class ModelEvaluatorFactory:
    @staticmethod
    def _get_trainer_from_model(
        tokenizer, model, tokenizer_max_length, batch_size, integrated_gradient_wrapper
    ):
        test_dataloader = DataLoader(
            CivicEvidenceDataSet.full_validation_dataset(
                tokenizer, tokenizer_max_length
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_dataloader_long = DataLoader(
            CivicEvidenceDataSet.full_test_dataset_long_only(
                tokenizer, tokenizer_max_length
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_dataloader_gpt4 = DataLoader(
            CivicEvidenceDataSet.full_test_dataset_gpt4(
                tokenizer, tokenizer_max_length
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        model_evaluator = ClassifierModelEvaluator(
            test_dataloader,
            model,
            "cuda" if torch.cuda.is_available() else "cpu",
            integrated_gradient_wrapper,
            test_dataloader_long,
            test_dataloader_gpt4,
        )
        return model_evaluator

    def create_biomed_roberta_model_evaluator(
        self, snapshot, batch_size
    ) -> ModelEvaluator:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_biomed_roberta_base(snapshot)
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Roberta
        )
        return self._get_trainer_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper
        )

    def create_roberta_model_evaluator(self, snapshot, batch_size) -> ModelEvaluator:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_roberta_base(snapshot)
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Roberta
        )
        return self._get_trainer_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper
        )

    def create_biomed_roberta_long_model_evaluator(
        self, snapshot, batch_size
    ) -> ModelEvaluator:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_long_biomed_roberta_pretrained(
            snapshot
        )
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Roberta
        )
        return self._get_trainer_from_model(
            tokenizer, model, 1024, batch_size, integrated_gradient_wrapper
        )

    def create_bert_model_evaluator(self, snapshot, batch_size) -> ModelEvaluator:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bert_base_uncased(snapshot)
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Bert
        )
        return self._get_trainer_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper
        )

    def create_pubmed_bert_model_evaluator(
        self, snapshot, batch_size
    ) -> ModelEvaluator:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_pubmed_bert(snapshot)
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Bert
        )
        return self._get_trainer_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper
        )

    def create_bio_link_bert_model_evaluator(
        self, snapshot, batch_size
    ) -> ModelEvaluator:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert(snapshot)
        integrated_gradient_wrapper = IntegratedGradientWrapper(
            model, IntegratedGradientsModelType.Bert
        )
        return self._get_trainer_from_model(
            tokenizer, model, 512, batch_size, integrated_gradient_wrapper
        )
