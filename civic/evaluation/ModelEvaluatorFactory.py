import torch.cuda
from torch.utils.data import DataLoader

from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.evaluation.ClassifierModelEvaluator import ClassifierModelEvaluator
from civic.evaluation.IntegratedGradientWrapper import (
    IntegratedGradientWrapper,
    IntegratedGradientsModelType,
)
from civic.evaluation.ModelEvaluator import ModelEvaluator
from civic.models.roberta.RobertaForCivicEvidenceClassification import (
    RobertaForCivicEvidenceClassification,
)


class ModelEvaluatorFactory:
    @staticmethod
    def _get_trainer_from_model(
        tokenizer, model, tokenizer_max_length, batch_size, integrated_gradient_wrapper
    ):
        test_dataloader = DataLoader(
            CivicEvidenceDataSet.full_test_dataset(tokenizer, tokenizer_max_length),
            batch_size=batch_size,
            shuffle=False,
        )
        model_evaluator = ClassifierModelEvaluator(
            test_dataloader,
            model,
            "cuda" if torch.cuda.is_available() else "cpu",
            integrated_gradient_wrapper,
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
