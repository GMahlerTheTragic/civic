from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.evaluation.ModelEvaluator import ModeEvaluator


class ModelEvaluatorFactory:
    @staticmethod
    def from_model_snapshot(snapshot_name, batch_size):
        model = AutoModel.from_pretrained(snapshot_name)
        tokenizer = AutoTokenizer.from_pretrained(snapshot_name)
        test_dataset = CivicEvidenceDataSet.full_test_dataset(tokenizer, 512)
        test_data_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        model_evaluator = ModeEvaluator(test_data_loader, model, "cuda")
        model_evaluator.do_evaluation()
