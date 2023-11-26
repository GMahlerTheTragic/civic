from torch.utils.data import DataLoader

from civic.monitoring import TrainingMonitor
from civic.training.BatchTrainingStep import BatchTrainingStep
from civic.training.BatchValidationStep import BatchValidationStep

import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy


class ModeEvaluator:
    def __init__(
        self,
        test_data_loader: DataLoader,
        model,
        device,
    ):
        self.test_data_loader = test_data_loader
        self.model = model
        self.device = device

    def do_evaluation(self):
        f1_score = MulticlassF1Score(num_classes=5, average=None)
        macro_f1_score = MulticlassF1Score(num_classes=5, average=None)
        micro_f1_score = MulticlassF1Score(num_classes=5, average=None)
        accuracy = Accuracy(task="multiclass")
        predicted_labels = []
        actual_labels = []
        for idx, batch in self.test_data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels.append(torch.argmax(logits, dim=1))
        return {
            "f1-scores": f1_score(predicted_labels, actual_labels),
            "micro-f1-scores": micro_f1_score(predicted_labels, actual_labels),
            "macro-f1-scores": macro_f1_score(predicted_labels, actual_labels),
            "accuracy": accuracy(predicted_labels, actual_labels),
        }
