from civic.metrics.MetricsContainer import MetricsContainer
from civic.training.BatchValidationStep import BatchValidationStep
import torch


class MultiLabelClassifierFineTuningBatchValidationStep(BatchValidationStep):
    def __init__(self, device, criterion=None):
        self.device = device
        self.criterion = criterion.to(self.device) if criterion else None

    @staticmethod
    def _compute_true_positives(predicted_labels, labels, label: int):
        positives_predicted_mask = torch.eq(predicted_labels, label)
        return torch.eq(labels[positives_predicted_mask], label).sum().item()

    @staticmethod
    def _compute_false_positives(predicted_labels, labels, label: int):
        positives_predicted_mask = torch.eq(predicted_labels, label)
        return torch.ne(labels[positives_predicted_mask], label).sum().item()

    @staticmethod
    def _compute_false_negatives(predicted_labels, labels, label: int):
        negatives_predicted_mask = torch.ne(predicted_labels, label)
        return torch.eq(labels[negatives_predicted_mask], label).sum().item()

    @staticmethod
    def _compute_correct(predicted_labels, labels):
        return torch.eq(predicted_labels, labels).sum().item()

    def validate_batch(self, batch, model):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        predicted_labels = torch.where(logits < 0, torch.tensor(0.0), torch.tensor(1.0))

        return logits, predicted_labels
