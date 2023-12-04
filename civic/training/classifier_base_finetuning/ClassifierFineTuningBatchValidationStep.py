from civic.metrics.MetricsContainer import MetricsContainer
from civic.training.BatchValidationStep import BatchValidationStep
import torch


class ClassifierFineTuningBatchValidationStep(BatchValidationStep):
    def __init__(self, device, criterion=None):
        self.device = device
        self.criterion = criterion.to(self.device)

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
        batch_size = batch["input_ids"].size(0)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        if self.criterion:
            loss = self.criterion(logits, labels)
        else:
            loss = outputs.loss

        predicted_labels = torch.argmax(logits, dim=1)

        return MetricsContainer(
            num_samples=labels.size(0),
            num_correct=self._compute_correct(labels, predicted_labels),
            true_positives=torch.tensor(
                [
                    self._compute_true_positives(predicted_labels, labels, k)
                    for k in range(5)
                ]
            ),
            false_positives=torch.tensor(
                [
                    self._compute_false_positives(predicted_labels, labels, k)
                    for k in range(5)
                ]
            ),
            false_negatives=torch.tensor(
                [
                    self._compute_false_negatives(predicted_labels, labels, k)
                    for k in range(5)
                ]
            ),
            val_loss=loss.item() * batch_size,
        )
