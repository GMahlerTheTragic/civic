from civic.training.IBatchValidationStep import IBatchValidationStep
import torch


class LonformerBaseFineTuningBatchValidationStep(IBatchValidationStep):
    def __init__(self, device):
        self.device = device

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
        return torch.ne(labels[negatives_predicted_mask], label).sum().item()

    @staticmethod
    def _compute_correct(predicted_labels, labels):
        return torch.eq(predicted_labels, labels).sum().item()

    def validate_batch(self, batch, model):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)

        return {
            "val_loss": loss.item(),
            "num_samples": labels.size(0),
            "num_correct": self._compute_correct(predicted_labels, labels),
            "true_positives": torch.tensor(
                [
                    self._compute_true_positives(predicted_labels, labels, k)
                    for k in range(5)
                ]
            ),
            "false_positives": torch.tensor(
                [
                    self._compute_false_positives(predicted_labels, labels, k)
                    for k in range(5)
                ]
            ),
            "false_negatives": torch.tensor(
                [
                    self._compute_false_negatives(predicted_labels, labels, k)
                    for k in range(5)
                ]
            ),
        }
