from dataclasses import dataclass

from civic.metrics import MetricsContainer


@dataclass
class MetricsAggregator:
    total_number_of_samples: int = 0
    total_correct: int = 0
    total_true_positives: int = 0
    total_false_positives: int = 0
    total_false_negatives: int = 0
    total_validation_loss: float = 0.0

    def accumulate(self, metrics_container: MetricsContainer):
        self.total_number_of_samples += metrics_container.num_samples
        self.total_correct += metrics_container.num_correct
        self.total_true_positives += metrics_container.true_positives
        self.total_false_positives += metrics_container.false_positives
        self.total_false_negatives += metrics_container.false_negatives
        self.total_validation_loss += metrics_container.val_loss

    def accuracy(self):
        return self.total_correct / self.total_number_of_samples

    def f1_scores(self):
        return (2 * self.total_true_positives) / (
            2 * self.total_true_positives
            + self.total_false_positives
            + self.total_false_negatives
        )

    def average_loss(self):
        return self.total_validation_loss / self.total_number_of_samples
