import numpy as np
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score
from civic.monitoring import TrainingMonitor
from civic.training import BatchTrainingStep, BatchValidationStep
from civic.training.ModelTrainer import ModelTrainer


CLASS_PROBABILITIES = torch.tensor(
    [121 / 3991, 1327 / 3991, 1368 / 3991, 1145 / 3991, 30 / 3991]
)


class MultiLabelClassifierFineTuningModelTrainer(ModelTrainer):
    def __init__(
        self,
        training_monitor: TrainingMonitor,
        batch_training_step: BatchTrainingStep,
        batch_validation_step: BatchValidationStep,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        model,
        optimizer,
        accelerator,
    ):
        self.training_monitor = training_monitor
        self.batch_training_step = batch_training_step
        self.batch_validation_step = batch_validation_step
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.model = model
        self.optimizer = optimizer
        self.accelerator = accelerator

    def do_model_training(self, n_epochs):
        with self.training_monitor as tm:
            best_val_loss = np.Inf
            for e in range(1, n_epochs + 1):
                self.accelerator.wait_for_everyone()
                self.model.train()
                train_loss = 0
                for idx, batch in enumerate(self.train_data_loader):
                    train_loss += self.batch_training_step.train_batch(
                        batch, self.model, self.optimizer, None
                    )
                    self.accelerator.print(
                        f"\rProcessed {idx + 1}/{len(self.train_data_loader)} batches",
                        end="",
                        flush=True,
                    )
                self.accelerator.wait_for_everyone()
                with torch.no_grad():
                    self.accelerator.print("\r Validation Loop")
                    idx = 0
                    logits_list = []
                    predictions_list = []
                    labels_list = []
                    for batch in self.validation_data_loader:
                        logits, predictions = self.batch_validation_step.validate_batch(
                            batch, self.model
                        )
                        logits, predictions, labels = (
                            self.accelerator.gather_for_metrics(
                                [logits, predictions, batch["label"]]
                            )
                        )
                        labels_list.append(labels)
                        logits_list.append(logits)
                        predictions_list.append(predictions)
                        idx += 1
                        self.accelerator.print(
                            f"\rProcessed {idx + 1}/{len(self.validation_data_loader)} batches",
                            end="",
                            flush=True,
                        )
                predictions = torch.concat(predictions_list)
                labels = torch.concat(labels_list)
                probabilities = torch.sigmoid(torch.concat(logits_list))
                average_validation_loss = binary_cross_entropy(
                    probabilities, labels
                ).detach()
                tm.log_training_metrics(
                    e,
                    n_epochs,
                    {
                        "val_loss": average_validation_loss,
                        "best_val_loss": best_val_loss,
                        "micro-f1-score": f1_score(
                            predictions.cpu(),
                            labels.cpu(),
                            average="micro",
                        ),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                )
                if average_validation_loss < best_val_loss:
                    best_val_loss = average_validation_loss
                    tm.save_model_checkpoint(e, self.model, self.optimizer, train_loss)
