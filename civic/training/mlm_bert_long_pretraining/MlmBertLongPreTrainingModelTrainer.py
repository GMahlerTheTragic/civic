import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch
from civic.monitoring import ITrainingMonitor
from civic.training import IBatchTrainingStep, IBatchValidationStep
from civic.training.IModelTrainer import IModelTrainer

CLASS_PROBABILITIES = torch.tensor(
    [121 / 3991, 1327 / 3991, 1368 / 3991, 1145 / 3991, 30 / 3991]
)


def lr_lambda(epoch):
    if epoch < 3:
        # Linear ramp-up over 3 epochs from 0.00001 to 0.0001
        return 1 + (10 - 1) * epoch / 3
    elif epoch == 3:
        # Constant learning rate for 1 epoch (0.0001)
        return 10
    elif (epoch > 3) and (epoch < 6):
        # Linear ramp-down over 3 epochs from 0.0001 to 0.0001
        return 10 - (10 - 1) * (epoch - 4) / 3
    else:
        return 1


class MlmBertLongPreTrainingModelTrainer(IModelTrainer):
    def __init__(
        self,
        training_monitor: ITrainingMonitor,
        batch_training_step: IBatchTrainingStep,
        batch_validation_step: IBatchValidationStep,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        model,
        optimizer,
    ):
        self.training_monitor = training_monitor
        self.batch_training_step = batch_training_step
        self.batch_validation_step = batch_validation_step
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.model = model
        self.optimizer = optimizer

    def do_model_training(self, n_epochs):
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        with self.training_monitor as tm:
            best_val_loss = np.Inf
            for e in range(1, n_epochs):
                self.model.train()
                train_loss = 0
                idx = 0
                for batch in self.train_data_loader:
                    train_loss += self.batch_training_step.train_batch(
                        batch, self.model, self.optimizer, None
                    )
                    idx += 1
                    print(
                        f"\rProcessed {idx}/{len(self.train_data_loader)} batches",
                        end="",
                        flush=True,
                    )
                scheduler.step()
                total_number_of_samples = 0
                total_correct = 0
                total_true_positives = 0
                total_false_positives = 0
                total_false_negatives = 0
                total_validation_loss = 0

                with torch.no_grad():
                    for batch in self.validation_data_loader:
                        validation_metrics = self.batch_validation_step.validate_batch(
                            batch, self.model
                        )
                        total_number_of_samples += validation_metrics["num_samples"]
                        total_correct += validation_metrics["num_correct"]
                        total_true_positives += validation_metrics["true_positives"]
                        total_false_positives += validation_metrics["false_positives"]
                        total_false_negatives += validation_metrics["false_negatives"]
                        total_validation_loss += validation_metrics["val_loss"]

                    accuracy = total_correct / total_number_of_samples
                    f1_scores = (2 * total_true_positives) / (
                        2 * total_true_positives
                        + total_false_positives
                        + total_false_negatives
                    )
                    micro_f1_score = (f1_scores * CLASS_PROBABILITIES).sum().item()
                    tm.log_training_metrics(
                        e,
                        n_epochs,
                        {
                            "train_loss": train_loss
                            / (
                                len(self.train_data_loader)
                                * self.train_data_loader.batch_size
                            ),
                            "val_loss": total_validation_loss
                            / (
                                len(self.validation_data_loader)
                                * self.validation_data_loader.batch_size
                            ),
                            "accuracy": accuracy,
                            "micro-f1-score": micro_f1_score,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                    )
                    if total_validation_loss < best_val_loss:
                        best_val_loss = total_validation_loss
                        tm.save_model_checkpoint(
                            e, self.model, self.optimizer, train_loss
                        )
