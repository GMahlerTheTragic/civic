import numpy as np
from torch.utils.data import DataLoader
import torch

from civic.metrics.MetricsAggregator import MetricsAggregator
from civic.monitoring import TrainingMonitor
from civic.training import BatchTrainingStep, BatchValidationStep
from civic.training.ModelTrainer import ModelTrainer


CLASS_PROBABILITIES = torch.tensor(
    [121 / 3991, 1327 / 3991, 1368 / 3991, 1145 / 3991, 30 / 3991]
)


class ClassifierFineTuningModelTrainer(ModelTrainer):
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

                metrics_aggregator = MetricsAggregator()
                self.accelerator.wait_for_everyone()
                with torch.no_grad():
                    self.accelerator.print("\r Validation Loop")
                    idx = 0
                    for batch in self.validation_data_loader:
                        validation_metrics = self.batch_validation_step.validate_batch(
                            batch, self.model
                        )
                        validation_metrics = self.accelerator.gather_for_metrics(
                            [validation_metrics]
                        )
                        for vm in validation_metrics:
                            metrics_aggregator.accumulate(vm)
                        idx += 1
                        self.accelerator.print(
                            f"\rProcessed {idx + 1}/{len(self.validation_data_loader)} batches",
                            end="",
                            flush=True,
                        )
                train_loss = sum(self.accelerator.gather_for_metrics([train_loss]))
                accuracy = metrics_aggregator.accuracy()
                f1_scores = metrics_aggregator.f1_scores()
                average_validation_loss = metrics_aggregator.average_loss()
                micro_f1_score = (f1_scores * CLASS_PROBABILITIES).sum().item()
                tm.log_training_metrics(
                    e,
                    n_epochs,
                    {
                        "train_loss": train_loss
                        / (
                            self.train_data_loader.batch_sampler.batch_size
                            * len(self.train_data_loader)
                            * self.accelerator.num_processes
                        ),
                        "val_loss": average_validation_loss,
                        "best_val_loss": best_val_loss,
                        "accuracy": accuracy,
                        "micro-f1-score": micro_f1_score,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                )
                if average_validation_loss < best_val_loss:
                    best_val_loss = average_validation_loss
                    tm.save_model_checkpoint(e, self.model, self.optimizer, train_loss)
