import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch

from civic.metrics.MetricsAggregator import MetricsAggregator
from civic.monitoring import ITrainingMonitor
from civic.training import IBatchTrainingStep, IBatchValidationStep
from civic.training.IModelTrainer import IModelTrainer
from civic.utils.AcceleratorSingleton import AcceleratorSingleton

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


class ClassifierFineTuningModelTrainer(IModelTrainer):
    accelerator = AcceleratorSingleton()

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
                self.accelerator.wait_for_everyone()
                self.model.train()
                train_loss = 0
                for idx, batch in enumerate(self.train_data_loader):
                    train_loss += self.batch_training_step.train_batch(
                        batch, self.model, self.optimizer, None
                    )
                    self.accelerator.print(
                        f"\rProcessed {idx}/{len(self.train_data_loader)} batches",
                        end="",
                        flush=True,
                    )
                scheduler.step()
                metrics_aggregator = MetricsAggregator()
                self.accelerator.wait_for_everyone()
                with torch.no_grad():
                    self.accelerator.print("Validation Loop \n")
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
                            f"\rProcessed {idx}/{len(self.validation_data_loader)} batches",
                            end="",
                            flush=True,
                        )

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
                                len(self.train_data_loader)
                                * self.train_data_loader.batch_sampler.batch_size
                                * self.accelerator.num_processes
                            ),
                            "val_loss": f1_scores,
                            "accuracy": accuracy,
                            "micro-f1-score": micro_f1_score,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                    )
                    if average_validation_loss < best_val_loss:
                        best_val_loss = average_validation_loss
                        tm.save_model_checkpoint(
                            e, self.model, self.optimizer, train_loss
                        )
