from civic.monitoring.ITrainingMonitor import ITrainingMonitor
import wandb
import torch
import os

from civic.utils.AcceleratorSingleton import AcceleratorSingleton
from civic.utils.filesystem_utils import create_folder_if_not_exists


class WandbTrainingMonitor(ITrainingMonitor):
    accelerator = AcceleratorSingleton()

    def __init__(self, config):
        self.accelerator.accelerator.init_trackers(
            project_name="civic",
            config=config,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.accelerator.end_training()

    def save_model_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        training_loss,
    ):
        create_folder_if_not_exists("model_checkpoints")
        self.accelerator.wait_for_everyone()
        run_id = self.accelerator.accelerator.get_tracker("wandb").run.id
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": self.accelerator.unwrap_model(
                    optimizer
                ).state_dict(),
                "loss": training_loss,
            },
            f"model_checkpoints/{run_id}",
        )

    def log_training_metrics(self, epoch, total_epochs, metrics):
        self.accelerator.log({"Epoch": epoch, **metrics})
        log_string = f"\nEpoch {epoch + 1}/{total_epochs}: "
        for metric, value in metrics.items():
            log_string += f"{metric}: {value:.4f} | "
        self.accelerator.print(log_string)
