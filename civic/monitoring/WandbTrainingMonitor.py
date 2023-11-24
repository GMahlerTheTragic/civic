import os

from civic.config import MODEL_CHECKPOINT_DIR
from civic.monitoring.TrainingMonitor import ITrainingMonitor
import torch

from civic.utils.AcceleratorSingleton import AcceleratorSingleton
from civic.utils.filesystem_utils import create_folder_if_not_exists


class WandbTrainingMonitor(ITrainingMonitor):
    def __init__(self, accelerator, config):
        self.accelerator = accelerator
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
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process():
            create_folder_if_not_exists(MODEL_CHECKPOINT_DIR)
            run_name = self.accelerator.accelerator.get_tracker("wandb").run.name
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.accelerator.accelerator.unwrap_model(
                        model
                    ).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": training_loss,
                },
                os.path.join(MODEL_CHECKPOINT_DIR, run_name),
            )

    def log_training_metrics(self, epoch, total_epochs, metrics):
        self.accelerator.log({"Epoch": epoch, **metrics})
        log_string = f"\nEpoch {epoch + 1}/{total_epochs}: "
        for metric, value in metrics.items():
            log_string += f"{metric}: {value:.10f} | "
        self.accelerator.print(log_string)
