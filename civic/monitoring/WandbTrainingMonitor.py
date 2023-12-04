import os

from civic.config import MODEL_CHECKPOINT_DIR
from civic.monitoring.TrainingMonitor import ITrainingMonitor
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
            run_name = self.accelerator.accelerator.get_tracker("wandb").run.name
            save_folder = os.path.join(MODEL_CHECKPOINT_DIR, run_name)
            create_folder_if_not_exists(save_folder)
            self.accelerator.accelerator.unwrap_model(model).save_pretrained(
                save_folder
            )

    def log_training_metrics(self, epoch, total_epochs, metrics):
        self.accelerator.log({"Epoch": epoch, **metrics})
        log_string = f"\nEpoch {epoch}/{total_epochs}: "
        for metric, value in metrics.items():
            log_string += f"{metric}: {value:.10f} | "
        self.accelerator.print(log_string)
