from civic.monitoring.ITrainingMonitor import ITrainingMonitor
import wandb
import torch

from civic.utils.filesystem_utils import create_folder_if_not_exists


class WandbTrainingMonitor(ITrainingMonitor):
    def __init__(self, config):
        self.run = wandb.init(project="civic", config=config)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wandb.finish()

    def save_model_checkpoint(self, epoch, model, optimizer, training_loss):
        create_folder_if_not_exists("model_checkpoints")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": training_loss,
            },
            f"model_checkpoints/longformer_finetuned_{self.run.id}",
        )
        artifact = wandb.Artifact("LongformerFineTuned", type="model")
        artifact.add_file(f"model_checkpoints/longformer_finetuned_{self.run.id}")
        self.run.log_artifact(artifact)

    def log_training_metrics(self, epoch, total_epochs, metrics):
        wandb.log({"Epoch": epoch, **metrics})
        log_string = f"\nEpoch {epoch + 1}/{total_epochs}: "
        for metric, value in metrics.items():
            log_string += f"{metric}: {value:.4f} | "
        print(log_string)
