import torch
from torch.utils.data import DataLoader

from civic import CivicEvidenceDataSet, Longformer
from civic.monitoring import ITrainingMonitor
from civic.monitoring.WandbTrainingMonitor import WandbTrainingMonitor
from civic.training import IModelTrainer, IBatchTrainingStep, IBatchValidationStep
from civic.training.longformer_base_finetuning.LonformerBaseFineTuningBatchValidationStep import (
    LonformerBaseFineTuningBatchValidationStep,
)

from civic.training.longformer_base_finetuning.LonformerFineTuningModelTrainer import (
    LongFormerFineTuningModelTrainer,
)

from torch.optim import AdamW

from civic.training.longformer_base_finetuning.LongformerBaseFineTuningBatchTrainingStep import (
    LongformerBaseFineTuningBatchTrainingStep,
)


class ModelTrainerFactory:
    @staticmethod
    def create_longformer_base_finetuning_model_trainer(
        learning_rate, batch_size
    ) -> IModelTrainer:
        tokenizer, model = Longformer.from_longformer_allenai_base_pretrained()
        device = torch.device("cpu")
        print(f"Using device: {device}")
        model.to(device)
        batch_training_step: IBatchTrainingStep = (
            LongformerBaseFineTuningBatchTrainingStep(device)
        )
        batch_validation_step: IBatchValidationStep = (
            LonformerBaseFineTuningBatchValidationStep(device)
        )
        train_dataset = CivicEvidenceDataSet.full_train_dataset(tokenizer, 4096)
        test_dataset = CivicEvidenceDataSet.full_test_dataset(tokenizer, 4096)

        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        training_monitor: ITrainingMonitor = WandbTrainingMonitor(
            config={
                "learning_rate": learning_rate,
                "architecture": "Longformer",
                "dataset": "CivicEvidenceDataSetFull",
                "batch_size": batch_size,
                "distributed": False,
            }
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        return LongFormerFineTuningModelTrainer(
            training_monitor,
            batch_training_step,
            batch_validation_step,
            train_data_loader,
            validation_data_loader,
            model,
            optimizer,
        )

    @staticmethod
    def create_long_former_pre_training_model_trainer() -> IModelTrainer:
        raise NotImplementedError

    @staticmethod
    def create_bio_med_lm_finetuning_model_trainer() -> IModelTrainer:
        raise NotImplementedError
