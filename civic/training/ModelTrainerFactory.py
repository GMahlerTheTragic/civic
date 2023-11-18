from torch.utils.data import DataLoader

from civic import CivicEvidenceDataSet, Longformer
from civic.models.BioMedLM import BioMedLM
from civic.models.bert_long.BertLongForMaskedLM import BertLongForMaskedLM

from civic.monitoring import ITrainingMonitor
from civic.monitoring.WandbTrainingMonitor import WandbTrainingMonitor
from civic.training import IModelTrainer, IBatchTrainingStep, IBatchValidationStep
from civic.training.classifier_base_finetuning.ClassifierFineTuningBatchValidationStep import (
    ClassifierFineTuningBatchValidationStep,
)

from civic.training.classifier_base_finetuning.ClassifierFineTuningModelTrainer import (
    ClassifierFineTuningModelTrainer,
)
import torch
from torch.optim import AdamW

from civic.training.classifier_base_finetuning.ClassifierFineTuningBatchTrainingStep import (
    ClassifierFineTuningBatchTrainingStep,
)
from civic.utils.AcceleratorSingleton import AcceleratorSingleton
from civic.utils.GpuDeviceHandler import GpuDeviceHandler

from civic.datasets.CivicEvidenceIterableDataSet import (
    CivicEvidenceIterableDataSet,
)


class ModelTrainerFactory:
    accelerator = AcceleratorSingleton()

    @classmethod
    def create_longformer_base_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> IModelTrainer:
        tokenizer, model = Longformer.from_longformer_allenai_base_pretrained()
        # device = GpuDeviceHandler.use_cuda_if_available()
        # model = GpuDeviceHandler.use_data_parallelism_if_available(model)
        device = cls.accelerator.device
        model.to(device)
        batch_training_step: IBatchTrainingStep = ClassifierFineTuningBatchTrainingStep(
            device, cls.accelerator
        )
        batch_validation_step: IBatchValidationStep = (
            ClassifierFineTuningBatchValidationStep(device)
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
        (model, optimizer, train_data_loader) = cls.accelerator.prepare(
            [model, optimizer, train_data_loader]
        )

        return ClassifierFineTuningModelTrainer(
            training_monitor,
            batch_training_step,
            batch_validation_step,
            train_data_loader,
            validation_data_loader,
            model,
            optimizer,
        )

    @classmethod
    def create_biomedbert_long_pre_training_model_trainer(
        learning_rate, batch_size
    ) -> IModelTrainer:
        tokenizer, model = BertLongForMaskedLM.from_biobert_snapshot()
        device = GpuDeviceHandler.use_cuda_if_available()
        model = GpuDeviceHandler.use_data_parallelism_if_available(model)
        model.to(device)
        batch_training_step: IBatchTrainingStep = ClassifierFineTuningBatchTrainingStep(
            device
        )
        batch_validation_step: IBatchValidationStep = (
            ClassifierFineTuningBatchValidationStep(device)
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
        (model, optimizer, train_data_loader) = accelerator.prepare(
            [model, optimizer, train_data_loader]
        )
        return ClassifierFineTuningModelTrainer(
            training_monitor,
            batch_training_step,
            batch_validation_step,
            train_data_loader,
            validation_data_loader,
            model,
            optimizer,
        )

    @classmethod
    def create_bio_med_lm_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> IModelTrainer:
        tokenizer, model = BioMedLM.from_last_stanford_snapshot()
        device = cls.accelerator.accelerator.device
        model.to(device)
        batch_training_step: IBatchTrainingStep = ClassifierFineTuningBatchTrainingStep(
            device, cls.accelerator
        )
        batch_validation_step: IBatchValidationStep = (
            ClassifierFineTuningBatchValidationStep(device)
        )

        train_dataset = CivicEvidenceDataSet.full_train_dataset(tokenizer, 1024)
        test_dataset = CivicEvidenceDataSet.full_test_dataset(tokenizer, 1024)

        train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
        validation_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        training_monitor: ITrainingMonitor = WandbTrainingMonitor(
            config={
                "learning_rate": learning_rate,
                "architecture": "BioMedLM",
                "dataset": "CivicEvidenceDataSetFull",
                "batch_size": batch_size,
                "distributed": False,
            }
        )

        optimizer = AdamW(model.parameters(), lr=learning_rate)
        (
            model,
            optimizer,
        ) = cls.accelerator.prepare(model, optimizer)
        train_data_loader = cls.accelerator.prepare(train_data_loader)
        validation_data_loader = cls.accelerator.prepare(validation_data_loader)
        return ClassifierFineTuningModelTrainer(
            training_monitor,
            batch_training_step,
            batch_validation_step,
            train_data_loader,
            validation_data_loader,
            model,
            optimizer,
        )
