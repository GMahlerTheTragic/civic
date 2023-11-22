from torch.utils.data import DataLoader

from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.models.bert.BertForCivicEvidenceClassification import (
    BertForCivicEvidenceClassification,
)
from civic.models.gpt_2.GPT2ForCivicEvidenceClassification import (
    GPT2ForCivicEvidenceClassification,
)
from civic.models.roberta.RobertaForCivicEvidenceClassification import (
    RobertaForCivicEvidenceClassification,
)
from civic.monitoring import TrainingMonitor
from civic.monitoring.WandbTrainingMonitor import WandbTrainingMonitor
from civic.training import ModelTrainer, BatchTrainingStep, BatchValidationStep
from civic.training.classifier_base_finetuning.ClassifierFineTuningBatchValidationStep import (
    ClassifierFineTuningBatchValidationStep,
)

from civic.training.classifier_base_finetuning.ClassifierFineTuningModelTrainer import (
    ClassifierFineTuningModelTrainer,
)
from torch.optim import AdamW

from civic.training.classifier_base_finetuning.ClassifierFineTuningBatchTrainingStep import (
    ClassifierFineTuningBatchTrainingStep,
)
from civic.utils.AcceleratorSingleton import AcceleratorSingleton


class ModelTrainerFactory:
    accelerator = AcceleratorSingleton()

    @staticmethod
    def _get_dataloaders_for_civic_evidence_finetuning(tokenizer, batch_size):
        train_dataset = CivicEvidenceDataSet.full_train_dataset(tokenizer, 1024)
        test_dataset = CivicEvidenceDataSet.full_test_dataset(tokenizer, 1024)

        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_data_loader, validation_data_loader

    @classmethod
    def _get_classification_steps(cls):
        batch_training_step: BatchTrainingStep = ClassifierFineTuningBatchTrainingStep(
            cls.accelerator.accelerator.device, cls.accelerator
        )
        batch_validation_step: BatchValidationStep = (
            ClassifierFineTuningBatchValidationStep(cls.accelerator.accelerator.device)
        )
        return batch_training_step, batch_validation_step

    @classmethod
    def _get_trainer_from_model(
        cls,
        model,
        tokenizer,
        batch_size,
        learning_rate,
        architecture,
        snapshot_name,
        gradient_accumulation_steps=1,
    ):
        model.to(cls.accelerator.accelerator.device)
        batch_training_step, batch_validation_step = cls._get_classification_steps()
        (
            train_data_loader,
            validation_data_loader,
        ) = ModelTrainerFactory._get_dataloaders_for_civic_evidence_finetuning(
            tokenizer, batch_size
        )
        training_monitor: TrainingMonitor = WandbTrainingMonitor(
            config={
                "learning_rate": learning_rate,
                "architecture": architecture,
                "snapshot": snapshot_name,
                "dataset": "CivicEvidenceDataSetFull",
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_processes": cls.accelerator.accelerator.num_processes,
                "effective_batch_size": batch_size
                * gradient_accumulation_steps
                * cls.accelerator.accelerator.num_processes,
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

    @classmethod
    def create_longformer_base_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_longformer_base()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="longformer",
            snapshot_name="allenai-longformer-base",
        )

    @classmethod
    def create_roberta_base_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_roberta_base()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta",
            snapshot_name="roberta-base",
        )

    @classmethod
    def create_biomed_roberta_base_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_biomed_roberta_base()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta",
            snapshot_name="biomed_roberta_base",
        )

    @classmethod
    def create_bert_base_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bert_base_uncased()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bert-base-uncased",
        )

    @classmethod
    def create_pubmed_bert_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_pubmed_bert()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="pubmed_bert",
        )

    @classmethod
    def create_bio_link_bert_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bio_link_bert",
        )

    @classmethod
    def create_biomed_lm_finetuning_model_trainer(
        cls, learning_rate, batch_size
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = GPT2ForCivicEvidenceClassification.from_biomed_lm_snapshot()
        return cls._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="gpt2",
            snapshot_name="biomed_lm",
        )

    @classmethod
    def create_biomed_roberta_long_pre_training_model_trainer(cls, training_args):
        """TODO"""
        pass
