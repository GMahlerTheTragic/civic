import os.path
from enum import Enum, auto

from datasets import load_from_disk, Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
)

from civic.config import HF_DATA_CACHE_DIR
from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet
from civic.datasets.CivicEvidenceMultiClassDataSet import CivicEvidenceMultiClassDataSet
from civic.models.bert.BertForCivicEvidenceClassification import (
    BertForCivicEvidenceClassification,
)
from civic.models.gpt_2.GPT2ForCivicEvidenceClassification import (
    GPT2ForCivicEvidenceClassification,
)
from civic.models.roberta.RobertaForCivicEvidenceClassification import (
    RobertaForCivicEvidenceClassification,
)
from civic.models.roberta.RobertaLongForMaskedLM import RobertaLongForMaskedLM
from civic.models.roberta.RobertaLongForSequenceClassification import (
    RobertaLongForSequenceClassification,
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
from civic.training.classifier_multi_label_finetuning.MultiLabelClassifierFineTuningBatchTrainingStep import (
    MultiLabelClassifierFineTuningBatchTrainingStep,
)
from civic.training.classifier_multi_label_finetuning.MultiLabelClassifierFineTuningBatchValidationStep import (
    MultiLabelClassifierFineTuningBatchValidationStep,
)
from civic.training.classifier_multi_label_finetuning.MultiLabelClassifierFineTuningModelTrainer import (
    MultiLabelClassifierFineTuningModelTrainer,
)
from civic.training.mlm_bert_long_pretraining.MlmBertLongPreTrainingModelTrainer import (
    MlmRobertaLongPreTrainingModelTrainer,
)


class CivicModelTrainingMode(Enum):
    ABSTRACTS_ONLY_MULTILABEL = auto()
    ABSTRACTS_ONLY_UNIQUE_ONLY = auto()
    ABSTRACTS_PLUS_PREPEND_METADATA = auto()


class ModelTrainerFactory:
    def __init__(self, accelerator):
        self.accelerator = accelerator

    @staticmethod
    def _get_dataloaders_for_civic_evidence_finetuning(
        tokenizer,
        batch_size,
        tokenizer_max_length,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY,
    ):

        if mode == CivicModelTrainingMode.ABSTRACTS_PLUS_PREPEND_METADATA:
            train_dataset = CivicEvidenceDataSet.full_train_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
            test_dataset = CivicEvidenceDataSet.full_validation_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=True
            )
        elif mode == CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY:
            train_dataset = CivicEvidenceDataSet.train_dataset_unique_abstracts(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
            test_dataset = CivicEvidenceDataSet.validation_dataset_unique_abstracts(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
        elif mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL:
            train_dataset = CivicEvidenceMultiClassDataSet.full_train_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
            test_dataset = CivicEvidenceMultiClassDataSet.full_val_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=False
            )
        else:
            raise RuntimeError("Enum option is not covered")

        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_data_loader, validation_data_loader

    def _get_classification_steps(
        self, weights=None, mode=CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY
    ):
        criterion = (
            CrossEntropyLoss(reduction="sum", weight=weights.float())
            if weights is not None
            else None
        )
        batch_training_step: BatchTrainingStep = (
            MultiLabelClassifierFineTuningBatchTrainingStep(
                self.accelerator.device, self.accelerator, criterion
            )
            if mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
            else ClassifierFineTuningBatchTrainingStep(
                self.accelerator.device, self.accelerator, criterion
            )
        )
        batch_validation_step: BatchValidationStep = (
            MultiLabelClassifierFineTuningBatchValidationStep(
                self.accelerator.device, criterion
            )
            if mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
            else ClassifierFineTuningBatchValidationStep(
                self.accelerator.device, criterion
            )
        )
        return batch_training_step, batch_validation_step

    @staticmethod
    def _get_problem_type_from_mode(mode: CivicModelTrainingMode):
        problem_type = (
            "multi_label_classification"
            if mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
            else "single_label_classification"
        )
        return problem_type

    def _get_trainer_from_model(
        self,
        model,
        tokenizer,
        batch_size,
        learning_rate,
        architecture,
        snapshot_name,
        tokenizer_max_length,
        gradient_accumulation_steps=1,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY,
    ):

        model.to(self.accelerator.device)
        (
            train_data_loader,
            validation_data_loader,
        ) = ModelTrainerFactory._get_dataloaders_for_civic_evidence_finetuning(
            tokenizer, batch_size, tokenizer_max_length, mode=mode
        )
        weights = (
            train_data_loader.dataset.inverse_class_prob_weights if weighted else None
        )
        batch_training_step, batch_validation_step = self._get_classification_steps(
            weights=weights, mode=mode
        )
        training_monitor: TrainingMonitor = WandbTrainingMonitor(
            accelerator=self.accelerator,
            config={
                "learning_rate": learning_rate,
                "architecture": architecture,
                "snapshot": snapshot_name,
                "dataset": "CivicEvidenceDataSetFull",
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "num_processes": self.accelerator.num_processes,
                "weighted": "True" if weighted else "False",
                "effective_batch_size": batch_size
                * gradient_accumulation_steps
                * self.accelerator.num_processes,
                "mode": mode,
            },
        )
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        (
            model,
            optimizer,
        ) = self.accelerator.prepare(model, optimizer)
        train_data_loader = self.accelerator.prepare(train_data_loader)
        validation_data_loader = self.accelerator.prepare(validation_data_loader)

        trainer = (
            MultiLabelClassifierFineTuningModelTrainer(
                training_monitor,
                batch_training_step,
                batch_validation_step,
                train_data_loader,
                validation_data_loader,
                model,
                optimizer,
                self.accelerator,
            )
            if mode == CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
            else ClassifierFineTuningModelTrainer(
                training_monitor,
                batch_training_step,
                batch_validation_step,
                train_data_loader,
                validation_data_loader,
                model,
                optimizer,
                self.accelerator,
            )
        )
        return trainer

    def create_roberta_base_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_roberta_base(
            problem_type=problem_type
        )
        if snapshot:
            model = RobertaForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta",
            snapshot_name="roberta-base",
            tokenizer_max_length=512,
            weighted=weighted,
            mode=mode,
        )

    def create_biomed_roberta_base_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_biomed_roberta_base(
            problem_type=problem_type
        )
        if snapshot:
            model = RobertaForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta",
            snapshot_name="biomed_roberta_base",
            tokenizer_max_length=512,
            weighted=weighted,
            mode=mode,
        )

    def create_biomed_roberta_long_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_long_biomed_roberta_pretrained(
            problem_type=problem_type
        )
        if snapshot:
            model = RobertaLongForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta_long",
            snapshot_name="biomed_roberta_long",
            tokenizer_max_length=1024,
            weighted=weighted,
            mode=mode,
        )

    def create_bert_base_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bert_base_uncased(
            problem_type=problem_type
        )
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bert-base-uncased",
            tokenizer_max_length=512,
            weighted=weighted,
            mode=mode,
        )

    def create_pubmed_bert_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_pubmed_bert(
            problem_type=problem_type
        )
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="pubmed_bert",
            tokenizer_max_length=512,
            weighted=weighted,
            mode=mode,
        )

    def create_bio_link_bert_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert(
            problem_type=problem_type
        )
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bio_link_bert",
            tokenizer_max_length=512,
            weighted=weighted,
            mode=mode,
        )

    def create_bio_link_bert_large_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
        mode=CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL,
    ) -> ModelTrainer:
        problem_type = self._get_problem_type_from_mode(mode)
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert_large(
            problem_type=problem_type
        )
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(
                snapshot, problem_type=problem_type
            )
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bio_link_bert_long",
            tokenizer_max_length=512,
            weighted=weighted,
            mode=mode,
        )

    def create_biomed_lm_finetuning_model_trainer(
        self,
        learning_rate,
        batch_size,
        snapshot,
        weighted=False,
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = GPT2ForCivicEvidenceClassification.from_biomed_lm_snapshot()
        if snapshot:
            model = GPT2ForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="gpt2",
            snapshot_name="biomed_lm",
            tokenizer_max_length=1024,
            weighted=weighted,
        )

    @staticmethod
    def create_biomed_roberta_long_pre_training_model_trainer(training_args, snapshot):
        tokenizer, model = RobertaLongForMaskedLM.from_biomed_roberta_snapshot()
        print(model)
        if snapshot:
            print(f"loading model from snapshot {snapshot}")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        dataset = load_from_disk(
            os.path.join(HF_DATA_CACHE_DIR, "pubmed-cache/pubmed_processed")
        )
        train_dataset = dataset["train"]
        val_dataset = Dataset.from_dict(dataset["test"][45000:48000])
        trainer = Trainer(
            tokenizer=tokenizer,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        return MlmRobertaLongPreTrainingModelTrainer(trainer)
