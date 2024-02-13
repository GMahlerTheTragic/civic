import os.path

from datasets import load_from_disk, Dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import (
    GPT2ForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    LongformerForSequenceClassification,
    DataCollatorForLanguageModeling,
    Trainer,
)

from civic.config import HF_DATA_CACHE_DIR
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
from civic.training.mlm_bert_long_pretraining.MlmBertLongPreTrainingModelTrainer import (
    MlmRobertaLongPreTrainingModelTrainer,
)


class ModelTrainerFactory:
    def __init__(self, accelerator):
        self.accelerator = accelerator

    @staticmethod
    def _get_dataloaders_for_civic_evidence_finetuning(
        tokenizer,
        batch_size,
        tokenizer_max_length,
        use_prepend_string=False,
        use_full_data_set=False,
    ):
        if use_full_data_set:
            train_dataset = CivicEvidenceDataSet.full_train_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=use_prepend_string
            )
            test_dataset = CivicEvidenceDataSet.full_validation_dataset(
                tokenizer, tokenizer_max_length, use_prepend_string=use_prepend_string
            )
        else:
            train_dataset = CivicEvidenceDataSet.train_dataset_unique_abstracts(
                tokenizer, tokenizer_max_length, use_prepend_string=use_prepend_string
            )
            test_dataset = CivicEvidenceDataSet.validation_dataset_unique_abstracts(
                tokenizer, tokenizer_max_length, use_prepend_string=use_prepend_string
            )
        train_data_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        validation_data_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_data_loader, validation_data_loader

    def _get_classification_steps(self, weights=None):
        criterion = (
            CrossEntropyLoss(reduction="sum", weight=weights.float())
            if weights is not None
            else None
        )
        batch_training_step: BatchTrainingStep = ClassifierFineTuningBatchTrainingStep(
            self.accelerator.device, self.accelerator, criterion
        )
        batch_validation_step: BatchValidationStep = (
            ClassifierFineTuningBatchValidationStep(self.accelerator.device, criterion)
        )
        return batch_training_step, batch_validation_step

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
    ):
        model.to(self.accelerator.device)
        (
            train_data_loader,
            validation_data_loader,
        ) = ModelTrainerFactory._get_dataloaders_for_civic_evidence_finetuning(
            tokenizer, batch_size, tokenizer_max_length
        )
        weights = (
            train_data_loader.dataset.inverse_class_prob_weights if weighted else None
        )
        batch_training_step, batch_validation_step = self._get_classification_steps(
            weights=weights
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
                "unique_abstracts": "True",
            },
        )
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        (
            model,
            optimizer,
        ) = self.accelerator.prepare(model, optimizer)
        train_data_loader = self.accelerator.prepare(train_data_loader)
        validation_data_loader = self.accelerator.prepare(validation_data_loader)

        return ClassifierFineTuningModelTrainer(
            training_monitor,
            batch_training_step,
            batch_validation_step,
            train_data_loader,
            validation_data_loader,
            model,
            optimizer,
            self.accelerator,
        )

    def create_longformer_base_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_longformer_base()
        if snapshot:
            model = LongformerForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="longformer",
            snapshot_name="allenai-longformer-base",
            tokenizer_max_length=4096,
            weighted=weighted,
        )

    def create_roberta_base_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_roberta_base()
        if snapshot:
            model = RobertaForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta",
            snapshot_name="roberta-base",
            tokenizer_max_length=512,
            weighted=weighted,
        )

    def create_biomed_roberta_base_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_biomed_roberta_base()
        if snapshot:
            model = RobertaForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta",
            snapshot_name="biomed_roberta_base",
            tokenizer_max_length=512,
            weighted=weighted,
        )

    def create_biomed_roberta_long_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = RobertaForCivicEvidenceClassification.from_long_biomed_roberta_pretrained()
        if snapshot:
            model = RobertaLongForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="roberta_long",
            snapshot_name="biomed_roberta_long",
            tokenizer_max_length=1024,
            weighted=weighted,
        )

    def create_bert_base_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bert_base_uncased()
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bert-base-uncased",
            tokenizer_max_length=512,
            weighted=weighted,
        )

    def create_pubmed_bert_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_pubmed_bert()
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="pubmed_bert",
            tokenizer_max_length=512,
            weighted=weighted,
        )

    def create_bio_link_bert_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert()
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bio_link_bert",
            tokenizer_max_length=512,
            weighted=weighted,
        )

    def create_bio_link_bert_large_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
    ) -> ModelTrainer:
        (
            tokenizer,
            model,
        ) = BertForCivicEvidenceClassification.from_bio_link_bert_large()
        if snapshot:
            model = BertForSequenceClassification.from_pretrained(snapshot)
        return self._get_trainer_from_model(
            model,
            tokenizer,
            batch_size,
            learning_rate,
            architecture="bert",
            snapshot_name="bio_link_bert_long",
            tokenizer_max_length=512,
            weighted=weighted,
        )

    def create_biomed_lm_finetuning_model_trainer(
        self, learning_rate, batch_size, snapshot, weighted=False
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
