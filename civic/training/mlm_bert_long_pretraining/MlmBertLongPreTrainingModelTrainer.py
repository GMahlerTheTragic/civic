import math
from dataclasses import dataclass, field

from torch.utils.data import DataLoader
import torch
from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)

from civic.monitoring import TrainingMonitor
from civic.training.BatchTrainingStep import BatchTrainingStep
from civic.training.BatchValidationStep import BatchValidationStep
from civic.training.ModelTrainer import ModelTrainer

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


@dataclass
class ModelArgs:
    attention_window: int = field(
        default=512, metadata={"help": "Size of attention window"}
    )
    max_pos: int = field(default=1024, metadata={"help": "Maximum position"})


parser = HfArgumentParser(
    (
        TrainingArguments,
        ModelArgs,
    )
)


training_args, model_args = parser.parse_args_into_dataclasses(
    look_for_args_file=False,
    args=[
        "--output_dir",
        "tmp",
        "--warmup_steps",
        "500",
        "--learning_rate",
        "0.00003",
        "--weight_decay",
        "0.01",
        "--adam_epsilon",
        "1e-6",
        "--max_steps",
        "3000",
        "--logging_steps",
        "500",
        "--save_steps",
        "500",
        "--max_grad_norm",
        "5.0",
        "--per_gpu_eval_batch_size",
        "8",
        "--per_gpu_train_batch_size",
        "2",  # 32GB gpu with fp32
        "--gradient_accumulation_steps",
        "32",
        "--evaluate_during_training",
        "--do_train",
        "--do_eval",
    ],
)
training_args.val_datapath = "wikitext-103-raw/wiki.valid.raw"
training_args.train_datapath = "wikitext-103-raw/wiki.train.raw"


class MlmBertLongPreTrainingModelTrainer(ModelTrainer):
    def __init__(
        self,
        training_monitor: TrainingMonitor,
        batch_training_step: BatchTrainingStep,
        batch_validation_step: BatchValidationStep,
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
        """TODO"""
        pass

    @staticmethod
    def do_pre_training(args, model, tokenizer, eval_only, model_path):
        val_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=args.val_datapath,
            block_size=tokenizer.max_len,
        )
        if eval_only:
            train_dataset = val_dataset
        else:
            print(
                f"Loading and tokenizing training data is usually slow: {args.train_datapath}"
            )
            train_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=args.train_datapath,
                block_size=tokenizer.max_len,
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            prediction_loss_only=True,
        )

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        print(f"Initial eval bpc: {eval_loss/math.log(2)}")

        if not eval_only:
            trainer.train(model_path=model_path)
            trainer.save_model()

            eval_loss = trainer.evaluate()
            eval_loss = eval_loss["eval_loss"]
            print(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")
