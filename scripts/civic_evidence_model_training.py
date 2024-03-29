from civic.models.roberta.RobertaLongModelArgs import RobertaLongModelArgs

import sys
import argparse

from transformers import (
    HfArgumentParser,
    TrainingArguments,
)

from civic.training import ModelTrainer
from civic.training.ModelTrainerFactory import (
    ModelTrainerFactory,
    CivicModelTrainingMode,
)
from civic.utils.AcceleratorSingleton import AcceleratorSingleton


def positive_integer(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer.")
    return ivalue


def positive_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a valid learning rate.")
    return fvalue


parser = argparse.ArgumentParser(description="Civic Model Training")

parser.add_argument(
    "--instance",
    choices=[
        "Bert",
        "PubmedBert",
        "BiolinkBert",
        "Roberta",
        "BiomedRoberta",
        "BioMedLMFineTuning",
        "BiomedRobertaLongPreTraining",
        "BiomedRobertaLong",
        "BiolinkBertLarge",
    ],
    help="What to train for",
)
parser.add_argument("--epochs", type=positive_integer, help="How many epochs to train")
parser.add_argument(
    "--learningrate", type=positive_float, help="The learning rate for gradient descend"
)
parser.add_argument(
    "--batchsize",
    type=positive_integer,
    help="The batch size to use during training and validation",
)
parser.add_argument(
    "--accumulation", type=positive_integer, help="Gradient accumulation steps"
)
parser.add_argument("--weighted", type=bool)
parser.add_argument("--resume")
parser.add_argument(
    "--mode",
    choices=[
        "ABSTRACTS_ONLY_MULTILABEL",
        "ABSTRACTS_ONLY_UNIQUE_ONLY",
        "ABSTRACTS_PLUS_PREPEND_METADATA",
    ],
    default="ABSTRACTS_ONLY_UNIQUE_ONLY",
)
args = parser.parse_args()


def _get_mode_from_mode_option(mode_option):
    if mode_option == "ABSTRACTS_ONLY_MULTILABEL":
        return CivicModelTrainingMode.ABSTRACTS_ONLY_MULTILABEL
    elif mode_option == "ABSTRACTS_ONLY_UNIQUE_ONLY":
        return CivicModelTrainingMode.ABSTRACTS_ONLY_UNIQUE_ONLY
    elif mode_option == "ABSTRACTS_PLUS_PREPEND_METADATA":
        return CivicModelTrainingMode.ABSTRACTS_PLUS_PREPEND_METADATA
    else:
        raise RuntimeError("Enum option not covered!")


def main():
    accelerator = AcceleratorSingleton(gradient_accumulation_steps=args.accumulation)
    model_trainer_factory = ModelTrainerFactory(accelerator)
    snapshot = args.resume if args.resume else None
    compute_weighted_loss = args.weighted
    mode = _get_mode_from_mode_option(args.mode)

    if args.instance == "Bert":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_bert_base_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
                compute_weighted_loss,
                mode=mode,
            )
        )
    elif args.instance == "PubmedBert":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_pubmed_bert_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
                compute_weighted_loss,
                mode=mode,
            )
        )
    elif args.instance == "BiolinkBert":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_bio_link_bert_finetuning_model_trainer(
                args.learningrate, args.batchsize, snapshot, compute_weighted_loss
            )
        )
    elif args.instance == "BiolinkBertLarge":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_bio_link_bert_large_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
                compute_weighted_loss,
                mode=mode,
            )
        )
    elif args.instance == "Roberta":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_roberta_base_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
                compute_weighted_loss,
                mode=mode,
            )
        )
    elif args.instance == "BiomedRoberta":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_biomed_roberta_base_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
                compute_weighted_loss,
                mode=mode,
            )
        )
    elif args.instance == "BiomedRobertaLong":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_biomed_roberta_long_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
                compute_weighted_loss,
                mode=mode,
            )
        )
    elif args.instance == "BioMedLMFineTuning":
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_biomed_lm_finetuning_model_trainer(
                args.learningrate,
                args.batchsize,
                snapshot,
            )
        )
    elif args.instance == "BiomedRobertaLongPreTraining":
        hf_parser = HfArgumentParser(
            (
                TrainingArguments,
                RobertaLongModelArgs,
            )
        )
        training_args, _ = hf_parser.parse_args_into_dataclasses(
            look_for_args_file=False,
            args=[
                "--output_dir",
                "tmp2",
                "--warmup_steps",
                "500",
                "--learning_rate",
                "0.00003",
                "--weight_decay",
                "0.0",
                "--adam_epsilon",
                "1e-6",
                "--max_steps",
                "3000",
                "--logging_steps",
                "100",
                "--save_steps",
                "500",
                "--max_grad_norm",
                "5.0",
                "--per_device_train_batch_size",
                "8",
                "--per_device_eval_batch_size",
                "8",  # 32GB gpu with fp32
                "--gradient_accumulation_steps",
                "16",
                "--fp16",
            ],
        )
        model_trainer: ModelTrainer = (
            model_trainer_factory.create_biomed_roberta_long_pre_training_model_trainer(
                training_args, snapshot
            )
        )
    else:
        print("Please provide a valid instance name.")
        sys.exit(-1)
    model_trainer.do_model_training(args.epochs)


if __name__ == "__main__":
    main()
