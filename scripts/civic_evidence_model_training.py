import math

from civic.config import MODEL_STORAGE_DIR
from civic.models.roberta.RobertaLongModelArgs import RobertaLongModelArgs


import os
import sys
import argparse

from transformers import HfArgumentParser, TrainingArguments


from civic.training import ModelTrainer
from civic.training.ModelTrainerFactory import ModelTrainerFactory


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
        "Longformer",
        "BioMedLMFineTuning",
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

args = parser.parse_args()


def main():
    if args.instance == "Bert":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_bert_base_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "PubmedBert":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_pubmed_bert_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "BiolinkBert":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_bio_link_bert_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "Roberta":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_roberta_base_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "BiomedRoberta":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_biomed_roberta_base_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "Longformer":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_longformer_base_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "BioMedLMFineTuning":
        model_trainer: ModelTrainer = (
            ModelTrainerFactory.create_biomed_lm_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    elif args.instance == "BiomedBertPretraining":
        hf_parser = HfArgumentParser(
            (
                TrainingArguments,
                RobertaLongModelArgs,
            )
        )
        training_args, model_args = hf_parser.parse_args_into_dataclasses(
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
                "--per_device_train_batch_size",
                "8",
                "--per_device_eval_batch_size",
                "2",  # 32GB gpu with fp32
                "--gradient_accumulation_steps",
                "32",
            ],
        )
        training_args.val_datapath = "data/02_processed/output_file.txt"
        training_args.train_datapath = "data/02_processed/output_file.txt"
        model_trainer = (
            ModelTrainerFactory.create_biomed_roberta_long_pre_training_model_trainer(
                training_args
            )
        )
        model_path = os.path.join(MODEL_STORAGE_DIR, "biomed_roberta_base-1024")
        print(f"Pretraining roberta-base-{model_args.max_pos} ... ")
        training_args.max_steps = 3
        eval_loss = model_trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        print(f"Initial eval bpc: {eval_loss/math.log(2)}")
        model_trainer.train(model_path)
        model_trainer.save_model()
        eval_loss = model_trainer.evaluate()
        eval_loss = eval_loss["eval_loss"]
        print(f"Eval bpc after pretraining: {eval_loss/math.log(2)}")
    else:
        print("Please provide a valid instance name.")
        sys.exit(-1)
    model_trainer.do_model_training(args.epochs)


if __name__ == "__main__":
    main()
