import sys

sys.path.append("../../../civic")

import argparse

from civic.training import IModelTrainer
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


parser = argparse.ArgumentParser(
    description="My Python script with command-line options."
)

parser.add_argument(
    "--instance", choices=["LongFormerBaseFineTuning"], help="What to train for"
)
parser.add_argument("--epochs", type=positive_integer, help="How many epochs to train")
parser.add_argument(
    "--epochbatches", type=positive_integer, help="How many batches per epoch"
)
parser.add_argument(
    "--learningrate", type=positive_float, help="The learning rate for gradient descend"
)
parser.add_argument(
    "--batchsize",
    type=positive_integer,
    help="The batch size to use during training and validation",
)
parser.add_argument(
    "--lossfunction",
    choices=["MSE", "BCE"],
    help="The loss function to compute gradients for",
)
parser.add_argument(
    "--acceptedonly",
    choices=["true", "false"],
    help="If only to use accepted evidence items",
)

args = parser.parse_args()


def main():
    model_trainer: IModelTrainer = None
    if args.instance == "LongFormerBaseFineTuning":
        model_trainer: IModelTrainer = (
            ModelTrainerFactory.create_longformer_base_finetuning_model_trainer(
                args.learningrate, args.batchsize
            )
        )
    model_trainer.do_model_training(args.epochs)


if __name__ == "__main__":
    main()
