from civic.evaluation.ModelEvaluator import ModelEvaluator
from civic.evaluation.ModelEvaluatorFactory import ModelEvaluatorFactory


import sys
import argparse


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
        "BiomedRobertaLongPreTraining",
        "BiomedRobertaLong",
        "BiolinkBertLarge",
    ],
    help="What to train for",
)
parser.add_argument("--snapshot")
parser.add_argument(
    "--batchsize",
    type=positive_integer,
    help="The batch size to use during training and validation",
)
args = parser.parse_args()


def main():
    model_evaluator_factory = ModelEvaluatorFactory()
    snapshot = args.snapshot
    batch_size = args.batchsize
    if args.instance == "BiomedRoberta":
        model_trainer: ModelEvaluator = (
            model_evaluator_factory.create_biomed_roberta_model_evaluator(
                snapshot, batch_size
            )
        )
    else:
        print("Please provide a valid instance name.")
        sys.exit(-1)
    metrics = model_trainer.do_evaluation()
    print(metrics)


if __name__ == "__main__":
    main()
