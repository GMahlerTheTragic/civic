import torch
import wandb
from torch.optim import AdamW

from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet


class Longformer:
    @staticmethod
    def from_last_stanford_snapshot():
        raise NotImplementedError


def train():
    raise NotImplementedError


if __name__ == "__main__":
    train()
