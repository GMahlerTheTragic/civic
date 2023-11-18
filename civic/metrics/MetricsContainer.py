from dataclasses import dataclass

import torch


@dataclass
class MetricsContainer:
    num_samples: int
    num_correct: int
    true_positives: torch.Tensor
    false_positives: torch.Tensor
    false_negatives: torch.Tensor
    val_loss: torch.Tensor
