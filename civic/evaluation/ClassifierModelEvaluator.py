from torch.utils.data import DataLoader

import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy

from civic.evaluation.ModelEvaluator import ModelEvaluator


class ClassifierModelEvaluator(ModelEvaluator):
    def __init__(
        self, test_data_loader: DataLoader, model, device, integrated_gradient_wrapper
    ):
        self.test_data_loader = test_data_loader
        self.model = model
        self.device = device
        self.integrated_gradient_wrapper = integrated_gradient_wrapper

    def do_evaluation(self):
        f1_score = MulticlassF1Score(num_classes=5, average=None).to(self.device)
        macro_f1_score = MulticlassF1Score(num_classes=5, average="macro").to(
            self.device
        )
        micro_f1_score = MulticlassF1Score(num_classes=5, average="micro").to(
            self.device
        )
        accuracy = Accuracy(task="multiclass", num_classes=5).to(self.device)

        predicted_labels = []
        actual_labels = []
        attribution_list = []

        self.model.to(self.device)
        self.model.eval()

        for idx, batch in enumerate(self.test_data_loader):
            input_ids = batch["input_ids"].to(self.device)
            ref_ids = batch["input_ref_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels.append(torch.argmax(logits, dim=1))
            attribution_list.append(
                self.integrated_gradient_wrapper.do_attribution(
                    input_ids, ref_ids, attention_mask
                )
            )
            print(
                self.integrated_gradient_wrapper.do_attribution(
                    input_ids, ref_ids, attention_mask
                )
            )
            print(
                f"\rProcessed {idx}/{len(self.test_data_loader)} batches",
                end="",
                flush=True,
            )

        return {
            "f1-scores": f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ),
            "micro-f1-scores": micro_f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ),
            "macro-f1-scores": macro_f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ),
            "accuracy": accuracy(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ),
            "attributions": attribution_list,
        }
