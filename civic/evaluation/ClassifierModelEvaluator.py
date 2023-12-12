from torch import softmax
from torch.utils.data import DataLoader

import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import Accuracy

from civic.evaluation.ModelEvaluator import ModelEvaluator


class ClassifierModelEvaluator(ModelEvaluator):
    def __init__(
        self,
        test_data_loader: DataLoader,
        model,
        device,
        integrated_gradient_wrapper,
        test_data_loader_long: DataLoader,
        test_data_loader_gpt4: DataLoader,
    ):
        self.test_data_loader = test_data_loader
        self.test_data_loader_long = test_data_loader_long
        self.test_data_loader_gpt4 = test_data_loader_gpt4
        self.model = model
        self.device = device
        self.integrated_gradient_wrapper = integrated_gradient_wrapper
        self.f1_score = MulticlassF1Score(num_classes=5, average=None).to(self.device)
        self.macro_f1_score = MulticlassF1Score(num_classes=5, average="macro").to(
            self.device
        )
        self.micro_f1_score = MulticlassF1Score(num_classes=5, average="micro").to(
            self.device
        )
        self.accuracy = Accuracy(task="multiclass", num_classes=5).to(self.device)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.eval()

    def _evaluate(self, dataloader: DataLoader):
        predicted_labels = []
        actual_labels = []
        predicted_logits = []
        for idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_logits.append(logits.detach())
            predicted_labels.append(torch.argmax(logits, dim=1))
            print(
                f"\rProcessed {idx + 1}/{len(dataloader)} batches",
                end="",
                flush=True,
            )
        return {
            "f1-scores": self.f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).tolist(),
            "micro-f1-score": self.micro_f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
            "macro-f1-score": self.macro_f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
            "accuracy": self.accuracy(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
            "cross_entropy": self.cross_entropy(
                torch.concat(predicted_logits, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
        }

    def _evaluate_with_explainability(self, dataloader: DataLoader):
        predicted_labels = []
        actual_labels = []
        attribution_list = []
        predicted_logits = []
        for idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            ref_ids = batch["input_ref_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels.append(torch.argmax(logits, dim=1))
            predicted_logits.append(logits.detach())
            attribution_list.append(
                {
                    "input_text": batch["input_text"],
                    "attribution_scores": self.integrated_gradient_wrapper.do_attribution(
                        input_ids,
                        ref_ids,
                        attention_mask,
                        target=torch.argmax(logits, dim=1).item(),
                    ).tolist(),
                    "actual_label": batch["label"].item(),
                    "predicted_label": torch.argmax(logits, dim=1).item(),
                    "probabilities": softmax(logits, dim=1).item(),
                }
            )
            print(
                f"\rProcessed {idx + 1}/{len(dataloader)} batches",
                end="",
                flush=True,
            )
        return {
            "f1-scores": self.f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).tolist(),
            "micro-f1-score": self.micro_f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
            "macro-f1-score": self.macro_f1_score(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
            "accuracy": self.accuracy(
                torch.concat(predicted_labels, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
            "ig_attributions": attribution_list,
            "cross_entropy": self.cross_entropy(
                torch.concat(predicted_logits, dim=0),
                torch.concat(actual_labels, dim=0),
            ).item(),
        }

    def do_evaluation(self):
        print("Evaluating on full test set...")
        metrics_test = self._evaluate(self.test_data_loader)
        print("\rEvaluating on long abstracts only...")
        metrics_test_long = self._evaluate(self.test_data_loader_long)
        print("\rEvaluating on explainability subset...")
        print(metrics_test)
        print(metrics_test_long)
        metrics_test_gpt4 = self._evaluate(self.test_data_loader_gpt4)
        values = {
            "metrics_test": metrics_test,
            "metrics_test_long": metrics_test_long,
            "metrics_test_gpt4": metrics_test_gpt4,
        }

        return values
