from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score
from torch import softmax
from torch.utils.data import DataLoader
from pandas import DataFrame

import torch


from civic.evaluation.ModelEvaluator import ModelEvaluator


class MultiLabelClassifierModelEvaluator(ModelEvaluator):
    def __init__(
        self,
        test_data_loader: DataLoader,
        model,
        device,
        integrated_gradient_wrapper,
        test_data_loader_long: DataLoader,
        test_data_loader_gpt4: DataLoader,
        val_data_loader: DataLoader,
    ):
        self.test_data_loader = test_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader_long = test_data_loader_long
        self.test_data_loader_gpt4 = test_data_loader_gpt4
        self.model = model
        self.device = device
        self.integrated_gradient_wrapper = integrated_gradient_wrapper
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _get_prc_optimal_threshold(actual_labels, predicted_probabilities):
        prc = [
            precision_recall_curve(
                actual_labels[:, c],
                predicted_probabilities[:, c],
            )
            for c in range(actual_labels.shape[1])
        ]
        prcs = [
            DataFrame(p)
            .transpose()
            .rename(columns={0: "precision", 1: "recall", 2: "threshold"})
            for p in prc
        ]
        for p in prcs:
            p["f1"] = 2 * p["precision"] * p["recall"] / (p["precision"] + p["recall"])
        return [p["threshold"].iloc[p["f1"].argmax()] for p in prcs]

    def _find_optimal_threshold_on_val(self):
        actual_labels = []
        predicted_probabilities = []
        for idx, batch in enumerate(self.val_data_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_probabilities.append(torch.sigmoid(logits.detach()))
            print(
                f"\rProcessed {idx + 1}/{len(self.val_data_loader)} batches",
                end="",
                flush=True,
            )
        thresholds = self._get_prc_optimal_threshold(
            torch.concat(actual_labels, dim=0).cpu(),
            torch.concat(predicted_probabilities, dim=0).cpu(),
        )
        return thresholds

    def _evaluate(self, dataloader: DataLoader, thresholds):
        predicted_labels = []
        actual_labels = []
        for idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels.append(
                torch.where(
                    torch.sigmoid(logits.detach()).cpu() < torch.tensor(thresholds),
                    torch.tensor(0.0),
                    torch.tensor(1.0),
                )
            )
            print(
                f"\rProcessed {idx + 1}/{len(dataloader)} batches",
                end="",
                flush=True,
            )
        return {
            "predicted-labels": torch.concat(predicted_labels).tolist(),
            "f1-scores": f1_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
                average=None,
            ).tolist(),
            "macro-f1-score": f1_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
                average="macro",
            ).item(),
            "weighted-f1-score": f1_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
                average="weighted",
            ).item(),
            "accuracy": accuracy_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
            ).item(),
        }

    def _evaluate_with_explainability(self, dataloader: DataLoader, thresholds):
        predicted_labels = []
        actual_labels = []
        attribution_list = []
        for idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(self.device)
            ref_ids = batch["input_ref_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            actual_labels.append(batch["label"].to(self.device))
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels.append(
                torch.where(
                    torch.sigmoid(logits.detach()).cpu() < torch.tensor(thresholds),
                    torch.tensor(0.0),
                    torch.tensor(1.0),
                )
            )
            attribution_list.append(
                {
                    "input_text": batch["input_text"],
                    "attribution_scores": [
                        self.integrated_gradient_wrapper.do_attribution(
                            input_ids,
                            ref_ids,
                            attention_mask,
                            target=i,
                        ).tolist()
                        for i in range(5)
                    ],
                    "actual_label": batch["label"].tolist(),
                }
            )
            print(
                f"\rProcessed {idx + 1}/{len(dataloader)} batches",
                end="",
                flush=True,
            )
        return {
            "predicted-labels": torch.concat(predicted_labels).tolist(),
            "f1-scores": f1_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
                average=None,
            ).tolist(),
            "macro-f1-score": f1_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
                average="macro",
            ).item(),
            "weighted-f1-score": f1_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
                average="weighted",
            ).item(),
            "accuracy": accuracy_score(
                torch.concat(actual_labels, dim=0).cpu(),
                torch.concat(predicted_labels, dim=0).cpu(),
            ).item(),
            "ig_attributions": attribution_list,
        }

    def do_evaluation(self):
        print("Getting thresholds on val set...")
        thresholds = self._find_optimal_threshold_on_val()
        print("thresholds are")
        print(thresholds)
        print("Evaluating on test set")
        metrics_test = self._evaluate(self.test_data_loader, thresholds=thresholds)
        print("\rEvaluating on long abstracts only...")
        metrics_test_long = self._evaluate(
            self.test_data_loader_long, thresholds=thresholds
        )
        print("\rEvaluating on explainability subset...")

        print("f1s: {} \n".format(metrics_test["weighted-f1-score"]))
        print("f1-weighted: {} \n".format(metrics_test["f1-scores"]))
        print("accuracy: {} \n".format(metrics_test["accuracy"]))
        metrics_test_gpt4 = self._evaluate_with_explainability(
            self.test_data_loader_gpt4, thresholds=thresholds
        )
        values = {
            "metrics_test": metrics_test,
            "metrics_test_long": metrics_test_long,
            "metrics_test_gpt4": metrics_test_gpt4,
        }

        return values
