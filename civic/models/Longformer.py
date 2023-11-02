import torch
import wandb
from torch.optim import AdamW

from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet


class Longformer:
    @staticmethod
    def from_longformer_allenai_base_pretrained():
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(
            model_name, num_labels=5
        )
        return tokenizer, model

    @staticmethod
    def from_roberta_snapshot():
        raise NotImplementedError


def train():
    tokenizer, model = Longformer.from_longformer_allenai_base_pretrained()
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_dataset = CivicEvidenceDataSet.full_train_dataset(tokenizer, 4096)
    test_dataset = CivicEvidenceDataSet.full_test_dataset(tokenizer, 4096)

    # Define data loaders for batching and shuffling
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set training parameters
    num_epochs = 3
    learning_rate = 2e-5

    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    run = wandb.init(
        project="civic",
        config={
            "learning_rate": learning_rate,
            "architecture": "Longformer",
            "dataset": "CivicEvidenceDataSetFull",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "distributed": False,
        },
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        idx = 0
        for batch in train_loader:
            if idx == 2:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            idx += 1
            print(f"\rProcessed {idx}/{len(train_loader)} batches", end="", flush=True)

        # Validation loop
        model.eval()
        val_loss = 0.0
        num_correct = 0
        num_samples = 0
        all_predictions = torch.tensor([])
        all_real_labels = torch.tensor([])

        with torch.no_grad():
            idx = 0
            for batch in test_loader:
                if idx == 2:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                val_loss += loss.item()
                predicted_labels = torch.argmax(logits, dim=1)
                print(predicted_labels)
                num_correct += torch.eq(predicted_labels, labels).sum().item()
                num_samples += labels.size(0)
                all_predictions = torch.concat([all_predictions, predicted_labels])
                all_real_labels = torch.concat([all_real_labels, labels])
                idx += 1
        print(all_predictions)
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        print(num_correct)
        print(num_samples)
        val_accuracy = num_correct / num_samples
        micro_f1_score = multiclass_f1_score(
            all_predictions, all_real_labels, num_classes=5
        )

        wandb.log(
            {
                "val_accuracy": val_accuracy,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            "Longformer/model.pth",
        )
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file("Longformer/model.pth")
        run.log_artifact(artifact)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            f" | Val Accuracy: {val_accuracy:.4f} | Val MicroF1: {micro_f1_score:.4f}"
        )
    wandb.finish()


if __name__ == "__main__":
    train()
