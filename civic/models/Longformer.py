import torch
from torch.utils.data import DataLoader
from transformers import AdamW, LongformerTokenizer, LongformerForSequenceClassification

from civic.datasets.CivicEvidenceDataSet import CivicEvidenceDataSet


class Longformer:
    @staticmethod
    def create():
        model_name = "allenai/longformer-base-4096"
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForSequenceClassification.from_pretrained(
            model_name, num_labels=5
        )
        return tokenizer, model


def train():
    tokenizer, model = Longformer.create()
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_dataset = CivicEvidenceDataSet.full_train_dataset(tokenizer)
    test_dataset = CivicEvidenceDataSet.full_test_dataset(tokenizer)

    # Define data loaders for batching and shuffling
    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Set training parameters
    num_epochs = 3
    learning_rate = 2e-5

    # Define the optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)

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
                predicted_labels = torch.argmax(logits)
                num_correct += (predicted_labels == labels).sum().item()
                num_samples += labels.size(0)
                idx += 1

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        val_accuracy = num_correct / num_samples

        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}"
        )


if __name__ == "__main__":
    train()
