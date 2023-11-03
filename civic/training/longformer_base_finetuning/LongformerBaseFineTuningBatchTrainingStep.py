from civic.training.IBatchTrainingStep import IBatchTrainingStep


class LongformerBaseFineTuningBatchTrainingStep(IBatchTrainingStep):
    def __init__(self, device):
        self.device = device

    def train_batch(self, batch, model, optimizer, lr_scheduler):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        return loss.item()
