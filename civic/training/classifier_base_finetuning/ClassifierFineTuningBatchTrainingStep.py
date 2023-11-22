from civic.training.BatchTrainingStep import BatchTrainingStep


class ClassifierFineTuningBatchTrainingStep(BatchTrainingStep):
    def __init__(self, device, accelerator):
        self.device = device
        self.accelerator = accelerator

    def train_batch(self, batch, model, optimizer, lr_scheduler):
        batch_size = batch["input_ids"].size(0)
        with self.accelerator.accelerator.accumulate(model):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            self.accelerator.accelerator.backward(loss)
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

        return loss.item() * batch_size
