from civic.training.IBatchTrainingStep import IBatchTrainingStep
from civic.utils.GpuDeviceHandler import GpuDeviceHandler


class ClassifierFineTuningBatchTrainingStep(IBatchTrainingStep):
    def __init__(self, device, accelerator):
        self.device = device
        self.accelerator = accelerator

    def train_batch(self, batch, model, optimizer, lr_scheduler):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # if GpuDeviceHandler.is_model_using_data_parallel(model):
        #     loss = loss.sum()

        self.accelerator.accelerator.backward(loss)
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        return loss.item()
