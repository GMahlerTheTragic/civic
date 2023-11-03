from abc import abstractmethod


class IBatchTrainingStep:
    @abstractmethod
    def train_batch(self, batch, model, optimizer, lr_scheduler):
        raise NotImplementedError
