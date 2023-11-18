from abc import abstractmethod, ABC


class IBatchTrainingStep(ABC):
    @abstractmethod
    def train_batch(self, batch, model, optimizer, lr_scheduler):
        raise NotImplementedError
