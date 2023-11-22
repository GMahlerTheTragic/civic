from abc import abstractmethod, ABC


class BatchValidationStep(ABC):
    @abstractmethod
    def validate_batch(self, batch, model):
        raise NotImplementedError
