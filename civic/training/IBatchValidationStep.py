from abc import abstractmethod, ABC


class IBatchValidationStep(ABC):
    @abstractmethod
    def validate_batch(self, batch, model):
        raise NotImplementedError
