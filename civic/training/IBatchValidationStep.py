from abc import abstractmethod


class IBatchValidationStep:
    @abstractmethod
    def validate_batch(self, batch, model):
        raise NotImplementedError
