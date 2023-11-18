from abc import abstractmethod, ABC


class IModelTrainer(ABC):
    @abstractmethod
    def do_model_training(self, n_epochs):
        raise NotImplementedError
