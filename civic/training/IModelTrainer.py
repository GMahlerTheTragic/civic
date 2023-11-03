from abc import abstractmethod


class IModelTrainer:
    @abstractmethod
    def do_model_training(self, n_epochs):
        raise NotImplementedError
