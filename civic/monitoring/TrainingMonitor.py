from abc import abstractmethod, ABC


class ITrainingMonitor(ABC):
    @abstractmethod
    def __enter__(self):
        raise NotImplementedError

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError

    @abstractmethod
    def save_model_checkpoint(self, epoch, model, optimizer, training_loss):
        raise NotImplementedError

    @abstractmethod
    def log_training_metrics(self, epoch, total_epochs, metrics):
        raise NotImplementedError
