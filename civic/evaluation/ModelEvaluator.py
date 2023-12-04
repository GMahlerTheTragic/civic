from abc import ABC, abstractmethod


class ModelEvaluator(ABC):
    @abstractmethod
    def do_evaluation(self):
        raise NotImplementedError
