from abc import ABC, abstractmethod

class Metric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        pass