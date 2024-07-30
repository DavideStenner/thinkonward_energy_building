from abc import ABC, abstractmethod

class ModelBackTest(ABC):    
    @abstractmethod
    def backtest(self) -> None: 
        pass