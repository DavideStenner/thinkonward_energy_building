import numpy as np
import polars as pl

from src.base.model.inference import ModelPredict
from src.model.nn.initialize import TabularFFInit

class TabularFFInference(ModelPredict, TabularFFInit):     
    def load_feature_data(self, data: pl.DataFrame) -> any:
        pass
        
    def blend_model_predict(self, test_data: pl.DataFrame) -> np.ndarray:             
        pass
    
    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        pass