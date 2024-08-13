from typing import Any, Tuple

from src.model.nn.training import TabularFFTrainer
from src.model.nn.initialize import TabularFFInit
from src.model.nn.inference import TabularFFInference
from src.model.nn.explainer import TabularFFExplainer

from src.base.model.pipeline import ModelPipeline

class TabularFFPipeline(ModelPipeline, TabularFFTrainer, TabularFFExplainer, TabularFFInference):
    def __init__(self, 
            experiment_name:str, 
            params_nn: dict[str, Any],
            config_dict: dict[str, Any], data_columns: Tuple[str],
            fold_name: str = 'fold_info', 
        ):
        TabularFFInit.__init__(
            self, 
            experiment_name=experiment_name, params_nn=params_nn,
            config_dict=config_dict,
            data_columns=data_columns, 
            fold_name=fold_name
        )
        
    def activate_inference(self) -> None:
        self.load_model()
        self.inference = True
        
    def run_train(self) -> None:
        self.train()
        self.save_model()
        
    def explain_model(self) -> None:
        self.evaluate_score()
        self.get_feature_importance()
        self.get_oof_prediction()
        self.get_oof_insight()
        
    def train_explain(self) -> None:
        self.create_experiment_structure()
        self.initialize_logger()
        self.run_train()
        self.explain_model()