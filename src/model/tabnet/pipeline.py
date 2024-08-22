from typing import Any, Tuple

from src.model.tabnet.training import TabnetTrainer
from src.model.tabnet.initialize import TabnetInit
from src.model.tabnet.inference import TabnetInference
from src.model.tabnet.explainer import TabnetExplainer

from src.base.model.pipeline import ModelPipeline

class TabnetPipeline(ModelPipeline, TabnetTrainer, TabnetExplainer, TabnetInference):
    def __init__(self, 
            experiment_name:str, 
            params_tabnet: dict[str, Any],
            config_dict: dict[str, Any], data_columns: Tuple[str],
            fold_name: str = 'fold_info', 
        ):
        TabnetInit.__init__(
            self, experiment_name=experiment_name, params_tabnet=params_tabnet,
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