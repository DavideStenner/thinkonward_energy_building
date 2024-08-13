import torch

import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score
from src.base.nn.loss import Metric
from src.model.metric.official_metric import softmax_by_target, f1_by_target

class AUCScore(Metric):
    @property
    def name(self) -> str:
        return 'auc'
        
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        score_ = roc_auc_score(
            y_true=y_true.numpy(), y_score=y_pred.numpy()
        )
        return score_

class ClusteredF1Score(Metric):
    def __init__(self, target_position_list:list[int]):
        self.target_position_list:list[int] = target_position_list
        
    @property
    def name(self) -> str:
        return 'f1'
        
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        
        y_pred = softmax_by_target(
            target_position_list=self.target_position_list, 
            y_pred=y_pred
        )
        f1_score_list = f1_by_target(
            self.target_position_list, 
            y_true=y_true, y_pred=y_pred 
        )

        score_ = np.mean(f1_score_list)
        return score_

class ClusteredCrossEntropyLoss(nn.Module):
    def __init__(self, target_position_list:list[int]):
        super().__init__()
        self.target_position_list: list[int] = target_position_list
        self.base_criterion = nn.CrossEntropyLoss()
        
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ = 0
        for target_position in self.target_position_list:
            loss_ += self.base_criterion(
                input_[:, target_position],
                target[:, target_position]
            )
        
        return loss_
            
        