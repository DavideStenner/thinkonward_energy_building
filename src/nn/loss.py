import torch

from src.base.nn.loss import Metric
from sklearn.metrics import roc_auc_score


class AUCScore(Metric):
    @property
    def name(self) -> str:
        return 'auc'
        
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        score_ = roc_auc_score(
            y_true=y_true.numpy(), y_score=y_pred.numpy()
        )
        return score_
