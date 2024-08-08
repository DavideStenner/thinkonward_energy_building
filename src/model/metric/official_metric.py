import numpy as np
import xgboost as xgb

from typing import Dict, Tuple
from scipy.special import softmax
from sklearn.metrics import f1_score


def softmax_by_target(
        target_position_list: list[int], y_pred: np.ndarray
    ) -> np.ndarray:
    """Compute the softmax for multiple target in isolation"""
    for target_position in target_position_list:
        y_pred[:, target_position] = softmax(y_pred[:, target_position], axis=1)
    
    return y_pred

def xgb_multi_target_softmax_obj(
        target_position_list: list[list[int]], 
        y_pred: np.ndarray, eval_data: xgb.DMatrix
    ):
    '''Loss function. Computing the gradient and upper bound on the
    Hessian with a diagonal structure for XGBoost (note that this is
    not the true Hessian).
    Reimplements the `multi:softprob` inside XGBoost.
    https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_softmax.py
    https://stats.stackexchange.com/questions/448821/hessian-matrix-for-multiclass-softmax-in-gradient-boosting-xgboost-2p-i1-p
    '''
    labels = eval_data.get_label().reshape(y_pred.shape)

    eps = 1e-6

    #softmax by each target
    pred_prob_by_target = softmax_by_target(
        target_position_list=target_position_list, y_pred=y_pred
    )
    #for each column: p - 1 if target == 1 otherwise p
    #max(2 * p * (1 - p), eps)
    #hessian 
    
    gradient = pred_prob_by_target - labels
    hessian = np.clip(2 * pred_prob_by_target * (1 - pred_prob_by_target), a_min=eps, a_max=1-eps)
    return gradient, hessian

def xgb_eval_f1_hierarchical_macro(
        target_mapping: Dict[str, np.ndarray],
        y_pred: np.ndarray, eval_data: xgb.DMatrix,
    ) -> Tuple[str, float]:
    
    y_true = eval_data.get_label().reshape(y_pred.shape)
    f1_score_list = [
        f1_score(
            y_true=y_true[:, position_target].argmax(axis=1),
            y_pred=y_pred[:, position_target].argmax(axis=1),
            average='macro'
        )
        for _, position_target in target_mapping.items()
    ]

    score_final = np.mean(f1_score_list)
    return 'f1', score_final

def xgb_eval_f1_single_macro(
        position_target: np.ndarray, name_target:str,
        y_pred: np.ndarray, eval_data: xgb.DMatrix,
    ) -> Tuple[str, float]:
    
    y_true = eval_data.get_label().reshape(y_pred.shape)
    result = f1_score(
        y_true=y_true[:, position_target].argmax(axis=1),
        y_pred=y_pred[:, position_target].argmax(axis=1),
        average='macro'
    )

    return name_target, result
