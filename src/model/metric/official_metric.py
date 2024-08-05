import numpy as np
import xgboost as xgb

from typing import Dict, Tuple
from sklearn.metrics import f1_score


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
