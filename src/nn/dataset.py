import os
import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(
            self, 
            feature: torch.Tensor,
            target: torch.Tensor,
            cat_features_idxs: Optional[list[int]],
            config: Dict[str, Any],
            inference: bool,
        ):

        self.num_row = len(feature)
        self.num_col = feature.shape[1]
        self.config = config
        self.inference = inference
        
        if not self.inference:
            self.labels: torch.Tensor = target
        
        self.cat_features_idxs: list[int] = (
            cat_features_idxs 
            if cat_features_idxs
            else []
        )
        self.__init_categorical_utils(feature)
        
    def __init_categorical_utils(self, feature: torch.Tensor) -> None:
        
        self.has_categorical: bool = len(self.cat_features_idxs) > 0
        self.has_numerical: bool = (self.num_col>len(self.cat_features_idxs))
    
        if self.has_numerical:
            mean_ = np.load(
                os.path.join(
                    self.config['path_experiment'],
                    'mean.npy'
                )
            )
            std_ = np.load(
                os.path.join(
                    self.config['path_experiment'],
                    'std.npy'
                ),
            )
            #rearrange numerical features first
            self.numerical_features: torch.Tensor = torch.tensor(
                (
                    np.concatenate(
                        [
                            feature[:, feature_idx].reshape((-1, 1))
                            for feature_idx in range(self.num_col)
                            if feature_idx not in self.cat_features_idxs
                        ], axis=1,
                    ) - mean_
                )/std_, dtype=torch.float
            )

        if self.has_categorical:
            self.categorical_features: torch.Tensor = torch.tensor(
                np.concatenate(
                    [
                        feature[:, feature_idx].reshape((-1, 1))
                        for feature_idx in self.cat_features_idxs
                    ], axis=1,
                ), dtype=torch.long
            )
        assert self.has_categorical | self.has_numerical
        
    def __get_categorical_numerical_feature(self, item: int) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        if self.has_categorical & self.has_numerical:
            return [self.numerical_features[item], self.categorical_features[item]]
        
        elif self.has_categorical:
            return self.categorical_features[item]
        
        elif self.has_numerical:
            return self.numerical_features[item]
        
    def __len__(self):
        return self.num_row

    def __getitem__(self, item):
        feature_ = self.__get_categorical_numerical_feature(item)
        
        if self.inference:
            return feature_
        else:
            return feature_, self.labels[item]
    