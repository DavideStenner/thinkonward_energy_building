import os
import json
import pickle
import logging

import numpy as np
import polars as pl

from typing import Any, Union, Dict, Tuple
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from src.base.model.initialize import ModelInit
from src.utils.logging_utils import get_logger


class TabnetInit(ModelInit):
    def __init__(self, 
            experiment_name:str, 
            params_tabnet: dict[str, Any],
            config_dict: dict[str, Any], data_columns: Tuple[str],
            fold_name: str = 'fold_info'
        ):
        
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
        
        self.model_used: list[str] = ['binary', 'commercial', 'residential']
        self.model_metric_used: list[str] = {
            'binary': {'label': 'auc', 'maximize': True},
            'commercial': {'label': 'f1', 'maximize': True},
            'residential': {'label': 'f1', 'maximize': True},
        }
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name,
        )
        self.experiment_type_path: str = os.path.join(
            self.experiment_path, '{type}'
        )
        
        self.experiment_path_dict: dict[str, str] = {
            'feature_importance': os.path.join(
                self.experiment_type_path, 'insight'
            ),
            'insight': os.path.join(
                self.experiment_type_path, 'insight'
            ),
            'training': os.path.join(
                self.experiment_type_path, 'training'
            ),
            'shap': os.path.join(
                self.experiment_type_path, 'shap'
            ),
            'model': os.path.join(
                self.experiment_type_path, 'model'
            )
        }

        self.n_fold: int = config_dict['N_FOLD']
        
        self.fold_name: str = fold_name

        self.special_column_list: list[str] = config_dict['SPECIAL_COLUMNS']

        self.useless_col_list: list[str] = (
            self.special_column_list +
            [
                'fold_info', 'current_fold'
            ]
        )

        self.params_tabnet: dict[str, Any] = params_tabnet
        
        self.feature_list: list[str] = []
        
        self.get_categorical_columns(data_columns=data_columns)
        self.initialize_model_utils()
        self.initialize_target_utils()
        self.get_model_file_name_dict()
        self.get_target_mapper()
        self.get_target_col_list()
        
    def get_target_col_list(self) -> None:
        self.target_dict: Dict[str, list[str]] = {}
        
        for current_model in self.model_used:
            target_info: list[str] | str = self.config_dict['TARGET_DICT'][current_model.upper()]
            
            current_model_columns_list: pl.LazyFrame = (
                pl.scan_parquet(
                    os.path.join(
                        self.config_dict['PATH_GOLD_PARQUET_DATA'],
                        f'train_{current_model}.parquet'
                    )
                )
                .collect_schema().names()
            )
            if isinstance(target_info, list):
                target_list = [
                    col for col in current_model_columns_list
                    if any(
                        [
                            target_name in col
                            for target_name in target_info
                        ]
                    )
                ]
            else:
                target_list = [target_info]
        
            self.target_dict[current_model] = target_list
    
    def get_target_mapper(self) -> None:
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'target_mapper.json'
            ), 'r'
        ) as file_json:
            self.mapper_dummy_target: Dict[str, Dict[str, list[int]]] = json.load(file_json)
            
    def get_model_info(self) -> None:
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'mapper_category.json'
            ), 'r'
        ) as file_json:
            categorical_mapper: Dict[str, Dict[str, list[int]]] = json.load(file_json)['train_data']
        
        self.cat_dims: list[int] = [
            len(dict_mapper.keys())
            for _, dict_mapper in categorical_mapper.items()
        ]
        self.categorical_features_idx: list[str] = np.where(
            [
                col in self.categorical_col_list
                for col in self.feature_list
            ]
        )[0].tolist()

    def initialize_logger(self) -> None:
        self.training_logger: logging.Logger = get_logger(
            file_name='training_tabnet.txt', path_logger=self.experiment_path
        )

    def initialize_target_utils(self) -> None:
        #target mapper for residential/commercial
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'target_mapper.json'
            ), 'r'
        ) as file_json:
            self.target_mapper: Dict[str, list[int]] = json.load(file_json)
        
        
    def initialize_model_utils(self) -> None:
        self.build_id: str = 'bldg_id'
        
        for type_model in self.model_used:
            setattr(
                self, f'model_{type_model}_list', [] 
            )
            
            setattr(
                self, f'progress_{type_model}_list', [] 
            )

                    
    def get_categorical_columns(self, data_columns: Tuple[str]) -> None:
        #load all possible categorical feature
        cat_col_list = [
            "state"
        ]
        self.categorical_col_list: list[str] = list(
            set(cat_col_list)
            .intersection(set(data_columns))
        )

    def create_experiment_structure(self) -> None:
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
            
        for model_type in self.model_used:
            if not os.path.isdir(self.experiment_type_path.format(type=model_type)):
                os.makedirs(self.experiment_type_path.format(type=model_type))
            
            for dir_path_format in self.experiment_path_dict.values():
                dir_path: str = dir_path_format.format(type=model_type)
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

    def load_model(self) -> None: 
        self.load_used_feature()
        self.load_used_categorical_feature()
        self.load_best_result()
        self.load_params()
        
        self.load_model_list()
    
    def get_model_file_name_dict(self) -> None:
        self.model_file_name_dict: dict[str, str] =  {
            'progress_list': {
                type_model: f'progress_{type_model}_list.pkl'
                for type_model in self.model_used
            },
            'best_result': {
                type_model: f'best_result_{type_model}_tabnet.txt'
                for type_model in self.model_used
            },
            'model_pickle_list': {
                type_model: f'model_{type_model}_list_tabnet.pkl'
                for type_model in self.model_used
            },
            'model_list': {
                type_model: f'tabnet_{type_model}' + '_{fold_}'
                for type_model in self.model_used
            }
        }
    
    def save_progress_list(self, progress_list: list, type_model: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['progress_list'][type_model]
            ), 'wb'
        ) as file:
            pickle.dump(progress_list, file)

    def load_progress_list(self, type_model: str) -> list:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['progress_list'][type_model]
            ), 'rb'
        ) as file:
            progress_list = pickle.load(file)
            
        return progress_list
    def save_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_tabnet.json'
            ), 'w'
        ) as file:
            json.dump(self.params_tabnet, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_tabnet.json'
            ), 'r'
        ) as file:
            self.params_tabnet = json.load(file)
    
    def save_best_result(self, best_result: dict[str, Union[int, float]], type_model: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['best_result'][type_model]
            ), 'w'
        ) as file:
            json.dump(best_result, file)
        
    def load_best_result(self, type_model: str) -> dict[str, Union[int, float]]:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                self.model_file_name_dict['best_result'][type_model]
            ), 'r'
        ) as file:
            best_result = json.load(file)
            
        return best_result

    def save_pickle_model_list(self, model_list: list[TabNetClassifier | TabNetMultiTaskClassifier], type_model: str) -> None:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                'model',
                self.model_file_name_dict['model_pickle_list'][type_model]
            ), 'wb'
        ) as file:
            pickle.dump(model_list, file)
    
    def load_pickle_model_list(self, type_model: str) -> list[TabNetClassifier | TabNetMultiTaskClassifier]:
        with open(
            os.path.join(
                self.experiment_type_path.format(type=type_model),
                'model',
                self.model_file_name_dict['model_pickle_list'][type_model]
            ), 'rb'
        ) as file:
            model_list = pickle.load(file)
    
        return model_list

    def load_model_list(self, type_model: str, file_name: str) -> list[TabNetClassifier | TabNetMultiTaskClassifier]:
        
        tabnet_class: TabNetClassifier | TabNetMultiTaskClassifier = (
            TabNetClassifier if type_model == 'binary'
            else TabNetMultiTaskClassifier
        )
        
            
        return [
            tabnet_class().load_model(
                os.path.join(
                    self.experiment_type_path.format(type=type_model),
                    'model',
                    file_name.format(fold_=fold_)
                )
            )
            for fold_ in range(self.n_fold)
        ]    
            
    def save_used_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'feature_model': self.feature_list
                }, 
                file
            )
    
    def load_used_categorical_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_categorical_feature.txt'
            ), 'r'
        ) as file:
            self.categorical_col_list = json.load(file)['categorical_feature']
            
    def save_used_categorical_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_categorical_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'categorical_feature': self.categorical_col_list
                }, 
                file
            )

    def load_used_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'r'
        ) as file:
            self.feature_list = json.load(file)['feature_model']