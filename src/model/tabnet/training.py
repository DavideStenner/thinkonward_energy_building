import os
import gc
import numpy as np
import polars as pl
import xgboost as xgb

from functools import partial
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.tabnet.initialize import TabnetInit
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from src.model.metric.official_metric import xgb_eval_f1_hierarchical_macro, xgb_multi_target_softmax_obj

class TabnetTrainer(ModelTrain, TabnetInit):
    def _init_train(self) -> None:
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_PARQUET_DATA'],
                'train_binary.parquet'
            )
        )
        
        self.feature_list = [
            col for col in data.collect_schema().names()
            if col not in self.useless_col_list + [self.config_dict['TARGET_DICT']['BINARY']]
        ]
        self.categorical_col_list = [
            col for col in self.categorical_col_list
            if col not in self.useless_col_list
        ]
        self.training_logger.info(f'Using {len(self.categorical_col_list)} categorical features')

        #save feature list locally for later
        self.get_model_info()
        self.save_used_feature()
        self.save_used_categorical_feature()

    def access_fold(self, fold_: int, current_model: str) -> pl.LazyFrame:
        assert current_model in self.model_used

        fold_data = (
            pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_GOLD_PARQUET_DATA'],
                    f'train_{current_model}.parquet'
                )
            ).with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(fold_).alias('current_fold')
                )
            )
        )
        return fold_data
    
    def train_binary(self, fold_: int) -> None:
        
        #classification metric
        params_tabnet = self.params_tabnet['binary']
        
        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='binary')
            
        self.training_logger.info('Start binary training')
        model = TabNetClassifier(
            cat_idxs = self.categorical_features_idx,
            cat_dims=self.cat_dims,
            device_name='cpu',
            *params_tabnet,
        )

        model.fit(
            X_train=train_matrix[0], y_train=train_matrix[1],
            eval_set=[test_matrix], eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=10,
            patience=0,
            batch_size=256,
            virtual_batch_size=32,
            num_workers=0,
            drop_last=False,
        )


        model.save_model(
            os.path.join(
                self.experiment_type_path.format(type='binary'),
                'model',
                (
                    self.model_file_name_dict['model_list']['binary']
                    .format(fold_=fold_)
                )
            )
        )
        self.model_binary_list.append(model)
        self.progress_binary_list.append(model.history['valid_auc'])

        del train_matrix, test_matrix
        
        _ = gc.collect()
            
    def train_commercial(self, fold_: int) -> None:
        
        #commercial metric
        params_tabnet = self.params_tabnet['commercial']
        target_position_list: list[list[int]] = [
            position_list 
            for position_list in self.target_mapper['commercial'].values()
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='commercial')
        import copy
        temp_train = copy.deepcopy(train_matrix[1])
        temp_test = copy.deepcopy(test_matrix[1])
        train_matrix[1] = np.hstack(
            [
                temp_train[:, position_target].argmax(axis=1).reshape((-1, 1))
                for position_target in target_position_list
            ]
        )

        test_matrix[1] = np.hstack(
            [
                temp_test[:, position_target].argmax(axis=1).reshape((-1, 1))
                for position_target in target_position_list
            ]
        )

        self.training_logger.info('Start commercial training')
        model = TabNetMultiTaskClassifier(
            cat_idxs = self.categorical_features_idx,
            cat_dims=self.cat_dims,
            device_name='cpu', 
            *params_tabnet,
        )
        from pytorch_tabnet.metrics import Metric
        from sklearn.metrics import f1_score
        

        
        class MultiTaskF1Score(Metric):
            def __init__(self):
                self._name = "f1"
                self._maximize = True

            def __call__(self, y_true, y_score):
                score_final = f1_score(
                    y_true,
                    y_score.argmax(axis=1),
                    average='macro'
                )
                return score_final

        model.fit(
            X_train=train_matrix[0], y_train=train_matrix[1],
            eval_set=[test_matrix], eval_name=['valid'],
            eval_metric=[MultiTaskF1Score],
            max_epochs=1000,
            patience=0,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False, 
        )


        model.save_model(
            os.path.join(
                self.experiment_type_path.format(type='commercial'),
                'model',
                (
                    self.model_file_name_dict['model_list']['commercial']
                    .format(fold_=fold_)
                )
            )
        )

        self.model_commercial_list.append(model)
        self.progress_commercial_list.append(model.history['valid']['loss'])

        del train_matrix, test_matrix

        _ = gc.collect()
    
    def train_residential(self, fold_: int) -> None:
        
        #residential metric
        params_tabnet = self.params_tabnet['residential']
        target_mapping: Dict[str, np.ndarray] = self.target_mapper['residential']
        target_position_list: list[list[int]] = [
            position_list for position_list in self.target_mapper['residential'].values()
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='residential')
            
        self.training_logger.info('Start residential training')
        model = TabNetMultiTaskClassifier(
            cat_idxs = self.categorical_features_idx,
            cat_dims=self.cat_dims,
            device_name='cpu'
            *params_tabnet,
        )
        
        model.fit(
            X_train=train_matrix[0], y_train=train_matrix[1],
            eval_set=test_matrix,
            max_epochs=params_tabnet['max_epochs'],
            patience=0,
            batch_size=params_tabnet['batch_size'],
            virtual_batch_size=params_tabnet['virtual_batch_size'],
            num_workers=0,
            drop_last=False,
        )


        model.save_model(
            os.path.join(
                self.experiment_type_path.format(type='residential'),
                'model',
                (
                    self.model_file_name_dict['model_list']['residential']
                    .format(fold_=fold_)
                )
            )
        )

        self.model_residential_list.append(model)
        self.progress_residential_list.append(model.history['valid']['loss'])

        del train_matrix, test_matrix
        
        _ = gc.collect()
        
    def get_dataset(self, fold_: int, current_model: str) -> Tuple[Tuple[np.ndarray]]:
        fold_data = self.access_fold(fold_=fold_, current_model=current_model)
        
        target_info: list[str] | str = self.config_dict['TARGET_DICT'][current_model.upper()]
        if isinstance(target_info, list):
            target_list = [
                col for col in fold_data.collect_schema().names()
                if any(
                    [
                        target_name in col
                        for target_name in target_info
                    ]
                )
            ]
        else:
            target_list = [target_info]
            
        train_filtered = fold_data.filter(
            (pl.col('current_fold') == 't')
        )
        test_filtered = fold_data.filter(
            (pl.col('current_fold') == 'v')
        )
        
        assert len(
            set(
                train_filtered.select(self.build_id).unique().collect().to_series().to_list()
            ).intersection(
                test_filtered.select(self.build_id).unique().collect().to_series().to_list()
            )
        ) == 0
        
        train_rows = train_filtered.select(pl.count()).collect().item()
        test_rows = test_filtered.select(pl.count()).collect().item()
        
        self.training_logger.info(f'{train_rows} train rows; {test_rows} test rows; {len(self.feature_list)} feature; {len(target_list)} target')
        
        train_matrix = [
            train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            train_filtered.select(target_list).collect().to_pandas().to_numpy('float64'),
        ]
        test_matrix = [
            test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            test_filtered.select(target_list).collect().to_pandas().to_numpy('float64'),
        ]
        return train_matrix, test_matrix
        
    def train(self) -> None:
        
        self._init_train()
        
        for fold_ in range(self.n_fold):
            self.training_logger.info(f'\n\nStarting fold {fold_}\n\n\n')
            self.training_logger.info('Collecting dataset')
            
            # if 'binary' in self.model_used:
            #     self.train_binary(fold_=fold_)
            
            if 'commercial' in self.model_used:
                self.train_commercial(fold_=fold_)
            
            if 'residential' in self.model_used:
                self.train_residential(fold_=fold_)

    def save_model(self)->None:
        for type_model in self.model_used:
            
            self.save_pickle_model_list(
                getattr(
                    self, f'model_{type_model}_list'
                ), 
                type_model,
            )
            self.save_progress_list(
                getattr(
                    self, f'progress_{type_model}_list'
                ), 
                type_model
            )

        self.save_params()