import os
import gc
import numpy as np
import polars as pl
import xgboost as xgb

from functools import partial
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.xgbm.initialize import XgbInit
from src.model.metric.official_metric import xgb_eval_f1_hierarchical_macro, xgb_multi_target_softmax_obj

class XgbTrainer(ModelTrain, XgbInit):
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
        params_xgb = self.params_xgb['binary']
        
        progress = {}

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='binary')

        progress = {}
            
        self.training_logger.info('Start binary training')
        model = xgb.train(
            params=params_xgb,
            dtrain=train_matrix, 
            num_boost_round=params_xgb['num_boost_round'],
            evals=[(test_matrix, 'valid')],
            evals_result=progress, verbose_eval=50
        )
        model.save_model(
            os.path.join(
                self.experiment_path,
                'binary',
                (
                    self.model_file_name_dict['model_list']['binary']
                    .format(fold_=fold_)
                )
            )
        )

        self.model_binary_list.append(model)
        self.progress_binary_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
            
    def train_commercial(self, fold_: int) -> None:
        
        #commercial metric
        params_xgb = self.params_xgb['commercial']
        target_mapping: Dict[str, np.ndarray] = self.target_mapper['commercial']
        target_position_list: list[list[int]] = [
            position_list for position_list in self.target_mapper['commercial'].values()
        ]
        progress = {}

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='commercial')

        progress = {}
            
        self.training_logger.info('Start commercial training')
        model = xgb.train(
            params=params_xgb,
            dtrain=train_matrix, 
            num_boost_round=params_xgb['num_boost_round'],
            evals=[(test_matrix, 'valid')],
            evals_result=progress, verbose_eval=10, 
            obj=partial(xgb_multi_target_softmax_obj, target_position_list),
            custom_metric=partial(xgb_eval_f1_hierarchical_macro, target_mapping)
        )
        model.save_model(
            os.path.join(
                self.experiment_path,
                'commercial',
                (
                    self.model_file_name_dict['model_list']['commercial']
                    .format(fold_=fold_)
                )
            )
        )

        self.model_commercial_list.append(model)
        self.progress_commercial_list.append(progress)

        del train_matrix, test_matrix

        _ = gc.collect()
    
    def train_residential(self, fold_: int) -> None:
        
        #residential metric
        params_xgb = self.params_xgb['residential']
        target_mapping: Dict[str, np.ndarray] = self.target_mapper['residential']
        target_position_list: list[list[int]] = [
            position_list for position_list in self.target_mapper['residential'].values()
        ]

        progress = {}

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='residential')

        progress = {}
            
        self.training_logger.info('Start residential training')
        model = xgb.train(
            params=params_xgb,
            dtrain=train_matrix, 
            num_boost_round=params_xgb['num_boost_round'],
            evals=[(test_matrix, 'valid')],
            evals_result=progress, verbose_eval=10,
            obj=partial(xgb_multi_target_softmax_obj, target_position_list),
            custom_metric=partial(xgb_eval_f1_hierarchical_macro, target_mapping)
        )
        model.save_model(
            os.path.join(
                self.experiment_path,
                'residential',
                (
                    self.model_file_name_dict['model_list']['residential']
                    .format(fold_=fold_)
                )
            )
        )

        self.model_residential_list.append(model)
        self.progress_residential_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()
        
    def get_dataset(self, fold_: int, current_model: str) -> Tuple[xgb.DMatrix]:
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
        
        train_matrix = xgb.DMatrix(
            train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            train_filtered.select(target_list).collect().to_pandas().to_numpy('float64'),
            feature_names=self.feature_list, enable_categorical=True, feature_types=self.feature_types_list,
            #mean prob as base margin
            base_margin=np.tile(
                train_filtered.select(pl.col(target_list).mean())
                .collect().to_pandas()
                .to_numpy('float64'),
                (train_filtered.select(pl.len()).collect().item(), 1)
            ),
        )
        test_matrix = xgb.DMatrix(
            test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            test_filtered.select(target_list).collect().to_pandas().to_numpy('float64'),
            feature_names=self.feature_list, enable_categorical=True, feature_types=self.feature_types_list,
        )
        return train_matrix, test_matrix
        
    def train(self) -> None:
        
        self._init_train()
        
        for fold_ in range(self.n_fold):
            self.training_logger.info(f'\n\nStarting fold {fold_}\n\n\n')
            self.training_logger.info('Collecting dataset')
            
            if 'binary' in self.model_used:
                self.train_binary(fold_=fold_)
            
            if 'residential' in self.model_used:
                self.train_residential(fold_=fold_)

            if 'commercial' in self.model_used:
                self.train_commercial(fold_=fold_)
            
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