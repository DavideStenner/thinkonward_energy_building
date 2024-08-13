import os
import gc
import numpy as np
import polars as pl
import lightgbm as lgb

from functools import partial
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.lgbm.initialize import LgbmInit
 
class LgbmTrainer(ModelTrain, LgbmInit):
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
        params_lgb = self.params_lgb['binary']
        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(
                period=50, 
                show_stdv=False
            )
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='binary')

        self.training_logger.info('Start binary training')
        model = lgb.train(
            params=params_lgb,
            train_set=train_matrix, 
            num_boost_round=params_lgb['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
        )

        model.save_model(
            os.path.join(
                self.experiment_path,
                'binary',
                (
                    self.model_file_name_dict['model_list']['binary']
                    .format(fold_=fold_)
                )
            ), importance_type='gain'
        )

        self.model_binary_list.append(model)
        self.progress_binary_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()

    def train_commercial(self, fold_: int) -> None:
        #classification metric
        params_lgb = self.params_lgb['train_commercial']
        progress = {}

        callbacks_list = [
            lgb.record_evaluation(progress),
            lgb.log_evaluation(
                period=50, 
                show_stdv=False
            )
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='train_commercial')

        self.training_logger.info('Start commercial training')
        model = lgb.train(
            params=params_lgb,
            train_set=train_matrix, 
            num_boost_round=params_lgb['n_round'],
            valid_sets=[test_matrix],
            valid_names=['valid'],
            callbacks=callbacks_list,
        )

        model.save_model(
            os.path.join(
                self.experiment_path,
                'commercial',
                (
                    self.model_file_name_dict['model_list']['commercial']
                    .format(fold_=fold_)
                )
            ), importance_type='gain'
        )

        self.model_commercial_list.append(model)
        self.progress_commercial_list.append(progress)

        del train_matrix, test_matrix
        
        _ = gc.collect()



    def get_dataset(self, fold_: int, current_model: str) -> Tuple[lgb.Dataset]:
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
        
        train_matrix = lgb.Dataset(
            train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            train_filtered.select(target_list).collect().to_pandas().to_numpy('float64'),
            feature_name=self.feature_list, categorical_feature=self.categorical_col_list
        )
        test_matrix = lgb.Dataset(
            test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float64'),
            test_filtered.select(target_list).collect().to_pandas().to_numpy('float64'),
            feature_name=self.feature_list, categorical_feature=self.categorical_col_list
        )
        return train_matrix, test_matrix

    def train(self) -> None:
        
        self._init_train()
        
        for fold_ in range(self.n_fold):
            self.training_logger.info(f'\n\nStarting fold {fold_}\n\n\n')
            self.training_logger.info('Collecting dataset')
            
            if 'binary' in self.model_used:
                self.train_binary(fold_=fold_)
            
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