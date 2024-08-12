import os
import gc
import numpy as np
import polars as pl
import pytorch_lightning as L

from torch import nn
from pytorch_lightning.loggers import CSVLogger
from typing import Tuple, Dict

from src.base.model.training import ModelTrain
from src.model.nn.initialize import TabularFFInit
from src.nn.loss import AUCScore
from src.nn.light_module import TablePredictor, TabularDataModule

class TabularFFTrainer(ModelTrain, TabularFFInit):
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
        
        if fold_ == 0:
            #save mean and std
            numerical_feature_list = [
                col for col in self.feature_list
                if col not in self.categorical_col_list
            ]
            mean_ = (
                fold_data.select(pl.col(numerical_feature_list).mean()).collect()
                .to_pandas().to_numpy('float64')
            )
            std_ = (
                fold_data.select(pl.col(numerical_feature_list).std()).collect()
                .to_pandas().to_numpy('float64')
            )

            np.save(
                os.path.join(
                    self.experiment_type_path.format(type=current_model),
                    'mean.npy'
                ),
                mean_
            )
            np.save(
                os.path.join(
                    self.experiment_type_path.format(type=current_model),
                    'std.npy'
                ),
                std_
            )


        return fold_data
    
    def train_binary(self, fold_: int) -> None:
        
        #classification metric
        params = self.params_nn['binary']
        
        train_matrix, valid_matrix = self.get_dataset(fold_=fold_, current_model='binary')
            
        self.training_logger.info('Start binary training')
        
        #update config
        params['model']['num_features'] = train_matrix[0].shape[1]
        params['model']['cat_features_idxs'] = self.categorical_features_idx
        params['model']['num_labels'] = train_matrix[1].shape[1]
        params['model']['cat_dim'] = self.cat_dims
        params['dataset']['path_experiment'] = self.experiment_type_path.format(type='binary')
        
        classifier = TablePredictor(
            config=params['model'], 
            criterion=nn.BCEWithLogitsLoss(), metric=AUCScore()
        )
        data_module = TabularDataModule(
            config=params['dataset'], 
            train_matrix=train_matrix,
            valid_matrix=valid_matrix,
            cat_features_idxs=self.categorical_features_idx,
        )
        
        loggers = CSVLogger(
            save_dir=self.experiment_path,
            name='csv_log.csv',
        )

        trainer = L.Trainer(
            logger=[loggers],
            **params['trainer']
        )

        trainer.fit(classifier, data_module)

        trainer.save_checkpoint(
            os.path.join(
                self.experiment_type_path.format(type='binary'),
                'model',
                (
                    self.model_file_name_dict['model_list']['binary']
                    .format(fold_=fold_)
                )
            )
        )

        self.model_binary_list.append(classifier)
        self.progress_binary_list.append(classifier.history[self.model_metric_used['binary']['label']])

        del train_matrix, valid_matrix
        
        _ = gc.collect()
            
    def train_commercial(self, fold_: int) -> None:
        
        #commercial metric
        params_tabnet = self.params_tabnet['commercial']
        target_mapping: Dict[str, np.ndarray] = self.target_mapper['commercial']
        target_position_list: list[list[int]] = [
            position_list for position_list in self.target_mapper['commercial'].values()
        ]

        train_matrix, test_matrix = self.get_dataset(fold_=fold_, current_model='commercial')
            
        self.training_logger.info('Start commercial training')
        model = TablePredictor(
            cat_idxs = self.categorical_features_idx,
            cat_dims=self.cat_dims,
            device_name='cpu'
            **params_tabnet,
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
            **params_tabnet,
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
            
            if 'binary' in self.model_used:
                self.train_binary(fold_=fold_)
            
            # if 'commercial' in self.model_used:
            #     self.train_commercial(fold_=fold_)
            
            # if 'residential' in self.model_used:
            #     self.train_residential(fold_=fold_)

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