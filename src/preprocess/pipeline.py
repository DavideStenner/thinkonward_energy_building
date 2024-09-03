import os
import gc
import json

import numpy as np
import polars as pl

from typing import Any, Tuple, Dict

from src.base.preprocess.pipeline import BasePipeline
from src.preprocess.import_data import PreprocessImport
from src.preprocess.initialize import PreprocessInit
from src.preprocess.add_feature import PreprocessAddFeature
from src.preprocess.cv_fold import PreprocessFoldCreator

class PreprocessPipeline(BasePipeline, PreprocessImport, PreprocessAddFeature, PreprocessFoldCreator):

    def __init__(self, config_dict: dict[str, Any]):
                
        PreprocessInit.__init__(
            self, 
            config_dict=config_dict, 
        )

    def __get_dummy_target_mapping(
            self, 
            original_target_list: list[str], dummy_column_names: list[str]
        ) -> Dict[str, np.ndarray]:
                
        #correct order of dummy target based on target list
        target_dummy_list = [
            col for col in dummy_column_names
            if any(
                [
                    target_name in col
                    for target_name in original_target_list
                ]
            )
        ]
        #useful mapper
        target_dict = {
            target_base: np.where(
                [
                    target_base in col
                    for col in target_dummy_list
                ]
            )[0].tolist()
            for target_base in original_target_list
        }

        return target_dict

    def save_data(self) -> None:       
        self.preprocess_logger.info('saving every processed dataset + target')
        mapper_dummy_target = {}
        
        #for each file join back to dataset and also save mapping of dummy to original feature
        for name_file, lazy_frame in self.dict_target.items():

            #make dummy
            if name_file != 'train_binary':
                current_target: str = (
                    name_file
                    .split('train_')[1]
                    .strip().upper()
                )
                dataset_label = lazy_frame.collect()
                dataset_label_dummy = (
                    dataset_label
                    .to_dummies(self.config_dict['TARGET_DICT'][current_target])
                )
                target_dict = self.__get_dummy_target_mapping(
                    self.config_dict['TARGET_DICT'][current_target],
                    dataset_label_dummy.collect_schema().names()
                )
                mapper_dummy_target[current_target.lower()] = target_dict
                
            else:
                dataset_label = lazy_frame.collect()
                dataset_label_dummy = dataset_label
                
            self.preprocess_logger.info(f'saving {name_file}')
            (
                dataset_label
                .join(
                    self.data, 
                    on=self.build_id, how='inner'
                )
                .write_parquet(
                os.path.join(
                        self.config_dict['PATH_GOLD_PARQUET_DATA'],
                        f'{name_file}_label.parquet'
                    )
                )
            )
            self.preprocess_logger.info(f'saving {name_file} dummy')
            (
                dataset_label_dummy
                .join(
                    self.data, 
                    on=self.build_id, how='inner'
                )
                .write_parquet(
                os.path.join(
                        self.config_dict['PATH_GOLD_PARQUET_DATA'],
                        f'{name_file}.parquet'
                    )
                )
            )
        self.preprocess_logger.info(f'saving target mapper')
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'target_mapper.json'
            ), 'w'
        ) as file_json:
            json.dump(mapper_dummy_target, file_json)
            
    def collect_feature(self) -> None:
        self.base_data: pl.DataFrame = self.base_data.collect()
        self.minute_data: pl.DataFrame = self.minute_data.collect()
        
    def collect_all(self) -> None:
        self.collect_feature()
        
    @property
    def feature_list(self) -> Tuple[str]:
        self.import_all()
        self.create_feature()

        self.merge_all()

        data_columns = self._get_col_name(self.data)
        
        #reset dataset
        self.import_all()
        
        return data_columns
        
    def preprocess_inference(self) -> None:
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()
                    
        self.preprocess_logger.info(
            f'Collecting dataset with {len(self._get_col_name(self.data))} columns and {self._get_number_rows(self.data)} rows'
        )
        self.data: pl.DataFrame = self.data.collect()

        self.preprocess_logger.info('Saving test dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_PARQUET_DATA'],
                f'test_data.parquet'
            )
        )
        _ = gc.collect()

    def preprocess_train(self) -> None:
        self.preprocess_logger.info('beginning preprocessing training dataset')
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()
        
        self.preprocess_logger.info(
            f'Collecting dataset with {len(self._get_col_name(self.data))} columns and {self._get_number_rows(self.data)} rows'
        )
        self.data: pl.DataFrame = self.data.collect()
        
        _ = gc.collect()
        
        self.preprocess_logger.info('Creating fold_info column ...')
        self.create_fold()
        
        self.preprocess_logger.info('Saving multiple training dataset')
        self.save_data()
                
    def begin_training(self) -> None:
        self.import_all()
        
    def begin_inference(self) -> None:
        self.preprocess_logger.info('beginning preprocessing inference dataset')
        
        #reset data
        self.data = None
        self.inference: bool = True
        
        self.import_all()

    def __call__(self) -> None:
        if self.inference:
            self.preprocess_inference()

        else:
            self.import_all()
            self.preprocess_train()
