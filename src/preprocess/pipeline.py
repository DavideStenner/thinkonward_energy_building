import os
import gc

import polars as pl

from typing import Any, Tuple

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

    def save_data(self) -> None:
        self.preprocess_logger.info('saving processed dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_GOLD_PARQUET_DATA'],
                'train_data.parquet'
            )
        )
        for name_file, lazy_frame in self.dict_target.items():
            self.preprocess_logger.info(f'saving {name_file}')
            lazy_frame.collect().write_parquet(
                os.path.join(
                    self.config_dict['PATH_GOLD_PARQUET_DATA'],
                    f'{name_file}.parquet'
                )
            )
            
    def collect_feature(self) -> None:
        self.base_data: pl.DataFrame = self.base_data.collect()
        
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
        print('Creating feature')
        self.create_feature()

        print('Merging All')
        self.merge_all()
                    
        print('Collecting test....')
        self.data: pl.DataFrame = self.data.collect()
        _ = gc.collect()

    def preprocess_train(self) -> None:
        self._initialize_preprocess_logger()
        
        self.preprocess_logger.info('Creating feature')
        self.create_feature()

        self.preprocess_logger.info('Merging all')
        self.merge_all()
        
        self.preprocess_logger.info(
            f'Collecting dataset with {len(self._get_col_name(self.data))} columns'
        )
        self.data: pl.DataFrame = self.data.collect()
        
        _ = gc.collect()
        
        self.preprocess_logger.info('Creating fold_info column ...')
        self.create_fold()
        
        self.save_data()
                
    def begin_training(self) -> None:
        self.import_all()
        
    def begin_inference(self) -> None:
        print('Beginning inference')
        
        #reset data
        self.data = None
        self.inference: bool = True
        
        self.import_all()

    def __call__(self, subset_feature: list[str] = None) -> None:
        if self.inference:
            self.preprocess_inference(subset_feature=subset_feature)

        else:
            self.import_all()
            self.preprocess_train()
