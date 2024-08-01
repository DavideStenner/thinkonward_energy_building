import os
import polars as pl

from src.base.preprocess.import_data import BaseImport
from src.preprocess.initialize import PreprocessInit

class PreprocessImport(BaseImport, PreprocessInit):
    def scan_all_dataset(self):
        stage_file: str = (
            'test' if self.inference
            else 'train'
        )
        self.base_data: pl.LazyFrame = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_ORIGINAL_DATA'],
                f'{stage_file}_data.parquet'
            )
        )
        if not self.inference:
            self.label_data: pl.LazyFrame = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_ORIGINAL_DATA'],
                f'train_data_label.parquet'
            )
        )
        
    def downcast_data(self):
        self.base_data = (
            self.base_data.with_columns(
                pl.col('timestamp').cast(pl.Datetime),
                pl.col('out.electricity.total.energy_consumption').cast(pl.Float64),
                pl.col('in.state').cast(pl.UInt8),
                pl.col(self.build_id).cast(pl.UInt16),
            )
            .rename(
                {
                    'out.electricity.total.energy_consumption': 'energy_consumption',
                    'in.state': 'state'
                }
            )
        )
        if not self.inference:
            self.label_data = self.label_data.with_columns(
                [
                    pl.col(self.build_id).cast(pl.UInt16)
                ] +
                [
                    pl.col(col).cast(pl.UInt8)
                    for col in self.all_target_list
                ]
            )
            
    def import_all(self) -> None:
        self.scan_all_dataset()
        self.downcast_data()