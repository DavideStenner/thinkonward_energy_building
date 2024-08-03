import polars as pl

from typing import Union, Dict, Optional
from itertools import product, chain
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):    
    
    def __create_slice_hour_aggregation(self) -> pl.LazyFrame:
        

        all_tou_consumption = (
            self.base_data
            .with_columns(
                (
                    pl.col('timestamp').dt.hour()
                    .replace(self.slice_hour_mapping).alias('tou')
                ),
                pl.col('timestamp').dt.month().alias('month'),
                (
                    pl.col('timestamp').dt.month()
                    .replace(self.month_season_mapping).alias('season')
                ),
                pl.col('timestamp').dt.week().alias('weeknum')
            )
            .group_by(
                'bldg_id',
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('season')==season)&
                            (pl.col('tou')==tou)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_season_{season}_tou_{tou}')
                    )
                    for season, tou in product(range(3), self.tou_unique)
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('tou')==tou)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_month_{month}_tou_{tou}')
                    )
                    for month, tou in product(range(1, 12+1), self.tou_unique)
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('weeknum')==week) &
                            (pl.col('tou')==tou)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_week_{week}_tou_{tou}')
                    )
                    for week, tou in product(range(1, 53), self.tou_unique)
                ]
            )
        )
        all_tou_consumption

