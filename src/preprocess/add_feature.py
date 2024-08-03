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

    def __create_hour_aggregation(self) -> pl.LazyFrame:
        all_hour_consumption = (
            self.base_data
            .with_columns(
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
                        .filter(pl.col('season')==season)
                        .mean()
                        .alias(f'average_hour_consumption_season_{season}')
                    )
                    for season in range(3)
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('month')==month)
                        .mean()
                        .alias(f'average_hour_consumption_month_{month}')
                    )
                    for month in range(1, 12+1)
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('weeknum')==week)
                        .mean()
                        .alias(f'average_hour_consumption_week_{week}')
                    )
                    for week in range(1, 53)
                ]
            )
        )
        return all_hour_consumption

    def __create_daily_aggregation(self) -> pl.LazyFrame:
        all_daily_consumption = (
            self.base_data
            .group_by(
                'bldg_id',
                pl.col('timestamp').dt.truncate('1d')
            )
            .agg(
                pl.col('energy_consumption').sum().alias('daily_consumption')
            )
            .with_columns(
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
                        pl.col('daily_consumption')
                        .filter(pl.col('season')==season)
                        .mean()
                        .alias(f'average_daily_consumption_season_{season}')
                    )
                    for season in range(3)
                ] +
                [
                    (
                        pl.col('daily_consumption')
                        .filter(pl.col('month')==month)
                        .mean()
                        .alias(f'average_daily_consumption_month_{month}')
                    )
                    for month in range(1, 12+1)
                ] +
                [
                    (
                        pl.col('daily_consumption')
                        .filter(pl.col('weeknum')==week)
                        .mean()
                        .alias(f'average_daily_consumption_week_{week}')
                    )
                    for week in range(1, 53)
                ]
            )
        )
        return all_daily_consumption
    
        
    def create_feature(self) -> None:   
        self.lazy_feature_list.append(
            self.__create_daily_aggregation()
        )
        self.lazy_feature_list.append(
            self.__create_hour_aggregation()
        )
        self.lazy_feature_list.append(
            self.__create_slice_hour_aggregation()
        )
            
    def merge_all(self) -> None:
        self.data = self.base_data.select(self.build_id, 'state').unique()

        for lazy_feature_dataframe in self.lazy_feature_list:
            self.data = (
                self.data
                .join(
                    lazy_feature_dataframe, 
                    on=self.build_id, how='left'
                )
            )

