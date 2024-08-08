import polars as pl

from typing import Union, Dict, Optional
from itertools import product, chain
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):    
    
    def __create_slice_hour_aggregation(self) -> pl.LazyFrame:
        """
            Create average hour consumption over
                - season, tou
                - month, tou
                - weeknum, tou

        Returns:
            pl.LazyFrame: query
        """

        all_tou_consumption = (
            self.base_data
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
                    for season, tou in product(self.season_list, self.tou_unique)
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
                    for month, tou in product(self.month_list, self.tou_unique)
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
                    for week, tou in product(self.weeknum_list, self.tou_unique)
                ]
            )
        )
        return all_tou_consumption

    def create_slice_day_aggregation(self) -> pl.LazyFrame:
        """
        Create average daily consumption over
            - season, is_weekend
            - month, is_weekend
            - weeknum, is_weekend

        Returns:
            pl.LazyFrame: query
        """
        all_day_consumption = (
            self.base_data
            .group_by(
                'bldg_id',
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('season')==season) &
                            (pl.col('is_weekend')==is_weekend)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_season_{season}_is_weekend_{is_weekend}')
                    )
                    for season, is_weekend in product(self.season_list, [0, 1])
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('is_weekend')==is_weekend)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_month_{month}_is_weekend_{is_weekend}')
                    )
                    for month, is_weekend in product(self.month_list, [0, 1])
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('weeknum')==week) &
                            (pl.col('is_weekend')==is_weekend)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_week_{week}_is_weekend_{is_weekend}')
                    )
                    for week, is_weekend in product(self.weeknum_list, [0, 1])
                ]
            )
        )
        return all_day_consumption
    
    def __create_hour_weeknum_aggregation(self) -> pl.LazyFrame:
        """
        Create average daily consumption over
            - season, weekday
            - month, weekday

        Returns:
            pl.LazyFrame: query
        """
        all_day_consumption = (
            self.base_data
            .group_by(
                'bldg_id',
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('season')==season) &
                            (pl.col('weekday')==weekday)
                        )
                        .mean()
                        .alias(f'average_daily_consumption_season_{season}_weekday_{weekday}')
                    )
                    for season, weekday in product(self.season_list, self.weekday_list)
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('weekday')==weekday)
                        )
                        .mean()
                        .alias(f'average_daily_consumption_month_{month}_weekday_{weekday}')
                    )
                    for month, weekday in product(self.month_list, self.weekday_list)
                ]
            )
        )
        return all_day_consumption
    
    def __create_hour_aggregation(self) -> pl.LazyFrame:
        """
        Create hour average consumption over:
        - season
        - month
        - week

        Returns:
            pl.LazyFrame: query
        """
        all_hour_consumption = (
            self.base_data
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
                    for season in self.season_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('month')==month)
                        .mean()
                        .alias(f'average_hour_consumption_month_{month}')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('weeknum')==week)
                        .mean()
                        .alias(f'average_hour_consumption_week_{week}')
                    )
                    for week in self.weeknum_list
                ]
            )
        )
        return all_hour_consumption

    def __create_daily_aggregation(self) -> pl.LazyFrame:
        """
        Create daily average consumption over:
        - season
        - month
        - week

        Returns:
            pl.LazyFrame: query
        """
        all_daily_consumption = (
            self.base_data
            .group_by(
                'bldg_id', 'season', 'month', 'weeknum', 'day'
            )
            .agg(
                pl.col('energy_consumption').sum().alias('daily_consumption')
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
                    for season in self.season_list
                ] +
                [
                    (
                        pl.col('daily_consumption')
                        .filter(pl.col('month')==month)
                        .mean()
                        .alias(f'average_daily_consumption_month_{month}')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('daily_consumption')
                        .filter(pl.col('weeknum')==week)
                        .mean()
                        .alias(f'average_daily_consumption_week_{week}')
                    )
                    for week in self.weeknum_list
                ]
            )
        )
        return all_daily_consumption
    
    def __create_total_average_consumptions(self) -> pl.LazyFrame:
        """
        Total average consumption over
        - weekday
        - hour

        Returns:
            pl.LazyFrame: _description_
        """
        total_average_consumptions = (
            self.base_data
            .group_by(
                'bldg_id',
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('weekday')==weekday)
                        .mean()
                        .alias(f'total_average_consumption_weekday_{weekday}')
                    )
                    for weekday in self.weekday_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('hour')==hour)
                        .mean()
                        .alias(f'total_average_consumption_hour_{hour}')
                    )
                    for hour in self.hour_list
                ]
            )
        )
        return total_average_consumptions

    def __create_total_consumptions(self) -> pl.LazyFrame:
        """
        Total consumption over
        - season
        - month
        - weeknum

        Returns:
            pl.LazyFrame: query
        """
        total_consumptions = (
            self.base_data
            .group_by(
                'bldg_id',
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('season')==season)
                        .sum()
                        .alias(f'total_consumption_season_{season}')
                    )
                    for season in self.season_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('month')==month)
                        .sum()
                        .alias(f'total_consumption_month_{month}')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('weeknum')==weeknum)
                        .sum()
                        .alias(f'total_consumption_week_{weeknum}')
                    )
                    for weeknum in self.weeknum_list
                ] +
                [
                    pl.col('energy_consumption').sum().alias('total_consumption_ever')
                ]
            )
        )
        return total_consumptions
    
    def create_utils_features(self) -> None:
        """Create utils information as month"""
        
        self.base_data = (
            self.base_data
            .with_columns(
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.truncate('1d').alias('day'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
            .with_columns(
                (pl.col('weekday')>=6).cast(pl.UInt8).alias('is_weekend'),
                (
                    pl.col('hour')
                    .replace(self.slice_hour_mapping).alias('tou')
                ),
                (
                    pl.col('month')
                    .replace(self.month_season_mapping).alias('season')
                )
            )
        )
    def create_feature(self) -> None:   
        self.create_utils_features()
        
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

