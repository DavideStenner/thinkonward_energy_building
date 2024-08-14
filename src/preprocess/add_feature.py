import holidays
import pandas as pd
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
                        .alias(f'average_hour_consumption_season_{season}_weekday_{weekday}')
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
                        .alias(f'average_hour_consumption_month_{month}_weekday_{weekday}')
                    )
                    for month, weekday in product(self.month_list, self.weekday_list)
                ]
            )
        )
        return all_day_consumption
    
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
                        .filter(pl.col('weekday')==weekday)
                        .sum()
                        .alias(f'total_consumption_weekday_{weekday}')
                    )
                    for weekday in self.weekday_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('hour')==hour)
                        .sum()
                        .alias(f'total_consumption_hour_{hour}')
                    )
                    for hour in self.hour_list
                ] +
                [
                    pl.col('energy_consumption').sum().alias('total_consumption_ever')
                ]
            )
        )
        return total_consumptions
    
    def __create_daily_holidays_feature(self) -> pl.LazyFrame:
        daily_holiday_consumption = (
            self.base_data
            .group_by(
                'bldg_id', 'day'
            )
            .agg(
                pl.col('is_national_holiday').first(), pl.col('is_state_holiday').first(),
                pl.col('energy_consumption').sum().alias('daily_consumption')
            )
            .group_by(
                'bldg_id',
            )
            .agg(
                [
                    (
                        pl.col('daily_consumption')
                        .filter(pl.col('is_national_holiday'))
                        .mean()
                        .alias(f'average_daily_consumption_national_holiday')
                    ),
                    (
                        pl.col('daily_consumption')
                        .filter(pl.col('is_state_holiday'))
                        .mean()
                        .alias(f'average_daily_consumption_state_holiday')
                    )
                ] 
            )
        )
        return daily_holiday_consumption

    def __create_tou_holidays_feature(self) -> pl.LazyFrame:
        all_tou_consumption_holidays = (
            self.base_data
            .group_by(
                'bldg_id', 
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('is_national_holiday')) &
                            (pl.col('tou')==tou)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_national_holiday_tou_{tou}')
                    )
                    for tou in self.tou_unique
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('is_state_holiday')) &
                            (pl.col('tou')==tou)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_state_holiday_tou_{tou}')
                    )
                    for tou in self.tou_unique
                ]
            )
        )
        return all_tou_consumption_holidays
    
    def __create_variation_respect_state(self) -> pl.LazyFrame:
        total_variation_consumptions = (
            self.base_data
            .group_by(
                'bldg_id', 'state'
            )
            .agg(
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('season')==season)
                        .sum()
                        .alias(f'total_consumption_season_{season}_vs_state')
                    )
                    for season in self.season_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('month')==month)
                        .sum()
                        .alias(f'total_consumption_month_{month}_vs_state')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('weekday')==weekday)
                        .sum()
                        .alias(f'total_consumption_weekday_{weekday}_vs_state')
                    )
                    for weekday in self.weekday_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(pl.col('hour')==hour)
                        .sum()
                        .alias(f'total_consumption_hour_{hour}_vs_state')
                    )
                    for hour in self.hour_list
                ] +
                [
                    pl.col('energy_consumption').sum().alias('total_consumption_ever_vs_state')
                ]
            )
            .with_columns(
                [
                    (
                        pl.col(f'total_consumption_season_{season}_vs_state')/
                        pl.col(f'total_consumption_season_{season}_vs_state').mean().over('state')
                    )
                    for season in self.season_list
                ] +
                [
                    (
                        pl.col(f'total_consumption_month_{month}_vs_state')/
                        pl.col(f'total_consumption_month_{month}_vs_state').mean().over('state')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col(f'total_consumption_weekday_{weekday}_vs_state')/
                        pl.col(f'total_consumption_weekday_{weekday}_vs_state').mean().over('state')
                    )
                    for weekday in self.weekday_list
                ] +
                [
                    (
                        pl.col(f'total_consumption_hour_{hour}_vs_state')/
                        pl.col(f'total_consumption_hour_{hour}_vs_state').mean().over('state')
                    )
                    for hour in self.hour_list
                ] +
                [
                    pl.col('total_consumption_ever_vs_state')/pl.col('total_consumption_ever_vs_state').mean().over('state')
                ]
            )
            .drop('state')
        )
        return total_variation_consumptions
    
    def __create_holidays_utils(self) -> pl.LazyFrame:
 
        national_holidays = {
            date_: True
            for date_ in holidays.country_holidays('US', years=2018).keys()
        }
        state_holidays_mapper = {
            state_index: [
                date_.strftime('%Y-%m-%d')
                for date_ in holidays.country_holidays('US', subdiv=state_name, years=2018).keys()
                if date_ not in national_holidays.keys()
            ]
            for state_name, state_index in self.state_mapper.items()
        }

        state_holidays = pd.DataFrame(
            {'state': self.state_mapper.values()}
        )
        state_holidays['day'] = state_holidays['state'].apply(lambda x: state_holidays_mapper[x])
        state_holidays = state_holidays.explode(column='day')
        
        state_holidays = (
            pl.from_dataframe(state_holidays)
            .with_columns(
                pl.col('state').cast(pl.UInt8),
                pl.col('day').cast(pl.Date).cast(pl.Datetime), 
                pl.lit(True).cast(pl.Boolean).alias('is_state_holiday')
            )
            .filter(pl.col('day').is_null().not_())
        )
        if isinstance(self.base_data, pl.LazyFrame):
            state_holidays = state_holidays.lazy()
            
        self.base_data = (
            self.base_data
            .with_columns(
                (
                    pl.col('day').replace(national_holidays, default=False)
                    .cast(pl.Boolean).alias('is_national_holiday')
                )
            )
            .join(
                state_holidays, 
                on=['state', 'day'], how='left'
            )
            .with_columns(pl.col('is_state_holiday').fill_null(False))

        )
        
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
        self.__create_holidays_utils()
        
    def __create_increment_minutes_features(self) -> pl.LazyFrame:
        """use minutes feature and aggregates


        Returns:
            pl.LazyFrame: query
        """
        
        minutes_increment_features = (
            self.minute_data
            .sort(self.build_id, 'timestamp')
            .with_columns(
                (pl.col('timestamp').dt.hour() + pl.col('timestamp').dt.minute()/100).alias('hour_minute'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
            .with_columns(
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .shift(1)
                    .over(self.build_id)
                    .alias('past_hour_energy_consumption')
                )
            )
            .filter(
                #only between 3-13, monday-friday
                (pl.col('hour')>=3) &
                (pl.col('hour')<=13) &
                (pl.col('weekday')<=5)
            )
            .with_columns(
                (
                    (pl.col('energy_consumption') - pl.col('past_hour_energy_consumption'))
                    .alias('difference_energy_consumption_past')
                )
            )
            .group_by(
                self.build_id, 'month', 'weeknum', 'hour_minute'
            )
            #max over week
            .agg(
                
                #opening time has more consume than before
                pl.col('difference_energy_consumption_past').max()
            )
            #min over month
            .group_by(
                self.build_id, 'month', 'hour_minute'
            )
            .agg(
                pl.col('difference_energy_consumption_past').min()
            )
            #now average of min max difference
            .group_by(
                self.build_id, 'hour_minute'
            )
            .agg(
                pl.col('difference_energy_consumption_past').mean().alias(
                    'average_robust_increment'
                )
            )
            #pivot
            .group_by(self.build_id)
            .agg(
                [
                    pl.col('average_robust_increment')
                    .filter(
                        pl.col('hour_minute') == hour_minute
                    )
                    .first()
                    .alias(
                        f'average_robust_increment_{hour_minute}'
                    )
                    for hour_minute in self.increment_hour_minute_list
                ]
            )
        )
        return minutes_increment_features

    def __create_increment_minutes_by_day_features(self) -> pl.LazyFrame:
        """use minutes feature and aggregates


        Returns:
            pl.LazyFrame: query
        """
        
        minutes_increment_features = (
            self.minute_data
            .sort(self.build_id, 'timestamp')
            .with_columns(
                (pl.col('timestamp').dt.hour() + pl.col('timestamp').dt.minute()/100).alias('hour_minute'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
            .with_columns(
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .shift(1)
                    .over(self.build_id)
                    .alias('past_hour_energy_consumption')
                )
            )
            .filter(
                #only between 3-13, monday-friday
                (pl.col('hour')>=3) &
                (pl.col('hour')<=13)
            )
            .with_columns(
                (
                    (pl.col('energy_consumption') - pl.col('past_hour_energy_consumption'))
                    .alias('difference_energy_consumption_past')
                )
            )
            .group_by(
                self.build_id, 'month', 'weeknum', 'weekday'
            )
            #max over week
            .agg(
                
                #opening time has more consume than before
                pl.col('difference_energy_consumption_past').max()
            )
            #min over month
            .group_by(
                self.build_id, 'month', 'weekday'
            )
            .agg(
                pl.col('difference_energy_consumption_past').min()
            )
            #now average of min max difference
            .group_by(
                self.build_id, 'weekday'
            )
            .agg(
                pl.col('difference_energy_consumption_past').mean().alias(
                    'average_robust_increment'
                )
            )
            #pivot
            .group_by(self.build_id)
            .agg(
                [
                    pl.col('average_robust_increment')
                    .filter(
                        pl.col('weekday') == weekday
                    )
                    .first()
                    .alias(
                        f'average_robust_increment_weekday_{weekday}'
                    )
                    for weekday in self.weekday_list
                ]
            )
        )
        return minutes_increment_features

    def __create_drop_minutes_by_day_features(self) -> pl.LazyFrame:
        """use minutes feature and aggregates


        Returns:
            pl.LazyFrame: query
        """
        
        minutes_drop_features = (
            self.minute_data
            .sort(self.build_id, 'timestamp')
            .with_columns(
                (pl.col('timestamp').dt.hour() + pl.col('timestamp').dt.minute()/100).alias('hour_minute'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
            .with_columns(
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .shift(-4)
                    .over(self.build_id)
                    .alias('future_hour_energy_consumption')
                )
            )
            .filter(
                #only between 3-13, monday-friday
                (pl.col('hour')>=14) &
                (pl.col('hour')<=24)
            )
            .with_columns(
                (
                    (pl.col('energy_consumption') - pl.col('future_hour_energy_consumption'))
                    .alias('difference_energy_consumption_future')
                )
            )
            .group_by(
                self.build_id, 'month', 'weeknum', 'weekday'
            )
            #min over week
            .agg(
                
                #ending time has less consume than before
                pl.col('difference_energy_consumption_future').min()
            )
            #max over month
            .group_by(
                self.build_id, 'month', 'weekday'
            )
            .agg(
                pl.col('difference_energy_consumption_future').max()
            )
            #now average of min max difference
            .group_by(
                self.build_id, 'weekday'
            )
            .agg(
                pl.col('difference_energy_consumption_future').mean().alias(
                    'average_robust_drop'
                )
            )
            #pivot
            .group_by(self.build_id)
            .agg(
                [
                    pl.col('average_robust_drop')
                    .filter(
                        pl.col('weekday') == weekday
                    )
                    .first()
                    .alias(
                        f'average_robust_drop_weekday_{weekday}'
                    )
                    for weekday in self.weekday_list
                ]
            )
        )
        return minutes_drop_features

    def __create_drop_minutes_features(self) -> pl.LazyFrame:
        """use minutes feature and aggregates


        Returns:
            pl.LazyFrame: query
        """
        
        minutes_drop_features = (
            self.minute_data
            .sort(self.build_id, 'timestamp')
            .with_columns(
                (pl.col('timestamp').dt.hour() + pl.col('timestamp').dt.minute()/100).alias('hour_minute'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
            .with_columns(
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .shift(-4)
                    .over(self.build_id)
                    .alias('future_hour_energy_consumption')
                )
            )
            .filter(
                #only between 3-13, monday-friday
                (pl.col('hour')>=14) &
                (pl.col('hour')<=24) &
                (pl.col('weekday')<=5)
            )
            .with_columns(
                (
                    (pl.col('energy_consumption') - pl.col('future_hour_energy_consumption'))
                    .alias('difference_energy_consumption_future')
                )
            )
            .group_by(
                self.build_id, 'month', 'weeknum', 'hour_minute'
            )
            #min over week
            .agg(
                
                #ending time has less consume than before
                pl.col('difference_energy_consumption_future').min()
            )
            #max over month
            .group_by(
                self.build_id, 'month', 'hour_minute'
            )
            .agg(
                pl.col('difference_energy_consumption_future').max()
            )
            #now average of min max difference
            .group_by(
                self.build_id, 'hour_minute'
            )
            .agg(
                pl.col('difference_energy_consumption_future').mean().alias(
                    'average_robust_drop'
                )
            )
            #pivot
            .group_by(self.build_id)
            .agg(
                [
                    pl.col('average_robust_drop')
                    .filter(
                        pl.col('hour_minute') == hour_minute
                    )
                    .first()
                    .alias(
                        f'average_robust_drop_{hour_minute}'
                    )
                    for hour_minute in self.drop_hour_minute_list
                ]
            )
        )
        return minutes_drop_features

    def __create_range_work_minutes_features(self) -> pl.LazyFrame:
        """range_work


        Returns:
            pl.LazyFrame: query
        """
        
        minutes_drop_features = (
            self.minute_data
            .sort(self.build_id, 'timestamp')
            .with_columns(
                pl.col('timestamp').dt.day().alias('day'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
            .with_columns(
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .shift(1)
                    .over(self.build_id)
                    .alias('past_hour_energy_consumption')
                )
            )
            .filter(
                #monday-friday
                (pl.col('weekday')<=5)
            )
            .with_columns(
                (
                    (pl.col('energy_consumption') - pl.col('past_hour_energy_consumption'))
                    .alias('difference_energy_consumption_past')
                )
            )
            .group_by(
                self.build_id, 'month', 'weeknum', 'day'
            )
            #calculate range work
            .agg(
                #opening time has more consume than before
                pl.col('timestamp').filter(
                    (
                        pl.col('difference_energy_consumption_past') == (
                            pl.col('difference_energy_consumption_past')
                            .filter(
                                #only between 3-13, 
                                (pl.col('hour')>=3) &
                                (pl.col('hour')<=13)
                            )
                        ).max()
                    ) &
                    (pl.col('hour')>=3) &
                    (pl.col('hour')<=13)
                ).min().alias('time_begin'),
                #opening time has less consume than before
                pl.col('timestamp').filter(
                    (
                        pl.col('difference_energy_consumption_past') == (
                            pl.col('difference_energy_consumption_past')
                            .filter(
                                #after 3, 
                                pl.col('hour')>=3
                            )
                        ).min()
                    ) &
                    (pl.col('hour')>=3)
                ).max().alias('time_end'),
            )
            .with_columns(
                (pl.col('time_end')-pl.col('time_begin')).dt.total_minutes().alias('range_work')
            )
            #max over weeknum
            .group_by(
                self.build_id, 'month', 'weeknum'
            )
            .agg(
                pl.col('range_work').max()
            )
            #min over month
            .group_by(
                self.build_id, 'month'
            )
            .agg(
                pl.col('range_work').min()
            )
            #now average of min max difference
            .group_by(
                self.build_id
            )
            .agg(
                pl.col('range_work').mean().alias(
                    'average_robust_range_work'
                )
            )
        )
        return minutes_drop_features
            
    def create_feature(self) -> None:   
        self.create_utils_features()
        
        self.lazy_feature_list.append(
            self.__create_daily_aggregation()
        )
        self.lazy_feature_list.append(
            self.__create_slice_hour_aggregation()
        )
        self.lazy_feature_list.append(
            self.__create_total_consumptions()
        )
        self.lazy_feature_list.append(
            self.create_slice_day_aggregation()
        )
        self.lazy_feature_list.append(
            self.__create_hour_weeknum_aggregation()
        )
        self.lazy_feature_list.append(
            self.__create_range_work_minutes_features()
        )
        self.lazy_feature_list.append(
            self.__create_increment_minutes_features()
        )
        self.lazy_feature_list.append(
            self.__create_increment_minutes_by_day_features()
        )
        self.lazy_feature_list.append(
            self.__create_tou_holidays_feature()
        )
        self.lazy_feature_list.append(
            self.__create_daily_holidays_feature()
        )
        self.lazy_feature_list.append(
            self.__create_drop_minutes_features()
        )
        self.lazy_feature_list.append(
            self.__create_drop_minutes_by_day_features()
        )
        self.lazy_feature_list.append(
            self.__create_variation_respect_state()
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

