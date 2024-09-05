import holidays
import pandas as pd
import polars as pl

from typing import Dict, Tuple
from itertools import product
from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):
    @property
    def __filter_range_work(self) -> Tuple[Dict[str, pl.Expr], Dict[str, pl.Expr]]:
        filter_begin_expr_dict = {
            ##COMMERCIAL 
            #BEGIN
            #only between 3-13, workday
            'com_begin_workday': (pl.col('hour')>=3) & (pl.col('hour')<=13) & (pl.col('weekday')<=5),
            
            #only between 3-13, weekend
            'com_begin_weekend': (pl.col('hour')>=3) & (pl.col('hour')<=13) & (pl.col('weekday')>5),
            
            
            ##RESIDENTIAL
            #BEGIN
            #workday
            'res_begin_workday': (pl.col('weekday')<=5),
            
            #weekend
            'res_begin_weekend': (pl.col('weekday')>5),
        }
            
        filter_end_expr_dict = {
            ##COMMERCIAL 
            #END
            #after 3, workday
            'com_end_workday': (pl.col('hour')>=3) & (pl.col('weekday')<=5),
            
            #after 3, weekend
            'com_end_weekend': (pl.col('hour')>=3) & (pl.col('weekday')>5),
            
            
            ##RESIDENTIAL
            #END
            #after 3, workday
            'res_end_workday': (pl.col('weekday')<=5),
            
            #after 3, weekend
            'res_end_weekend': (pl.col('weekday')>5),
        }
        return filter_begin_expr_dict, filter_end_expr_dict
         
    def __create_hour_aggregation(self) -> pl.LazyFrame:
        """
        Create multiple aggregation over bldg_id only

        Returns:
            pl.LazyFrame: query
        """
        all_hour_aggregation = (
            self.base_data
            .group_by(
                'bldg_id',
            )
            .agg(
                #DAILY AGGREGATION
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
                ] +
                #TOTAL CONSUMPTION
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
                ] +
                #HOLIDAY AGGREGATION
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
                ] +
                #TOU HOLIDAYS
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
                ] +
                #SLICE HOUR AGGREGATION
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
                ] +
                #SLICE DAY AGGREGATION
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
                ] +
                #PIVOTED INFORMATION
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('hour')==hour)
                        )
                        .mean()
                        .alias(f'average_consumption_hour_{hour}_over_month_{month}')
                    )
                    for hour, month in product(self.hour_list, self.month_list)
                ] +
                #HOUR WEEKNUM AGGREGATION
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
                ] +
                #PEAK OVER MONTH
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month)
                        )
                        .mean()
                        .alias(f'average_hour_consumption_month_{month}')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month)
                        )
                        .max()
                        .alias(f'max_hour_consumption_month_{month}')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month)
                        )
                        .min()
                        .alias(f'min_hour_consumption_month_{month}')
                    )
                    for month in self.month_list
                ] +
                [
                    (
                        pl.col('energy_consumption')
                        .filter(
                            (pl.col('month')==month)
                        )
                        .median()
                        .alias(f'median_hour_consumption_month_{month}')
                    )
                    for month in self.month_list
                ]
            )
        )
        return all_hour_aggregation
    
    def __create_hour_profile_consumption(self) -> list[pl.LazyFrame]:
        #by hour and day
        profile_consumption = (
            self.base_data
            .group_by(
                'bldg_id', 'day'
            )
            #find how many min/max for every hour in % and for every day
            .agg(
                [
                    (
                        pl.col('hour')
                        .filter(
                            pl.col('energy_consumption') == pl.col('energy_consumption').max()
                        )
                        .unique()
                        .alias('hour_max')
                    ),
                    (
                        pl.col('hour')
                        .filter(
                            pl.col('energy_consumption') == pl.col('energy_consumption').min()
                        )
                        .unique()
                        .alias('hour_min')
                    )
                ]
            )
        )
        max_profile = (
            profile_consumption
            .select('bldg_id', pl.col('hour_max')).explode(['hour_max'])
            .group_by('bldg_id', 'hour_max')
            .agg(pl.len()/365)
            .group_by('bldg_id').agg(
                [
                    pl.col('len').filter(pl.col('hour_max')==hour).alias(
                        f'profile_max_hour_{hour}'
                    )
                    .first()
                    for hour in self.hour_list
                ]
            )
        )
        min_profile = (
            profile_consumption
            .select('bldg_id', pl.col('hour_min')).explode(['hour_min'])
            .group_by('bldg_id', 'hour_min')
            .agg(pl.len()/365)
            .group_by('bldg_id').agg(
                [
                    pl.col('len').filter(pl.col('hour_min')==hour).alias(
                        f'profile_min_hour_{hour}'
                    )
                    .first()
                    for hour in self.hour_list
                ]
            )
        )

        difference_profile = (
            self.base_data
            .with_columns(
                (
                    (pl.col('energy_consumption') - pl.col('energy_consumption').mean().over('bldg_id', 'day'))
                    .alias('energy_vs_mean')
                ),
                (
                    (pl.col('energy_consumption') - pl.col('energy_consumption').min().over('bldg_id', 'day'))
                    .alias('energy_vs_min')
                ),
                (
                    (pl.col('energy_consumption') - pl.col('energy_consumption').max().over('bldg_id', 'day'))
                    .alias('energy_vs_max')
                )
            )
            .select('bldg_id', 'hour', 'energy_vs_mean', 'energy_vs_min', 'energy_vs_max')
            .group_by('bldg_id', 'hour')
            .agg(
                [
                    pl.col(col).mean()
                    for col in ['energy_vs_mean', 'energy_vs_min', 'energy_vs_max']
                ]
            )
            .group_by('bldg_id')
            .agg(
                [
                    (
                        pl.col(col_name)
                        .filter(pl.col('hour')==hour)
                        .first()
                        .alias(
                            f'profile_{col_name}_hour_{hour}'
                        )
                    )
                    for col_name, hour in product(
                        ['energy_vs_mean', 'energy_vs_min', 'energy_vs_max'],
                        self.hour_list
                    )
                ]
            )
        )

        return [max_profile, min_profile, difference_profile]

    def __create_weekday_profile_consumption(self) -> list[pl.LazyFrame]:
        #by hour and day
        profile_consumption = (
            self.base_data
            .group_by(
                'bldg_id', 'weeknum'
            )
            #find how many min/max for every hour in % and for every day
            .agg(
                [
                    (
                        pl.col('weekday')
                        .filter(
                            pl.col('daily_consumption') == pl.col('daily_consumption').max()
                        )
                        .unique()
                        .alias('weekday_max')
                    ),
                    (
                        pl.col('weekday')
                        .filter(
                            pl.col('daily_consumption') == pl.col('daily_consumption').min()
                        )
                        .unique()
                        .alias('weekday_min')
                    )
                ]
            )
        )
        max_profile = (
            profile_consumption
            .select('bldg_id', pl.col('weekday_max')).explode(['weekday_max'])
            .group_by('bldg_id', 'weekday_max')
            .agg(pl.len()/53)
            .group_by('bldg_id').agg(
                [
                    pl.col('len').filter(pl.col('weekday_max')==weekday).alias(
                        f'profile_max_weekday_{weekday}'
                    )
                    .first()
                    for weekday in self.weekday_list
                ]
            )
        )
        min_profile = (
            profile_consumption
            .select('bldg_id', pl.col('weekday_min')).explode(['weekday_min'])
            .group_by('bldg_id', 'weekday_min')
            .agg(pl.len()/53)
            .group_by('bldg_id').agg(
                [
                    pl.col('len').filter(pl.col('weekday_min')==weekday).alias(
                        f'profile_min_weekday_{weekday}'
                    )
                    .first()
                    for weekday in self.weekday_list
                ]
            )
        )

        difference_profile = (
            self.base_data
            .with_columns(
                (
                    (pl.col('daily_consumption') - pl.col('daily_consumption').mean().over('bldg_id', 'weeknum'))
                    .alias('daily_energy_vs_mean')
                ),
                (
                    (pl.col('daily_consumption') - pl.col('daily_consumption').min().over('bldg_id', 'weeknum'))
                    .alias('daily_energy_vs_min')
                ),
                (
                    (pl.col('daily_consumption') - pl.col('daily_consumption').max().over('bldg_id', 'weeknum'))
                    .alias('daily_energy_vs_max')
                )
            )
            .select('bldg_id', 'weekday', 'daily_energy_vs_mean', 'daily_energy_vs_min', 'daily_energy_vs_max')
            .group_by('bldg_id', 'weekday')
            .agg(
                [
                    pl.col(col).mean()
                    for col in ['daily_energy_vs_mean', 'daily_energy_vs_min', 'daily_energy_vs_max']
                ]
            )
            .group_by('bldg_id')
            .agg(
                [
                    (
                        pl.col(col_name)
                        .filter(pl.col('weekday')==weekday)
                        .first()
                        .alias(
                            f'profile_{col_name}_weekday_{weekday}'
                        )
                    )
                    for col_name, weekday in product(
                        ['daily_energy_vs_mean', 'daily_energy_vs_min', 'daily_energy_vs_max'],
                        self.weekday_list
                    )
                ]
            )
        )

        return [max_profile, min_profile, difference_profile]

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
                ),
                (
                    pl.col('energy_consumption').sum().over([self.build_id, 'day']).alias('daily_consumption')
                )
            )
        )
        self.minute_data = (
            self.minute_data
            .sort(self.build_id, 'timestamp')
            .with_columns(
                (pl.col('timestamp').dt.hour() + pl.col('timestamp').dt.minute()/100).alias('hour_minute'),
                pl.col('timestamp').dt.truncate('1d').alias('day'),
                pl.col('timestamp').dt.month().alias('month'),
                pl.col('timestamp').dt.hour().alias('hour'),
                pl.col('timestamp').dt.week().alias('weeknum'),
                pl.col('timestamp').dt.weekday().alias('weekday')
            )
        )
        
        self.__create_holidays_utils()
    
    def __create_past_future_difference_hour(self) -> list[pl.LazyFrame]:
        """use minutes feature and aggregates


        Returns:
            pl.LazyFrame: query
        """
        base_transformation = (
            self.base_data
            .with_columns(
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .over(self.build_id)
                    .alias('past_hour_energy_consumption')
                ),
                (
                    pl.col('energy_consumption')
                    .rolling_mean(window_size=4)
                    .shift(-5)
                    .over(self.build_id)
                    .alias('future_hour_energy_consumption')
                )
            )
            .with_columns(
                (
                    (pl.col('future_hour_energy_consumption') - pl.col('past_hour_energy_consumption'))
                    .alias('difference_energy_consumption')
                )
            )
        )
        hour_rangework_features = (
            base_transformation
            .group_by(
                self.build_id, 'month', 'is_weekend', 'day'
            )
            #calculate range work
            .agg(
                #opening time has more consume than before
                pl.col('timestamp').filter(
                    (
                        pl.col('difference_energy_consumption') == (
                            pl.col('difference_energy_consumption')
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
                        pl.col('difference_energy_consumption') == (
                            pl.col('difference_energy_consumption')
                            .filter(
                                #after 3, 
                                pl.col('hour')>=6
                            )
                        ).min()
                    ) &
                    (pl.col('hour')>=6)
                ).max().alias('time_end'),
                #opening time has more consume than before
                pl.col('difference_energy_consumption').filter(
                    (
                        pl.col('difference_energy_consumption') == (
                            pl.col('difference_energy_consumption')
                            .filter(
                                #only between 3-13, 
                                (pl.col('hour')>=3) &
                                (pl.col('hour')<=13)
                            )
                        ).max()
                    ) &
                    (pl.col('hour')>=3) &
                    (pl.col('hour')<=13)
                ).first().alias('drop_time_begin'),
                #opening time has less consume than before
                pl.col('difference_energy_consumption').filter(
                    (
                        pl.col('difference_energy_consumption') == (
                            pl.col('difference_energy_consumption')
                            .filter(
                                #after 3, 
                                pl.col('hour')>=6
                            )
                        ).min()
                    ) &
                    (pl.col('hour')>=6)
                ).last().alias('drop_time_end'),
            )
            .with_columns(
                (pl.col('time_end')-pl.col('time_begin')).dt.total_minutes().alias('range_work')
            )
            .group_by(
                self.build_id, 'month', 'is_weekend'
            )
            .agg(
                pl.col('drop_time_begin').mean(),
                pl.col('drop_time_end').mean(),
                pl.col('range_work').mean()
            )
            #min over month
            .group_by(
                self.build_id
            )
            .agg(
                [
                    (
                        pl.col('drop_time_end')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('is_weekend')==is_wekeend)
                        )
                        .mean()
                        .alias(f'average_drop_time_end_{month}_is_wekeend_{is_wekeend}')
                    )
                    for month, is_wekeend in product(
                        self.month_list,
                        [0, 1]
                    )
                ] +
                [
                    (
                        pl.col('drop_time_begin')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('is_weekend')==is_wekeend)
                        )
                        .mean()
                        .alias(f'average_drop_time_begin_{month}_is_wekeend_{is_wekeend}')
                    )
                    for month, is_wekeend in product(
                        self.month_list,
                        [0, 1]
                    )
                ] +
                [
                    (
                        pl.col('range_work')
                        .filter(
                            (pl.col('month')==month) &
                            (pl.col('is_weekend')==is_wekeend)
                        )
                        .mean()
                        .alias(f'average_range_work_{month}_is_wekeend_{is_wekeend}')
                    )
                    for month, is_wekeend in product(
                        self.month_list,
                        [0, 1]
                    )
                ]
            )
        )

        hour_drop_features = (
            base_transformation
            .filter(
                (pl.col('weekday')<=5)
            )
            .group_by(
                self.build_id, 'month', 'hour'
            )
            .agg(
                pl.col('difference_energy_consumption').mean()
            )
            .group_by(
                self.build_id
            )
            .agg(
                [
                    (
                        pl.col('difference_energy_consumption')
                        .filter(
                            (pl.col('month')==month)&
                            (pl.col('hour')==hour)
                        )
                        .mean()
                        .alias(f'average_difference_energy_consumption_{month}_{hour}')
                    )
                    for month, hour in product(self.month_list, self.hour_list)
                ]
            )
        )
        return [hour_rangework_features, hour_drop_features]

    def __create_increment_minutes_features(self) -> pl.LazyFrame:
        """use minutes feature and aggregates


        Returns:
            pl.LazyFrame: query
        """
        
        minutes_increment_features = (
            self.minute_data
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
        
        #add single query
        self.lazy_feature_list += [
            self.__create_hour_aggregation(),
            self.__create_range_work_hour_features(),
            self.__create_increment_decrement_hour_features(),
            self.__create_increment_decrement_hour_by_day_features()
        ]
        #list of query
        self.lazy_feature_list += (
            self.__create_hour_profile_consumption() +
            self.__create_weekday_profile_consumption() +
            self.__create_past_future_difference_hour()
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

        self.data = (
            self.data
            .join(
                self.economic_data,
                on='state', how='left'
            )
        )
