import os
import json
import logging
import polars as pl
from typing import Any, Union, Dict

from src.utils.logging_utils import get_logger
from src.base.preprocess.initialize import BaseInit

class PreprocessInit(BaseInit):
    def __init__(self, 
            config_dict: dict[str, Any],
        ):
        self.config_dict: dict[str, Any] = config_dict
        self.n_folds: int = self.config_dict['N_FOLD']

        self.inference: bool = False
        self._initialize_all()
        
    def _initialize_all(self) -> None:
        self._initialize_empty_dataset()       
        self._initialize_col_list()
        self._initialize_dict_mapper()
        self._initialize_utils()

    def _initialize_utils(self) -> None:
        self.lazy_feature_list: list[pl.LazyFrame] = []
        
    def _initialize_preprocess_logger(self) -> None:
        self.preprocess_logger: logging.Logger = get_logger('preprocess.txt')
    
    def _initialize_dict_mapper(self) -> None:
        self.month_season_mapping: Dict[int, int] = {
            #cold
            1: 0, 2: 0, 12: 0, 
            #hot
            6: 1, 7: 1, 8: 1,
            #mild
            3: 2, 4: 2, 5: 2,
            9: 2, 10: 2, 11: 2
        }
        #https://co.my.xcelenergy.com/s/billing-payment/residential-rates/time-of-use-pricing
        self.slice_hour_mapping: Dict[int, int] = {
            #off peak
            0: 0, 1: 0, 2:0, 3: 0, 4:0, 5: 0, 6: 0, 7: 0, 
            8: 0, 9: 0, 10: 0, 11: 0, 12: 0,
            20: 0, 21: 0, 22: 0, 23: 0,
            #mid peak
            13: 1, 14: 1, 15: 1,
            #on peak
            16: 2, 17: 2, 18: 2, 19: 2,
        }
        
        self.tou_unique: list[int] = list(set(self.slice_hour_mapping.values()))

    def _initialize_col_list(self) -> None:
        self.build_id: str = self.config_dict['BUILDING_ID']
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'], 
                'mapper_category.json'
            ), 'r'            
        ) as file_dtype:
            self.commercial_index = json.load(file_dtype)['train_label']['building_stock_type']['commercial']
        
        self.target_col_binary: str = 'building_stock_type'
        self.target_col_com_list: list[str] = [
            'in.comstock_building_type_group_com', 'in.heating_fuel_com',
            'in.hvac_category_com', 'in.number_of_stories_com',
            'in.ownership_type_com', 'in.vintage_com',
            'in.wall_construction_type_com', 'in.tstat_clg_sp_f..f_com',
            'in.tstat_htg_sp_f..f_com', 'in.weekday_opening_time..hr_com',
            'in.weekday_operating_hours..hr_com',
        ]
        self.target_col_res_list: list[str] = [
            'in.bedrooms_res', 'in.cooling_setpoint_res',
            'in.heating_setpoint_res', 'in.geometry_building_type_recs_res',
            'in.geometry_floor_area_res', 'in.geometry_foundation_type_res',
            'in.geometry_wall_type_res', 'in.heating_fuel_res',
            'in.income_res', 'in.roof_material_res',
            'in.tenure_res', 'in.vacancy_status_res',
            'in.vintage_res',
        ]
        self.all_target_list: list[str] = (
            [self.target_col_binary] +
            self.target_col_com_list +
            self.target_col_res_list
        )
    def _initialize_empty_dataset(self) -> None:
        self.base_data: Union[pl.LazyFrame, pl.DataFrame]
        self.label_data: Union[pl.LazyFrame, pl.DataFrame]
        self.data: Union[pl.LazyFrame, pl.DataFrame]
        
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        if isinstance(data, pl.LazyFrame):
            return data.collect().item()
        else:
            return data.item()