import logging
import polars as pl
from typing import Any, Union

from src.utils.logging_utils import get_logger
from src.base.preprocess.initialize import BaseInit

class PreprocessInit(BaseInit):
    def __init__(self, 
            config_dict: dict[str, Any],
        ):
        self.config_dict: dict[str, Any] = config_dict
        self.n_folds: int = self.config_dict['N_FOLD']

        self.inference: bool = False
        self._initialize_empty_dataset()       
        self._initialize_col_list()
    
    def _initialize_preprocess_logger(self):
        self.preprocess_logger: logging.Logger = get_logger('preprocess.txt')
    
    def _initialize_col_list(self):
        self.build_id: str = self.config_dict['BUILDING_ID']
    
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
    def _initialize_empty_dataset(self):
        self.base_data: Union[pl.LazyFrame, pl.DataFrame]
        self.label_data: Union[pl.LazyFrame, pl.DataFrame]
        self.data: Union[pl.LazyFrame, pl.DataFrame]
        
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        if isinstance(data, pl.LazyFrame):
            return data.collect().item()
        else:
            return data.item()