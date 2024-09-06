import os
import sys

sys.path.append(os.getcwd())

if __name__ == '__main__':
    import warnings
    import polars as pl

    from glob import glob
    from tqdm import tqdm
    from typing import Dict
    from src.utils.import_utils import import_config
    from src.utils.logging_utils import get_logger
    from src.utils.dtype import remap_category

    warnings.simplefilter("ignore", pl.exceptions.CategoricalRemappingWarning)
    logger = get_logger(file_name='add_new_data.log')

    config_dict = import_config()
    
    #import mapper
    mapper_label = import_config(
        os.path.join(
            config_dict['PATH_MAPPER_DATA'], 
            'mapper_category.json'
        )
    )

    #import and save label file
    logger.info('importing new data file')

    data_hour_list = []
    
    folder_map = config_dict['ADDITIONAL_DICT_INFO']
    total_number_data = len(
        glob(
            'data_dump/*/*/*.parquet'
        )
    )
    bar_file = tqdm(total = total_number_data+1)

    for type_building, type_dict in folder_map.items():
        type_building_path: str = os.path.join(
            'data_dump', type_dict['path']
        )
        
        state_folder_list = os.listdir(type_building_path)
        
        for state_folder in state_folder_list:
            state_string: str = state_folder.split('=')[-1]
            
            file_path_list: list[str] = os.listdir(
                os.path.join(
                    type_building_path,
                    state_folder,
                )
            )
            for file_path in file_path_list:
                
                minute_result = (
                    pl.scan_parquet(
                        os.path.join(
                            type_building_path,
                            state_folder,
                            file_path
                        )
                    ).select(
                        pl.col('timestamp'), 
                        pl.col('out.electricity.total.energy_consumption'), 
                        pl.col('bldg_id'),
                        pl.lit(state_string).cast(pl.Utf8).alias('in.state'),
                        pl.lit(type_building).cast(pl.Utf8).alias('build_type')
                    )
                    .with_columns(
                        pl.col('timestamp').cast(pl.Datetime),
                        pl.col('out.electricity.total.energy_consumption').cast(pl.Float64),
                        pl.col('in.state').cast(pl.Utf8),
                        pl.col('bldg_id').cast(pl.Int64)
                    )
                    .with_columns(
                        pl.col('timestamp').dt.offset_by('-15m')
                    )
                    .collect()
                )
                hour_result = (
                    minute_result
                    .group_by(
                        'bldg_id', 'in.state', 'build_type',
                        pl.col('timestamp').dt.truncate('1h')
                    )
                    .agg(
                        pl.col('out.electricity.total.energy_consumption').sum()
                    )
                )
                data_hour_list.append(hour_result)
                
                bar_file.update(1)
    
    bar_file.close()
    logger.info('Collecting dataset')
    
    data_hour: pl.DataFrame = pl.concat(data_hour_list)

    for title_dataset_, dataset_ in [['hour', data_hour]]:
        num_rows = dataset_.select(pl.len()).item()
        num_cols = len(dataset_.collect_schema().names())
        
        logger.info(f'{title_dataset_} file has {num_rows} rows and {num_cols} cols')
        
    logger.info('Remapping dataset hour')

    data_hour = remap_category(
        data=data_hour, mapper_mask_col=mapper_label['train_data']
    )

    #METADATA
    metadata_res = (
        pl.scan_parquet(
            os.path.join(
                'data_dump',
                folder_map['residential']['metadata']
            )
        )
        .select(
            pl.col('bldg_id'),
            pl.lit('residential').alias('build_type'),
            pl.col('in.bedrooms').alias('in.bedrooms_res'),
            pl.col('in.cooling_setpoint').alias('in.cooling_setpoint_res'),
            pl.col('in.heating_setpoint').alias('in.heating_setpoint_res'),
            pl.col('in.geometry_building_type_recs').alias('in.geometry_building_type_recs_res'),
            pl.col('in.geometry_floor_area').alias('in.geometry_floor_area_res'),
            pl.col('in.geometry_foundation_type').alias('in.geometry_foundation_type_res'),
            pl.col('in.geometry_wall_type').alias('in.geometry_wall_type_res'),
            pl.col('in.heating_fuel').alias('in.heating_fuel_res'),
            pl.col('in.income').alias('in.income_res'),
            pl.col('in.roof_material').alias('in.roof_material_res'),
            pl.col('in.tenure').alias('in.tenure_res'),
            pl.col('in.vacancy_status').alias('in.vacancy_status_res'),
            pl.col('in.vintage').alias('in.vintage_res')
        )
    )
    metadata_com = (
        pl.scan_parquet(
            os.path.join(
                'data_dump',
                folder_map['commercial']['metadata']
            )
        )
        .select(
            pl.col('bldg_id'),
            pl.lit('commercial').alias('build_type'),
            pl.col('in.comstock_building_type_group').alias('in.comstock_building_type_group_com'),
            pl.col('in.heating_fuel').alias('in.heating_fuel_com'),
            pl.col('in.hvac_category').alias('in.hvac_category_com'),
            pl.col('in.number_of_stories').alias('in.number_of_stories_com'),
            pl.col('in.ownership_type').alias('in.ownership_type_com'),
            pl.col('in.vintage').alias('in.vintage_com'),
            pl.col('in.wall_construction_type').alias('in.wall_construction_type_com'),
            pl.col('in.tstat_clg_sp_f..f').alias('in.tstat_clg_sp_f..f_com'),
            pl.col('in.tstat_htg_sp_f..f').alias('in.tstat_htg_sp_f..f_com'),
            pl.col('in.weekday_opening_time..hr').alias('in.weekday_opening_time..hr_com'),
            pl.col('in.weekday_operating_hours..hr').alias('in.weekday_operating_hours..hr_com'),
        )
    )

    metadata = (
        pl.concat(
            [metadata_res, metadata_com],
            how='diagonal',
        )
        .collect()
        .join(
            data_hour.select('bldg_id', 'build_type').unique(), 
            on=['bldg_id', 'build_type']
        )
    )

    #remap id
    id_build_list: list[str] = (
        metadata
        .select(
            pl.col('bldg_id').cast(pl.Utf8) + pl.col('build_type')
        )
        .unique()
        .to_numpy()
        .reshape((-1))
        .tolist()
    )
    mapper_id = {
        key_: 30_000 + id_
        for id_, key_ in enumerate(id_build_list,)
    }
    #remap id on dataset
    data_hour = (
        data_hour
        .with_columns(
            (
                (
                    pl.col('bldg_id').cast(pl.Utf8) + pl.col('build_type')
                )
                .replace(mapper_id)
                .cast(pl.Int64)
                .alias('bldg_id')
            )
        )
        .select(
            ['timestamp', 'out.electricity.total.energy_consumption', 'in.state', 'bldg_id']
        )
    )


    #remap metadata
    metadata = (
        metadata
        .with_columns(
            (
                (
                    pl.col('bldg_id').cast(pl.Utf8) + pl.col('build_type')
                )
                .replace(mapper_id)
                .cast(pl.Int64)
                .alias('bldg_id')
            )
        )
        .rename({'build_type': 'building_stock_type'})
    )

    logger.info('Remapping metadata')
    metadata = remap_category(
        data=metadata, mapper_mask_col=mapper_label['train_label']
    )
    
    logger.info(f'Starting saving hour dataset')
    data_hour.write_parquet(
        os.path.join(
            config_dict['PATH_SILVER_PARQUET_DATA'],
            'train_data_additional.parquet'
        )
    )
    
    logger.info(f'Starting saving metadata dataset')
    metadata.write_parquet(
        os.path.join(
            config_dict['PATH_SILVER_PARQUET_DATA'],
            'train_label_additional.parquet'
        )
    )