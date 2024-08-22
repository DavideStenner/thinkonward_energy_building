import os
import sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    import json
    import polars as pl
    
    from tqdm import tqdm
    from typing import Dict
    from src.utils.dtype import get_mapper_categorical, remap_category, correct_ordinal_categorical
    from src.utils.import_utils import import_config
    from src.utils.logging_utils import get_logger
            
    mapper_dataset: Dict[str, Dict[str, int]] = {}
    logger = get_logger(file_name='concat_data.log')

    config_dict = import_config()

    #import and save label file
    logger.info('importing and saving label file')
    train_label = (
        pl.scan_parquet(
            os.path.join(
                config_dict['PATH_ORIGINAL_DATA'],
                config_dict['ORIGINAL_TRAIN_LABEL_FOLDER'],
                config_dict['TRAIN_LABEL_FILE_NAME']
            )
        )
    )
    num_rows = train_label.select(pl.len()).collect().item()
    num_cols = len(train_label.collect_schema().names())

    logger.info(f'label file has {num_rows} rows and {num_cols} cols\n\n')
    
    train_label, mapper_train_label = get_mapper_categorical(
        config_dict=config_dict, data=train_label, logger=logger
    )
    mapper_dataset['train_label'] = correct_ordinal_categorical(mapper_train_label)
    
    train_label.sink_parquet(
        os.path.join(
            config_dict['PATH_SILVER_PARQUET_DATA'],
            config_dict['TRAIN_LABEL_FILE_NAME']
        )
    )
    #begin train test feature
    for dataset_label, path_folder in [
        ['train', config_dict['ORIGINAL_TRAIN_CHUNK_FOLDER']],
        ['test', config_dict['ORIGINAL_TEST_CHUNK_FOLDER']]
    ]:
        logger.info(f'Scanning {dataset_label} dataset chunk')
        dataset_chunk_folder: str = os.path.join(
            config_dict['PATH_ORIGINAL_DATA'],
            path_folder
        )
        data_hour_list: list[pl.DataFrame] = []
        data_minute_list: list[pl.DataFrame] = []
        
        for file_name in tqdm(os.listdir(dataset_chunk_folder)):
            minute_result = (
                pl.scan_parquet(
                    os.path.join(dataset_chunk_folder, file_name),
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
                    'bldg_id', 'in.state', 
                    pl.col('timestamp').dt.truncate('1h')
                )
                .agg(
                    pl.col('out.electricity.total.energy_consumption').sum()
                )
            )
            data_hour_list.append(hour_result)
            data_minute_list.append(minute_result)
            
        data_hour = pl.concat(data_hour_list)
        data_minute = pl.concat(data_minute_list)
        
        for title_dataset_, dataset_ in [['hour', data_hour], ['minute', data_minute]]:
            num_rows = dataset_.select(pl.len())
            num_cols = len(dataset_.collect_schema().names())
            
            num_rows = (
                num_rows.collect().item()
                if isinstance(num_rows, pl.LazyFrame)
                else num_rows.item()
            )

            logger.info(f'{dataset_label}-{title_dataset_} file has {num_rows} rows and {num_cols} cols')

        if dataset_label == 'train':
            data_hour, mapper_data = get_mapper_categorical(
                config_dict=config_dict, data=data_hour, logger=logger
            )
            mapper_dataset[f'{dataset_label}_data'] = mapper_data
            data_minute = remap_category(
                data=data_minute, mapper_mask_col=mapper_data
            )
        else:
            data_hour = remap_category(
                data=data_hour, mapper_mask_col=mapper_data
            )
            data_minute = remap_category(
                data=data_minute, mapper_mask_col=mapper_data
            )
            
        logger.info(f'Starting saving {dataset_label} hour dataset')
        data_hour.write_parquet(
            os.path.join(
                config_dict['PATH_SILVER_PARQUET_DATA'],
                config_dict[f'{dataset_label.upper()}_FEATURE_HOUR_FILE_NAME']
            )
        )
        logger.info(f'Starting saving {dataset_label} minute dataset')
        data_minute.write_parquet(
            os.path.join(
                config_dict['PATH_SILVER_PARQUET_DATA'],
                config_dict[f'{dataset_label.upper()}_FEATURE_MINUTE_FILE_NAME']
            )
        )

    #saving mapper
    with open(
        os.path.join(
            config_dict['PATH_MAPPER_DATA'], 'mapper_category.json'
        ), 'w'            
    ) as file_dtype:
        
        json.dump(mapper_dataset, file_dtype)