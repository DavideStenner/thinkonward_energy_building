import os
import sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    import json
    import polars as pl
    
    from tqdm import tqdm
    from typing import Dict
    from src.utils.dtype import get_mapper_categorical, remap_category
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
    mapper_dataset['train_label'] = mapper_train_label
    
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

        data = pl.concat(
            [
                (
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
                    .group_by(
                        'bldg_id', 'in.state', 
                        pl.col('timestamp').dt.truncate('1h')
                    )
                    .agg(
                        pl.col('out.electricity.total.energy_consumption').sum()
                    )
                    .collect()
                )
                for file_name in tqdm(os.listdir(dataset_chunk_folder))
            ]
        )
        num_rows = data.select(pl.len())
        num_cols = len(data.collect_schema().names())
        
        num_rows = (
            num_rows.collect().item()
            if isinstance(num_rows, pl.LazyFrame)
            else num_rows.item()
        )

        logger.info(f'{dataset_label} file has {num_rows} rows and {num_cols} cols')

        if dataset_label == 'train':
            data, mapper_data = get_mapper_categorical(
                config_dict=config_dict, data=data, logger=logger
            )
            mapper_dataset[f'{dataset_label}_data'] = mapper_data
        else:
            data = remap_category(
                data=data, mapper_mask_col=mapper_data
            )
            
        logger.info(f'Starting saving {dataset_label} dataset')
        data.write_parquet(
            os.path.join(
                config_dict['PATH_SILVER_PARQUET_DATA'],
                config_dict[f'{dataset_label.upper()}_FEATURE_FILE_NAME']
            )
        )

    #saving mapper
    with open(
        os.path.join(
            config_dict['PATH_MAPPER_DATA'], 'mapper_category.json'
        ), 'w'            
    ) as file_dtype:
        
        json.dump(mapper_dataset, file_dtype)