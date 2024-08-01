import os
import sys
sys.path.append(os.getcwd())

if __name__ == '__main__':
    import polars as pl
    from src.utils.import_utils import import_config
    from src.utils.logging_utils import get_logger
    
    logger = get_logger(file_name='concat_data.log')

    config_dict = import_config()

    #import and save label file
    logger.info('importing and saving label file')
    train_label = pl.read_parquet(
        os.path.join(
            config_dict['PATH_ORIGINAL_DATA'],
            config_dict['ORIGINAL_TRAIN_LABEL_FOLDER'],
            config_dict['TRAIN_LABEL_FILE_NAME']
        )
    )
    logger.info(f'label file has {train_label.shape[0]} rows and {train_label.shape[0]} cols')
    train_label.write_parquet(
        os.path.join(
            config_dict['PATH_SILVER_PARQUET_DATA'],
            config_dict['TRAIN_LABEL_FILE_NAME']
        )
    )
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
                pl.scan_parquet(
                    os.path.join(dataset_chunk_folder, file_name),
                    hive_schema={
                        'timestamp': pl.Datetime('ns'),
                        'out.electricity.total.energy_consumption': pl.Float64,
                        'in.state': pl.Object,
                        'bldg_id': pl.Int64
                    }
                )
                for file_name in os.listdir(dataset_chunk_folder)
            ]
        )
        num_rows = data.select(pl.len()).collect().item()
        num_cols = len(data.collect_schema().names())
        
        logger.info(f'{dataset_label} file has {num_rows} rows and {num_cols} cols')

        logger.info(f'Starting sinking {dataset_label} dataset')
        data.sink_parquet(
            os.path.join(
                config_dict['PATH_SILVER_PARQUET_DATA'],
                config_dict[f'{dataset_label.upper()}_FEATURE_FILE_NAME']
            )
        )
