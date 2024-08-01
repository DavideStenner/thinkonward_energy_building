import logging

import polars as pl
import polars.selectors as cs

from typing import Mapping, Union, Tuple, Dict, Any

def get_mapper_categorical(
        config_dict: Dict[str, Any],
        data: Union[pl.LazyFrame, pl.DataFrame], 
        logger: logging.Logger,
        message_format: str='{col} has over {n_unique} different values'
    ) -> Tuple[Union[pl.LazyFrame, pl.DataFrame], Mapping[str, int]]:
    """
    check dataset and return int remapped dataset and the mapper

    Args:
        data (Union[pl.LazyFrame, pl.DataFrame]): dataset

    Returns:
        Union[pl.LazyFrame, pl.DataFrame], Mapping[str, int]: dataset and mapping dictionary
    """
    mapper_mask_col = {}
    lazy_mode = isinstance(data, pl.LazyFrame)

    categorical_col = (
        data.drop(config_dict['BUILDING_ID']).select(cs.by_dtype(pl.String)).collect_schema().names()
        if lazy_mode
        else 
        data.drop(config_dict['BUILDING_ID']).select(cs.by_dtype(pl.String)).columns
    )
    for col in categorical_col:
        
        unique_values = (
            data.select(col).drop_nulls().collect()[col].unique() 
            if lazy_mode 
            else data[col].drop_nulls().unique()
        )
        
        mapper_mask_col[col] = {
            value: i 
            for i, value in enumerate(unique_values.sort().to_list())
        }
        logger.info(
            message_format.format(
                col=col, 
                n_unique=len(unique_values)
            )
        )
    logger.info('\n\n')
    data = remap_category(data=data, mapper_mask_col=mapper_mask_col)
    return data, mapper_mask_col

def remap_category(
        data: Union[pl.LazyFrame, pl.DataFrame], 
        mapper_mask_col: Dict[str, int]
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
    data = data.with_columns(
        [
            pl.col(col_name).replace(replace_mask, default=None).cast(pl.UInt8)
            for col_name, replace_mask in mapper_mask_col.items()
        ]
    )
    return data