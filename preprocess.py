if __name__=='__main__':
    from src.utils.import_utils import import_config
    from src.preprocess.pipeline import PreprocessPipeline

    config_dict = import_config()
    
    pnl_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
    )
    pnl_preprocessor()