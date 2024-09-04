if __name__=='__main__':
    import argparse
    
    from src.utils.import_utils import import_config
    from src.preprocess.pipeline import PreprocessPipeline
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--additional', action='store_true')

    args = parser.parse_args()
    
    config_dict = import_config()
    config_dict['additional'] = args.additional
    
    pnl_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
    )
    #train datasets
    pnl_preprocessor()
    
    #also test set
    pnl_preprocessor.begin_inference()
    pnl_preprocessor()