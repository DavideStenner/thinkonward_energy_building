if __name__=='__main__':
    import argparse
    import warnings
    import matplotlib.pyplot as plt

    from src.utils.import_utils import import_config, import_params
    from src.preprocess.pipeline import PreprocessPipeline

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    plt.set_loglevel('WARNING')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='xgb', type=str)
    parser.add_argument('--all_model', action='store_true')
    
    args = parser.parse_args()

    config_dict = import_config()    
    
    preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
    )
    preprocessor.begin_training()
        
    if (args.model == 'lgb') | (args.all_model):
        raise NotImplementedError
    
    elif (args.model == 'xgb') | (args.all_model):
        from src.model.xgbm.pipeline import XgbPipeline

        params_model, experiment_name = import_params(model='xgb')

        trainer = XgbPipeline(
            experiment_name=experiment_name + "_xgb",
            params_xgb=params_model,
            config_dict=config_dict, data_columns=preprocessor.feature_list,
            log_evaluation=50, evaluate_shap=False,
        )
        trainer.train_explain()
