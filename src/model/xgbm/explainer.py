import os
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Union, Tuple
from sklearn.metrics import f1_score
from src.model.xgbm.initialize import XgbInit

class XgbExplainer(XgbInit):       
    def plot_train_curve(self, 
            progress_df: pd.DataFrame, 
            variable_to_plot: Union[str, list],  metric_to_eval: str,
            name_plot: str, type_model: str,
            best_epoch_lgb:int
        ) -> None:
        
        if isinstance(variable_to_plot, str):
            variable_to_plot = [variable_to_plot]
                        
        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=progress_df[['time'] + variable_to_plot].melt(
                id_vars='time',
                value_vars=variable_to_plot,
                var_name='metric_fold', value_name=metric_to_eval
            ), 
            x="time", y=metric_to_eval, hue='metric_fold'
        )
        plt.axvline(x=best_epoch_lgb, color='blue', linestyle='--')

        plt.title(f"Training plot curve of {metric_to_eval}")

        fig.savefig(
            os.path.join(
                self.experiment_path_dict['training'].format(type=type_model),
                f'{name_plot}.png'
            )
        )
        plt.close(fig)

    def evaluate_score(self) -> None:  
        for type_model in self.model_used:
            self.__evaluate_single_model(type_model=type_model)
    
    def __evaluate_single_model(self, type_model: str) -> None:
        metric_eval = self.model_metric_used[type_model]['label']
        metric_to_max = self.model_metric_used[type_model]['maximize']
      

        #load feature list
        self.load_used_feature()
        
        # Find best epoch
        progress_list = self.load_progress_list(
            type_model=type_model
        )

        progress_dict = {
            'time': range(self.params_xgb[type_model]['num_boost_round']),
        }

        list_metric = progress_list[0]['valid'].keys()
        
        for metric_ in list_metric:
            progress_dict.update(
                {
                    f"{metric_}_fold_{i}": progress_list[i]['valid'][metric_]
                    for i in range(self.n_fold)
                }
            )
                        
        progress_df = pd.DataFrame(progress_dict)
        
        for metric_ in list_metric:
            
            progress_df[f"average_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].mean(axis =1)
        
            progress_df[f"std_{metric_}"] = progress_df.loc[
                :, [metric_ in x for x in progress_df.columns]
            ].std(axis =1)

        if metric_to_max:
            best_epoch_lgb = int(progress_df[f"average_{metric_eval}"].argmax())
        else:
            best_epoch_lgb = int(progress_df[f"average_{metric_eval}"].argmin())

        best_score_lgb = progress_df.loc[
            best_epoch_lgb,
            f"average_{metric_eval}"
        ]
        lgb_std = progress_df.loc[
            best_epoch_lgb, f"std_{metric_eval}"
        ]

        self.training_logger.info(f'{type_model} Best epoch: {best_epoch_lgb}, CV-{metric_eval}: {best_score_lgb:.5f} ± {lgb_std:.5f}')

        best_result = {
            'best_epoch': best_epoch_lgb+1,
            'best_score': best_score_lgb
        }
        
        for metric_ in list_metric:
            #plot cv score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=f'average_{metric_}', metric_to_eval=metric_,
                name_plot=f'average_{metric_}_training_curve', type_model=type_model,
                best_epoch_lgb=best_epoch_lgb
            )
            #plot every fold score
            self.plot_train_curve(
                progress_df=progress_df, 
                variable_to_plot=[f'{metric_}_fold_{x}' for x in range(self.n_fold)],
                metric_to_eval=metric_,
                name_plot=f'training_{metric_}_curve_by_fold', type_model=type_model,
                best_epoch_lgb=best_epoch_lgb
            )

        #plot std score
        self.plot_train_curve(
            progress_df=progress_df, 
            variable_to_plot=f'std_{metric_eval}', metric_to_eval=metric_eval,
            name_plot='std_training_curve', type_model=type_model,
            best_epoch_lgb=best_epoch_lgb
        )
        
        self.save_best_result(
            best_result=best_result, type_model=type_model, 
        )
        
    def get_feature_importance(self) -> None:
        for type_model in self.model_used:
            if (
                'multi_strategy' in self.params_xgb[type_model].keys()
            ):
                #not supported
                pass
            else:
                self.__get_single_feature_importance(type_model=type_model)
    
    def __get_single_feature_importance(self, type_model: str) -> None:
        best_result = self.load_best_result(
            type_model=type_model
        )
        model_list: list[xgb.Booster] = self.load_pickle_model_list(
            type_model=type_model, 
        )

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(model_list):
            importance_dict = model.get_score(
                importance_type='gain'
            )

            feature_importances[f'fold_{fold_}'] = (
                feature_importances['feature'].map(importance_dict)
            ).fillna(0)

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        feature_importances = (
            feature_importances[['feature', 'average']]
            .sort_values(by='average', ascending=False)
        )

        #plain feature
        fig = plt.figure(figsize=(18,8))
        sns.barplot(data=feature_importances.head(50), x='average', y='feature')
        plt.title(f"{type_model} 50 TOP feature importance over {self.n_fold} average")

        fig.savefig(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=type_model), 
                'importance_plot.png'
            )
        )
        plt.close(fig)
        
        #feature importance excel
        feature_importances.to_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=type_model), 
                'feature_importances.xlsx'
            ),
            index=False
        )
    def __get_multi_class_insight_by_target(self) -> None:
        for current_model in ['commercial', 'residential']:
            best_result = self.load_best_result(current_model)
            
            #read data and pivot it for seaborn line
            oof_data = (
                pl.read_parquet(
                    os.path.join(
                        self.experiment_path_dict['training'].format(type=current_model),
                        f'multi_class_by_target.parquet'
                    )
                )
                .with_columns(
                    (pl.lit('fold_') + pl.col('fold').cast(pl.Utf8)).alias('fold')
                )
                .pivot(
                    'fold',
                    index=['iteration', 'col_name'], 
                    values='score',
                )
                .to_pandas()
            )
            #calculate average over each fold
            oof_data["average"] = oof_data.loc[
                :, [f'fold_{x}' for x in range(self.n_fold)]
            ].mean(axis=1)

            fig = plt.figure(figsize=(18,8))
            sns.lineplot(
                data=oof_data, 
                x="iteration", y='average', hue='col_name'
            )
            plt.axvline(x=best_result['best_epoch'], color='blue', linestyle='--')

            plt.title(f"Training plot curve of all metric")

            fig.savefig(
                os.path.join(
                    self.experiment_path_dict['training'].format(type=current_model),
                    f'all_target_training_curve.png'
                )
            )
            plt.close(fig)

    def __get_multi_class_score_by_target(self) -> None:
        #get f1 score for each single target, for each round tree
        model_information = {
            type_model: {
                'best_result': self.load_best_result(type_model),
                'model_list': self.load_pickle_model_list(
                    type_model=type_model
                )
            }
            for type_model in ['commercial', 'residential']
        }
        self.load_used_feature()
        self.load_params()
        self.get_target_mapper()
        
        #iterate over each model
        for current_model in ['commercial', 'residential']:
            num_iteration_model: int = self.params_xgb[current_model]['num_boost_round']
            current_model_information = model_information[current_model]
            dummy_target_mapper = self.mapper_dummy_target[current_model]
            f1_score_list: list = []

            #iterate over each fold
            for fold_ in tqdm(range(self.n_fold), total=self.n_fold):
                fold_data = (
                    pl.read_parquet(
                        os.path.join(
                            self.config_dict['PATH_GOLD_PARQUET_DATA'],
                            f'train_{current_model}.parquet'
                        )
                    )
                    .with_columns(
                        (
                            pl.col('fold_info').str.split(', ')
                            .list.get(fold_).alias('current_fold')
                        )
                    )
                    .filter(
                        (pl.col('current_fold') == 'v')
                    )
                )
                test_feature = (
                    fold_data
                    .select(self.feature_list)
                    .to_pandas().to_numpy('float64')
                )
                
                test_feature_matrix = xgb.DMatrix(
                    test_feature,
                    feature_names=self.feature_list
                )
                test_target = (
                    fold_data
                    .select(self.target_dict[current_model])
                    .to_pandas().to_numpy('float64')
                )
                #iterate over each round of training
                for iteration in range(num_iteration_model):
                                                        
                    prediction_: np.ndarray = (
                        current_model_information['model_list'][fold_].predict(
                            test_feature_matrix, 
                            iteration_range=(0, iteration+1)
                        )
                    )
                    #calculate f1 score and create list of list with fold, iteration, col name and score
                    #used for pivot
                    f1_score_list += [
                        [
                            fold_, iteration, col_name, 
                            f1_score(
                                y_true=test_target[:, position_target].argmax(axis=1),
                                y_pred=prediction_[:, position_target].argmax(axis=1),
                                average='macro'
                            )
                        ]
                        for col_name, position_target in dummy_target_mapper.items()
                    ]
            (
                pd.DataFrame(
                    f1_score_list, columns=['fold', 'iteration', 'col_name', 'score']
                )
                .to_parquet(
                    os.path.join(
                        self.experiment_path_dict['training'].format(type=current_model),
                        f'multi_class_by_target.parquet'
                    ),
                    index=False
                )
            )

    def get_oof_insight(self) -> None:
        self.__get_multi_class_insight_by_target()
            
    def get_oof_prediction(self) -> None:
        self.__get_multi_class_score_by_target()