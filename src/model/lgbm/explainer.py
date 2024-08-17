import re
import os
import shap
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt

from typing import Union, Tuple
from itertools import chain
from src.model.lgbm.initialize import LgbmInit

class LgbmExplainer(LgbmInit):       
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
        final_score_dict: dict[str, list[float]] = {
            'binary': [],
            'commercial': [],
            'residential': []
        }
        for type_model in self.model_used:
            best_score = self.__evaluate_single_model(type_model=type_model)
            final_score_dict[self.target_class_dict[type_model]].append(best_score)
        
        final_score = (
            final_score_dict['binary'][0] * 0.4 +
            (
                np.mean(final_score_dict['commercial']) * 0.5 +
                np.mean(final_score_dict['residential']) * 0.5
            ) * 0.6
        )
        print(f'\n\nFinal model pipeline {final_score:.6f}')

    def __evaluate_single_model(self, type_model: str) -> float:
        metric_eval = self.model_metric_used[type_model]['label']
        metric_to_max = self.model_metric_used[type_model]['maximize']
        
        #load feature list
        self.load_used_feature()
        
        # Find best epoch
        progress_list = self.load_progress_list(
            type_model=type_model
        )

        progress_dict = {
            'time': range(self.params_lgb['n_round']),
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

        print(f'{type_model} Best epoch: {best_epoch_lgb}, CV-{metric_eval}: {best_score_lgb:.5f} ± {lgb_std:.5f}')

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
        
        return best_score_lgb
    
    def get_feature_importance(self) -> None:
        self.load_used_feature()
        for type_model in self.model_used:
            self.__get_single_feature_importance(type_model=type_model)
            self.__get_feature_importance_by_category_feature(current_model=type_model)
            
    def __get_feature_importance_by_category_feature(self, current_model: str) -> None:
        def replace_multi(x: str) -> str:
            for string_ in [
                r'{season}', r'{tou}', r'{month}',
                r'{week}', r'{is_weekend}', r'{weekday}',
                r'{hour}', r'{weeknum}', r'{hour_minute}'
            ]:
                x = x.replace(string_, r'\d+')
            return x
        
        def get_first_if_any(x: list) -> any:
            if len(x)>0:
                return x[0]
            else:
                return None
            
        feature_importances = pd.read_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=current_model),
                'feature_importances.xlsx'
            )
        )
        feature_list: list[int] = [
            'average_hour_consumption_season_{season}_tou_{tou}',
            'average_hour_consumption_month_{month}_tou_{tou}',
            'average_hour_consumption_week_{week}_tou_{tou}',
            'average_hour_consumption_season_{season}_is_weekend_{is_weekend}',
            'average_hour_consumption_month_{month}_is_weekend_{is_weekend}',
            'average_hour_consumption_week_{week}_is_weekend_{is_weekend}',
            'average_daily_consumption_season_{season}_weekday_{weekday}',
            'average_daily_consumption_month_{month}_weekday_{weekday}',
            'average_hour_consumption_season_{season}',
            'average_hour_consumption_month_{month}',
            'average_hour_consumption_week_{week}',
            'average_daily_consumption_season_{season}',
            'average_daily_consumption_month_{month}',
            'average_daily_consumption_week_{week}',
            'total_average_consumption_weekday_{weekday}',
            'total_average_consumption_hour_{hour}',
            'total_consumption_season_{season}',
            'total_consumption_month_{month}',
            'total_consumption_week_{weeknum}',
            'average_robust_increment_{hour_minute}',
            'average_hour_consumption_national_holiday_tou_{tou}',
            'average_hour_consumption_state_holiday_tou_{tou}',
            'total_consumption_season_{season}_vs_state',
            'total_consumption_month_{month}_vs_state',
            'total_consumption_weekday_{weekday}_vs_state',
            'total_consumption_hour_{hour}_vs_state',
            'average_robust_increment_weekday_{weekday}',
            'average_robust_drop_weekday_{weekday}',
            'average_robust_drop_{hour_minute}',
            'total_consumption_season_{season}_vs_state',
            'total_consumption_month_{month}_vs_state',
            'total_consumption_weekday_{weekday}_vs_state',
            'total_consumption_hour_{hour}_vs_state',
            'total_consumption_ever_vs_state',
            'profile_max_hour_{hour}',
            'profile_min_hour_{hour}'
        ]
        feature_list = [
            replace_multi(x)
            for x in feature_list
        ]
        feature_importances['feature_cluster'] = feature_importances['feature'].apply(
            lambda x:
                get_first_if_any(
                    [
                        pattern_ for pattern_ in feature_list
                        if bool(re.match(pattern_, x))
                    ]
                )
        )
        feature_importances_cluster = (
            feature_importances
            .groupby('feature_cluster')['average']
            .agg(
                ['mean', 'min', 'max', 'count']
            )
            .sort_values('mean')
            .reset_index()
        )
        self.training_logger.info(
            f"Model {current_model} top2 features are {', '.join(feature_importances_cluster['feature_cluster'].iloc[:2])}"
        )
        (
            feature_importances_cluster
            .to_excel(
                os.path.join(
                    self.experiment_path_dict['feature_importance'].format(type=current_model),
                    'feature_importances_clustered.xlsx'
                ), 
                index=False
            )
        )
        
    def __get_single_feature_importance(self, type_model: str) -> None:
        best_result = self.load_best_result(
            type_model=type_model
        )
        model_list: list[lgb.Booster] = self.load_pickle_model_list(
            type_model=type_model, 
        )

        feature_importances = pd.DataFrame()
        feature_importances['feature'] = self.feature_list

        for fold_, model in enumerate(model_list):
            feature_importances[f'fold_{fold_}'] = model.feature_importance(
                importance_type='gain', iteration=best_result['best_epoch']
            )

        feature_importances['average'] = feature_importances[
            [f'fold_{fold_}' for fold_ in range(self.n_fold)]
        ].mean(axis=1)
        
        feature_importances = (
            feature_importances[['feature', 'average']]
            .sort_values(by='average', ascending=False)
        )
        self.training_logger.info(
            f"Model {type_model} top2 features are {', '.join(feature_importances['feature'].iloc[:2])}"
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
    def get_oof_insight(self) -> None:
        pass

    def get_oof_prediction(self) -> None:
        pass