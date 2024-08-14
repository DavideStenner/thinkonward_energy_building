import os
import re
import copy
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Union, Tuple, Dict
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
        self.load_used_feature()        
        for type_model in self.model_used:
            if (
                'multi_strategy' in self.params_xgb[type_model].keys()
            ):
                #not supported
                self.__get_permutation_importance(type_model=type_model)
        
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
    def __get_single_score_by_target(self) -> None:
        for current_model in ['commercial', 'residential']:
            best_epoch = self.load_best_result(current_model)['best_epoch']
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
                .filter(pl.col('iteration') == pl.lit(best_epoch-1))
                .drop('iteration')
                .pivot(
                    'fold',
                    index=['col_name'], 
                    values='score',
                )
                .to_pandas()
            )
            #calculate average over each fold
            oof_data["average"] = oof_data.loc[
                :, [f'fold_{x}' for x in range(self.n_fold)]
            ].mean(axis=1)
            (
                oof_data[['col_name', 'average']]
                .to_excel(
                    os.path.join(
                        self.experiment_path_dict['training'].format(type=current_model),
                        f'best_score_by_target.xlsx'
                    ),
                    index=False
                )
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
            plt.axvline(x=best_result['best_epoch'] + 1, color='blue', linestyle='--')

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
            
    def __oof_score(
            self, 
            dataset_list: list[Tuple[pd.DataFrame, pd.DataFrame]],
            position_target_list: list[list[int]],
            model_list: list[xgb.Booster], best_epoch: int
        ) -> float:
        """
        Get oof score on each validation set, using trained model with correct number of epoch.
        Score over each target and mean of all target also

        Args:
            dataset_list (list[Tuple[pd.DataFrame, pd.DataFrame]]): list of oof dataset feature/target

        Returns:
            float: cv oof score
        """
        score_oof = 0

        for fold_, (test_feature, test_target) in enumerate(dataset_list):
            pred_ = model_list[fold_].predict(
                data=xgb.DMatrix(
                    data=test_feature.to_numpy('float64'),
                    feature_names=self.feature_list
                ),
                iteration_range=(0, best_epoch)
            )
            score_fold = np.mean(
                [
                    f1_score(
                        y_true=test_target.to_numpy('float64')[:, position_target].argmax(axis=1),
                        y_pred=pred_[:, position_target].argmax(axis=1),
                        average='macro'
                    )
                    for position_target in position_target_list
                ]
            )

            score_oof += score_fold/self.n_fold

        return score_oof

    def __get_list_of_oof_dataset(self, current_model: str) -> list[Tuple[pd.DataFrame, pd.DataFrame]]:
        list_dataset: list[Tuple[pd.DataFrame, pd.DataFrame]] = []
        
        for fold_ in range(self.n_fold):
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
                .to_pandas()
            )
            
            test_target = (
                fold_data
                .select(self.target_dict[current_model])
                .to_pandas()
            )
            list_dataset.append([test_feature, test_target])
        
        return list_dataset
    
    def __get_permutation_importance(self, type_model: str) -> None:
        self.__get_multi_class_permutation_feature_importance(current_model=type_model)
        self.__get_permutation_importance_by_category_feature(current_model=type_model)
        
    def __shuffled_dataset(self, dataset: pd.DataFrame, feature: str) -> pd.DataFrame:
        dataset[feature] = dataset[feature].sample(frac=1).to_numpy('float64')
        return dataset
    
    def __get_multi_class_permutation_feature_importance(
            self, 
            current_model:str,
            #magical number
            num_repetition: int = 3
        ) -> None:
        """
        Permutation feature importance in cross validation

        Args:
            num_repetition (int, optional): how many times to repeat each fold shuffle. Defaults to 3.
        """
        model_list: list[xgb.Booster] = self.load_pickle_model_list(
            type_model=current_model
        )
        dataset_list: list[Tuple[pd.DataFrame, pd.DataFrame]] = self.__get_list_of_oof_dataset(current_model=current_model)
        best_epoch: int = self.load_best_result(current_model)['best_epoch']
        position_target_list: list[list[int]] = [
            position_target for _, position_target in self.mapper_dummy_target[current_model].items()
        ]
        
        base_score: float = self.__oof_score(
            dataset_list=dataset_list, position_target_list=position_target_list,
            model_list=model_list, best_epoch=best_epoch
        )
        self.training_logger.info(f'{current_model} has a base score of {base_score}')
                
        feature_importance_dict = {
            feature: base_score
            for feature in self.feature_list
        }
        self.training_logger.info(f'Starting {current_model} to calculate permutation importance over {len(self.feature_list)} features')
        for feature in tqdm(self.feature_list):
            shuffled_dataset = copy.deepcopy(dataset_list)

            for _ in range(num_repetition):                
                shuffled_dataset = [
                    [
                        self.__shuffled_dataset(
                            feature_dataset, feature
                        ), 
                        target_dataset
                    ]
                    for feature_dataset, target_dataset in shuffled_dataset
                ]
                result_shuffling = self.__oof_score(
                    dataset_list=shuffled_dataset, position_target_list=position_target_list,
                    model_list=model_list, best_epoch=best_epoch
                )
                feature_importance_dict[feature] -= (
                    result_shuffling/num_repetition
                )

        feature_importance_dict = [
            {
                'feature': feature,
                'importance': change_score
            }
            for feature, change_score in feature_importance_dict.items()
        ]
        result = pd.DataFrame(
            data=feature_importance_dict
        )
        result.to_excel(
            os.path.join(
                self.experiment_path_dict['feature_importance'].format(type=current_model),
                'feature_importances.xlsx'
            ), 
            index=False
        )
    
    def __get_permutation_importance_by_category_feature(self, current_model: str) -> None:
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
            'average_robust_drop_{hour_minute}'
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
        (
            feature_importances
            .groupby('feature_cluster')['importance']
            .agg(
                ['mean', 'min', 'max', 'count']
            )
            .reset_index()
            .to_excel(
                os.path.join(
                    self.experiment_path_dict['feature_importance'].format(type=current_model),
                    'feature_importances_clustered.xlsx'
                ), 
                index=False
            )
        )
        
    def get_oof_insight(self) -> None:
        self.__get_multi_class_insight_by_target()
        self.__get_single_score_by_target()
            
    def get_oof_prediction(self) -> None:
        self.__get_multi_class_score_by_target()