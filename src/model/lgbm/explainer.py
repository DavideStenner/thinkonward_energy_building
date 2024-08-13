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
        
    def get_feature_importance(self) -> None:
        for type_model in self.model_used:
            self.__get_single_feature_importance(type_model=type_model)
    
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

    def __calibration_curve(self, y_prob: np.ndarray, y_true: np.ndarray, n_bins:int = 10) -> Tuple[np.ndarray]:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)

        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]

        return prob_true, prob_pred
    
    def get_oof_insight(self) -> None:
        #read data
        oof_prediction = pl.read_parquet(
            os.path.join(self.experiment_path, 'oof_prediction.parquet')
        )
        
        #score plot
        for type_model in self.model_used:
            fig = plt.figure(figsize=(18,8))
            sns.lineplot(
                data=(
                    oof_prediction
                    .group_by('date_enter', 'fold')
                    .agg(pl.col(f"{type_model}_score").mean())
                    .sort('date_enter')
                ), 
                x="date_enter", y=f"{type_model}_score", hue='fold'
            )
            plt.title(f"Score prediction over date_enter")
            
            fig.savefig(
                os.path.join(
                    self.experiment_path_dict['insight'].format(type=type_model), 
                    f'{type_model}_score_over_date.png'
                )
            )
            plt.close(fig)
            
            #distribution of prediction
            sns.displot(
                oof_prediction.select(f"{type_model}_score"),
                x=f"{type_model}_score", height=8, aspect=18/8
            )
            plt.title(f"Score prediction distribution")
            plt.savefig(
                os.path.join(
                    self.experiment_path_dict['insight'].format(type=type_model), 
                    f'{type_model}_score_distribution.png'
                )
            )
            plt.close()
        
        #calibration curve
        prob_true, prob_pred = self.__calibration_curve(
            y_prob=oof_prediction.select('classification_score').to_numpy().reshape((-1)),
            y_true=oof_prediction.select(pl.col(self.target_col_name)>=0).to_numpy().reshape((-1))
        )
        calibration_df = pd.DataFrame(
            {
                'Fraction of Positives': prob_true,
                'Mean Predicted Probability': prob_pred
            }
        )
        fig = plt.figure(figsize=(18,8))
        sns.lineplot(
            data=calibration_df, 
            x="Mean Predicted Probability", y=f"Fraction of Positives",
            markers=True, 
        )
        plt.plot([0, 1], [0, 1])
        plt.title(f"Calibration Plots")
        
        fig.savefig(
            os.path.join(
                self.experiment_path_dict['insight'].format(type='classification'), 
                f'calibration_plot.png'
            )
        )
        plt.close(fig)

    def get_oof_prediction(self) -> None:
        model_information = {
            type_model: {
                'best_result': self.load_best_result(type_model),
                'model_list': self.load_pickle_model_list(
                    type_model=type_model
                )
            }
            for type_model in self.model_used
        }
        self.load_used_feature()
        
        prediction_list: list[pd.DataFrame] = []
        
        for fold_ in range(self.n_fold):

            fold_data = pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_PARQUET_DATA'],
                    'data.parquet'
                )
            ).with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(fold_).alias('current_fold')
                )
            ).filter(
                (pl.col('current_fold') == 'v')
            )

            test_feature = fold_data.select(self.feature_list).collect().to_pandas().to_numpy('float32')
            
            prediction_df = fold_data.select(
                [
                    'date_enter', 'time_enter', 
                    'is_call', 'stop_itm','pnl'
                ]
            ).collect().to_pandas()
            
            prediction_df['fold'] = fold_
            for type_model in self.model_used:
                current_model_information = model_information[type_model]
                
                prediction_df[f'{type_model}_score'] = (
                    current_model_information['model_list'][fold_].predict(
                        test_feature, 
                        num_iteration=current_model_information['best_result']['best_epoch']
                    )
                )

            prediction_list.append(prediction_df)

        (
            pl.from_dataframe(
                pd.concat(
                    prediction_list, axis=0
                )
            )
            .with_columns(
                pl.col('date_enter').cast(pl.Date),
                pl.col('fold').cast(pl.UInt8),
            )
            .sort(
                ['date_enter']
            )
            .write_parquet(
                os.path.join(self.experiment_path, 'oof_prediction.parquet')
            )
        )
        
    def get_shap_insight(
        self, 
        sample_shap_: int = 10_000,
        top_interaction: int=5,
        select_fold: int = None
    ) -> None:
        #define private function
        print('Starting to calculate shap')
        def get_corrected_corr_matrix(
                shap_array: np.ndarray, 
                noise_: float=0.01
            ) -> np.ndarray:

            #add noise on constant columns
            constant_col = np.where(
                np.std(shap_array, axis=0) == 0
            )[0]
            
            shap_array[:, constant_col] += (
                np.random.random(
                    (shap_array.shape[0], len(constant_col))
                )*noise_
            )
            corr_matrix = np.corrcoef(shap_array.T)
            return corr_matrix

        #best interaction as best correlation feature
        def get_best_interaction(
                idx: int, feature_list: list[str],
                corr_matrix: np.ndarray, top_n: int
            ) -> Tuple[Tuple[str, float]]:

            assert corr_matrix.shape[1] == len(feature_list)    
            
            best_position_ = np.argsort(
                np.abs(corr_matrix), axis=1
            )[idx, -(top_n+1):-1]
            return [
                [feature_list[position], corr_matrix[idx, position]]
                for position in best_position_
            ]

        self.load_best_result()
        self.load_model_list()
        self.load_used_feature()
        
        shap_list: list[np.ndarray] = []
        
        for fold_ in range(self.n_fold):
            if select_fold is not None:
                if fold_!=select_fold:
                    continue
                
            print(f'Shap folder {fold_}')
            fold_data = pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_PARQUET_DATA'],
                    'data.parquet'
                )
            ).with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(fold_).alias('current_fold')
                )
            ).filter(
                (pl.col('current_fold') == 'v')
            )
            
            test_feature = fold_data.select(self.feature_list).collect().to_pandas()
            
            #calculate shap on sampled feature
            shap_ = self.model_list[fold_].predict(
                data=test_feature.sample(sample_shap_).to_numpy('float32'),
                num_iteration=self.best_result['best_epoch'],
                pred_contrib=True
            )
                
            shap_list.append(shap_[:-1])
            
        shap_array = np.concatenate(
            shap_list, axis=0
        )[:, :-1]

        corr_matrix = get_corrected_corr_matrix(
            shap_array=shap_array
        )

        #get ordered best feature
        top_feature_list = pd.read_excel(
            os.path.join(self.experiment_path, 'feature_importances.xlsx'),
            usecols=['feature'],
        )['feature'].tolist()


        top_interaction_list = list(
            chain(
                *[
                    [
                        [
                            rank_base_feature, 
                            feature, feature_interaction, corr_coef, 
                            rank
                        ]
                        for rank, (feature_interaction, corr_coef) in enumerate(
                            get_best_interaction(
                                idx=rank_base_feature, 
                                feature_list=self.feature_list,
                                corr_matrix=corr_matrix,
                                top_n=top_interaction
                            )
                        )
                    ] for rank_base_feature, feature in enumerate(top_feature_list)
                ]
            )
        )
        
        top_interactive_df = pd.DataFrame(
            top_interaction_list,
            columns=[
                'rank_base_feature', 
                'top_feature', 'top_interaction', 'corr_coef',
                'rank_interaction'
            ]   
        )

        shap_df = pd.DataFrame(
            shap_array,
            columns=self.feature_list
        )
        
        #save info
        top_interactive_df.to_csv(
            os.path.join(self.experiment_shap_path, 'top_feature_interaction.csv'),
            index=False
        )
        shap_df.to_csv(
            os.path.join(self.experiment_shap_path, 'array_shap_interaction.csv'),
            index=False
        )
        
    def get_shap_beeswark(
        self, 
        select_fold: int,
        sample_shap_: int = 10_000,
    ) -> None:
        #define private function
        print('Starting to calculate shap')

        self.load_best_result()
        self.load_model_list()
        self.load_used_feature()
        
        feature_data = (
            pl.scan_parquet(
                os.path.join(
                    self.config_dict['PATH_PARQUET_DATA'],
                    'data.parquet'
                )
            )
            .with_columns(
                (
                    pl.col('fold_info').str.split(', ')
                    .list.get(select_fold).alias('current_fold')
                )
            )
            .filter(
                (pl.col('current_fold') == 'v')
            )
            .select(self.feature_list)
            .collect().to_pandas()
            .sample(sample_shap_)
            .to_numpy('float32')
        )
        model = self.model_list[select_fold]
        shap_values = shap.TreeExplainer(model).shap_values(feature_data)
        
        shap.summary_plot(shap_values, features=feature_data, feature_names=self.feature_list)
