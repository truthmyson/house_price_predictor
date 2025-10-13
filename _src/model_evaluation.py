import os
import sys

import numpy
import mlflow
from mlflow import MlflowClient
import pandas as pd
from abc import ABC, abstractmethod
import dagshub

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error

# initialize dagshub for the mlflow tracking
dagshub.init(repo_owner='nageteychristopher', repo_name='house_price_predictor', mlflow=True)

# we will connect our experiments to dagshub for remote tracking
mlflow.set_tracking_uri("https://dagshub.com/nageteychristopher/house_price_predictor.mlflow")


# this is a base class to evaluate a model
class EvaluateModel(ABC):
    @abstractmethod
    def evaluate(self, df_test, df_pred):
        """
        we will evaluate our model on the data set
        args:
            df_test - this will hold the x test data set
            df_pred - this will be the actual value for the y test
        returns:
            none
        """


# we will evaluate our linear model
class EvaluateLinearModel(EvaluateModel):
    def __init__(self, r2_value: bool=False, MSE: bool=False, RMSE: bool=False, MAE: bool=False):
        """
        we will initialize the metrixes for the model after prediction to e false
        args:
            r2_value: bool
            MSE: bool - this will activate or deactivate the metrix men square error
            RMSE: bool - this will activate or deactivate the metrix root mean square error
            MAE: bool - this will activate or deactivate the metrix mean absolute error
        returns:
            none
        """
        self._r2_value = r2_value
        self._MSE = MSE
        self._RMSE = RMSE
        self._MAE = MAE

    def evaluate(self, df_test, df_pred):
        """
        we will predict the on the x test data set and compare to our original y test 
        args:
            df_test - this will contain the x test set for our model
            df_pred - this will contaiin our actual y test set
        returns:
            none
        """
        run_id = '3e9618725ec94ffd9851a84beb543d9a'
        model_name = 'linear-model'

        model_url = f"runs:/{run_id}/model"
        
        # this will register the model for that particular run id
        # we will check if the model has already been registered

        filter = "run_id = '3e9618725ec94ffd9851a84beb543d9a'"
        model_versions = mlflow.search_model_versions(filter_string=filter)

        if len(model_versions) == 0:
            registor_model = mlflow.register_model(model_url,model_name)
            model_version = registor_model.version
        else:
            model_version = 1

        with mlflow.start_run(run_id='3e9618725ec94ffd9851a84beb543d9a'):
            # we will load the model
            load_model_url = f"models:/{model_name}/{model_version}"
            model = mlflow.sklearn.load_model(load_model_url)

            model_y_pred = model.predict(df_test)

            # we will calculate and log metrics to the mlflow
            if self._r2_value:
                _r2_value = r2_score(model_y_pred,df_pred)
                mlflow.log_metric("r2_score",_r2_value,run_id='3e9618725ec94ffd9851a84beb543d9a')
            if self._MAE:
                _MAE = mean_absolute_error(model_y_pred,df_pred)
                mlflow.log_metric("MAE",_MAE, run_id='3e9618725ec94ffd9851a84beb543d9a')
            if self._RMSE:
                _RMSE = root_mean_squared_error(model_y_pred,df_pred)
                mlflow.log_metric("RMSE",_RMSE, run_id='3e9618725ec94ffd9851a84beb543d9a')
            if self._MSE:
                _MSE = mean_squared_error(model_y_pred,df_pred)
                mlflow.log_metric("MSE", _MSE, run_id='3e9618725ec94ffd9851a84beb543d9a')


# we will evaluate our stacking model
class EvaluateStackingModel(EvaluateModel):
    def __init__(self, r2_value: bool=False, MSE: bool=False, RMSE: bool=False, MAE: bool=False):
        """
        we will initialize the metrics for the model after prediction to be false
        args:
            r2_value: bool
            MSE: bool - this will activate or deactivate the metrix men square error
            RMSE: bool - this will activate or deactivate the metrix root mean square error
            MAE: bool - this will activate or deactivate the metrix mean absolute error
        returns:
            none
        """
        self._r2_value = r2_value
        self._MSE = MSE
        self._RMSE = RMSE
        self._MAE = MAE

    def evaluate(self, df_test, df_pred):
        """
        we will predict the on the x test data set and compare to our original y test 
        args:
            df_test - this will contain the x test set for our model
            df_pred - this will contaiin our actual y test set
        returns:
            none
        """
        run_id = '654ac78ccb08495687d1b424635fb986'
        model_name = 'stacking-model'

        model_url = f"runs:/{run_id}/best_estimator"
        
        # this will register the model for that particular run id
        # we will check if the model has already been registered
        filter = "run_id = '654ac78ccb08495687d1b424635fb986'"
        model_versions = mlflow.search_model_versions(filter_string=filter)

        if len(model_versions) == 0:
            registor_model = mlflow.register_model(model_url,model_name)
            model_version = registor_model.version
        else:
            model_version = 1

        with mlflow.start_run(run_id='654ac78ccb08495687d1b424635fb986'):
            # we will load the model
            load_model_url = f"models:/{model_name}/{model_version}"
            model = mlflow.sklearn.load_model(load_model_url)

            model_y_pred = model.predict(df_test)

            # we will calculate and log metrics to the mlflow
            if self._r2_value:
                _r2_value = r2_score(model_y_pred,df_pred)
                mlflow.log_metric("r2_score",_r2_value)
            if self._MAE:
                _MAE = mean_absolute_error(model_y_pred,df_pred)
                mlflow.log_metric("MAE",_MAE)
            if self._RMSE:
                _RMSE = root_mean_squared_error(model_y_pred,df_pred)
                mlflow.log_metric("RMSE",_RMSE)
            if self._MSE:
                _MSE = mean_squared_error(model_y_pred,df_pred)
                mlflow.log_metric("MSE", _MSE)


# we will select our evaluator
class selectEvaluator():
    def __init__(self, r2_value: bool=False, MSE: bool=False, RMSE: bool=False, MAE: bool=False):
        """
        we will initialize the metrics for the model for prediction to be false
        args:
            r2_value: bool
            MSE: bool - this will activate or deactivate the metrix men square error
            RMSE: bool - this will activate or deactivate the metrix root mean square error
            MAE: bool - this will activate or deactivate the metrix mean absolute error
        returns:
            none
        """
        self._r2_value = r2_value
        self._MSE = MSE
        self._RMSE = RMSE
        self._MAE = MAE

    def set_strategy (self, strategy):
        """
        we will set the strategy for evaluating our model
        args:
            strategy - this will hold the strategy
        returns:
            none
        """
        self._strategy = strategy(r2_value=self._r2_value,MSE=self._MSE, RMSE=self._RMSE, MAE=self._MAE)

    def execute_model_evaluator(self, df_test, df_pred, evaluator: str):
        """
        we will select the model we want to use or evaluate our data on
        args:
            df_train - this will hold the x_train part of oour data
            df_pred - this the corresponding y_train data
            model: str - this will lets us decide whether we will use the linear evaluator or stacking evaluator ['linear','stacking']
            returns
                none
        """
        if isinstance(evaluator,str) and evaluator == 'linear':
            self.set_strategy(EvaluateLinearModel)
            self._strategy.evaluate(df_test=df_test,df_pred=df_pred)
        elif isinstance(evaluator,str) and evaluator == 'stacking':
            self.set_strategy(EvaluateStackingModel)
            self._strategy.evaluate(df_test=df_test,df_pred=df_pred)
        else:
            raise ValueError("evaluator not specified.. [ linear, stacking ]")