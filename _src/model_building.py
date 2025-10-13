import os
import sys
import numpy
import pandas as pd
from abc import ABC, abstractmethod
import mlflow
import dagshub

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor


# initialize dagshub for the mlflow tracking
dagshub.init(repo_owner='nageteychristopher', repo_name='house_price_predictor', mlflow=True)
# create an mlfolw experiment
mlflow.set_experiment("House Price Prediction")
# we will connect our experiments to dagshub for remote tracking
mlflow.set_tracking_uri("https://dagshub.com/nageteychristopher/house_price_predictor.mlflow")


# this the base class for training our model
class TrainModel(ABC):
    @abstractmethod
    def train(self, df_train, df_prid):
        """
        we will call this function to train our model
        args:
            df_train: pd.Dataframe - this will hold the training data (the features)
            df_pred: pd.Dataframe - this will hold the prediction (the goal)
        returns:
            none - we will save the model to mlflow for comparing
        """
        pass

# we will train the a linear regression model
class LinearRegressionModel(TrainModel):
    def train(self, df_train, df_prid):
        """
        we will train a linear regresion model
        args:
            none
        returns:
            none
        """
        # we will automatically log sklearn params, artifacts and metrics of our model
        mlflow.sklearn.autolog()
        # we will start a new run
        with mlflow.start_run(run_name='LinearRegressionModel'):

            clf = LinearRegression()
            param_grid = {
                'n_jobs': [1,1],
                'copy_X': [True,True]
            }

            grid_search = GridSearchCV(clf,param_grid=param_grid, scoring="accuracy",return_train_score=True,cv=10)

            grid_search.fit(df_train,df_prid)



# we will combine models and train them together called Model stacking
class StackingModels(TrainModel):
    def train(self, df_train, df_prid):
        """
        we will perform model stacking with sklearn
        args:
            none
        returns:
            none
        """
        # we will automatically log sklearn params, artifacts and metrics of our model
        mlflow.sklearn.autolog()

        # we will start a new run 
        with mlflow.start_run(run_name="Sklearn Model Stacking"):
            # base models 
            base_estimators = [
                ("rf", RandomForestRegressor(random_state=42)),
                ("gb", GradientBoostingRegressor(random_state=42, learning_rate=0.1))
            ]

            # final model
            meta_model = LinearRegression()

            # create a stack model
            model_stacking = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=3,
                passthrough=True
            )

            # perform hyperparameter tunning
            param_grid = {
                'rf__n_estimators': [100,200,300,500],
                'gb__n_estimators': [100,200,300,500],
                'rf__max_depth': [3,5,7],
                'gb__max_depth': [3,5,7]
            }

            grid_search = GridSearchCV(
                model_stacking,
                param_grid=param_grid,
                scoring="accuracy",
                cv=10
            )

            grid_search.fit(df_train,df_prid)


# create a class to seect a model trainer
class selectmodel():
    def set_strategy (self, strategy):
        """
        we will set the strategy for traning our model
        args:
            strategy - this will hold the strategy
        returns:
            none
        """
        self._strategy = strategy()

    def execute_model_train(self, df_train, df_pred, model: str):
        """
        we will select the model we want to use or tein our data on
        args:
            df_train - this will hold the x_train part of oour data
            df_pred - this the corresponding y_train data
            model: str - this will lets us decide whether we will use the linear regressor or model stacking ['linear','stacking']
            returns
                none
        """
        if isinstance(model,str) and model == 'linear':
            self.set_strategy(LinearRegressionModel)
            self._strategy.train(df_train=df_train,df_prid=df_pred)
        elif isinstance(model,str) and model == 'stacking':
            self.set_strategy(StackingModels)
            self._strategy.train(df_train=df_train,df_prid=df_pred)
        else:
            raise ValueError("model not specified.. [ linear, stacking ]")
        

        