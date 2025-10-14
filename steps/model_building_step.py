import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy
import pandas as pd
import logging
import joblib
from _src.model_building import selectmodel, LinearRegressionModel, StackingModels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _build_model(df_train, df_pred, model: str):
    """
    we will train the model
    args:
        df_train - this will hold the x_train part of oour data
        df_pred - this the corresponding y_train data
        model: str - this will lets us decide whether we will use the linear regressor or model stacking ['linear','stacking']
    returns
        none
    """
    logging.info("model building has started...")

    strategy = selectmodel()

    strategy.execute_model_train(df_train=df_train,df_pred=df_pred,model=model)

    logging.info("model building has ended...")

# load the data
df_train = joblib.load('data/arr_data/x_train.npy')
df_pred = joblib.load('data/arr_data/y_train.npy')


_build_model(df_train=df_train,df_pred=df_pred,model='linear') # i will comment out to avoid retraining the model in an existing nun name which could cause errors