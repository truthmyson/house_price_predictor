import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy
import pandas as pd
import logging
import joblib
from _src.model_evaluation import selectEvaluator, EvaluateLinearModel, EvaluateStackingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _evaluate_model(df_test, df_pred, evaluator: str, r2_value: bool=False, MSE: bool=False, RMSE: bool=False, MAE: bool=False):
    """
    we will evaluate the model
    args:
        df_test - this will hold the x_train part of oour data
        df_pred - this the corresponding y_train data
        evaluator: str - this will lets us decide whether we will use the linear evaluator or stacking evaluator ['linear','stacking']
    returns
        none
    """
    logging.info("model evaluation has started...")

    strategy = selectEvaluator(r2_value=r2_value, MSE=MSE, RMSE=RMSE, MAE=MAE)

    strategy.execute_model_evaluator(df_test=df_test,df_pred=df_pred,evaluator=evaluator)

    logging.info("model evaluation has ended...")

# load the data
df_test = joblib.load('data/arr_data/x_test.npy')
df_pred = joblib.load('data/arr_data/y_test.npy')

_evaluate_model(df_test=df_test,df_pred=df_pred,evaluator='stacking',r2_value=True,MSE=True,RMSE=True,MAE=True) # i will comment out to avoid re-evaluating the model in an existing run name which could cause errors