import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy
import pandas as pd
import logging
from _src.feature_engineering import selectFeatureEngineeringStrtegy, x_FetureEngineering, y_FetureEngineering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _feature_engineer(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    we will perform feature engineering oh are test and train sets
    args:
        x_train: pd.Dtaframe - this contins our x train data set
        x_test: pd.Dtaframe - this contains our x test data set
        y_train: pd.Series - this contains our y train series
        y_test: pd.Series - this contains our y test series
    args:
        numpy.array - we will return 4 numpy arrays of the test and train data sets
    """
    
    logging.info("feature engineering has started..")

    feature_engineer = selectFeatureEngineeringStrtegy()
    if isinstance(x_train, pd.DataFrame):
        feature_engineer.set_strategy(x_FetureEngineering)
        feature_engineer.execute_strategy(df=x_train,col_onehotencode=['mainroad','guestroom','airconditioning','hotwaterheating','basement','prefarea','furnishingstatus'],col_skewed=['price','area','parking','stories','bathrooms'],col_outliers=['price','area','parking','stories','bathrooms','bedrooms'],method='train')

    if isinstance(x_test, pd.DataFrame):
        feature_engineer.set_strategy(x_FetureEngineering)
        feature_engineer.execute_strategy(x_train,col_onehotencode=['mainroad','guestroom','airconditioning','hotwaterheating','basement','prefarea','furnishingstatus'],col_skewed=['price','area','parking','stories','bathrooms'],col_outliers=['price','area','parking','stories','bathrooms','bedrooms'],method='test')
    
    if isinstance(y_train, pd.Series):
        y_train = pd.DataFrame(y_train)
        feature_engineer.set_strategy(y_FetureEngineering)
        feature_engineer.execute_strategy(df=y_train,method='train')

    if isinstance(y_test, pd.Series):
        y_test = pd.DataFrame(y_test)
        feature_engineer.set_strategy(y_FetureEngineering)
        feature_engineer.execute_strategy(df=y_test,method='test')


    logging.info("feature engineering has ended..")
    

x_train = pd.DataFrame(pd.read_csv('data/splitted_data/x_train.csv'))
x_test = pd.DataFrame(pd.read_csv('data/splitted_data/x_test.csv'))
y_train = pd.Series(pd.read_csv('data/splitted_data/y_train.csv'))
y_test = pd.Series(pd.read_csv('data/splitted_data/y_test.csv'))

_feature_engineer(x_train,x_test,y_train,y_test)