import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy
import pandas as pd
import logging
from _src.data_splitting import SelectSplitter, TestTrainSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _split_data(path):
    """
    we will spit our data frame into train and testing seets
    args:
        df: pd.Dataframe - this eill hold our data frame
    returns:
        pd.Dataframes - we will return 2 pandas dataframe that is (x_train, x_test)
        pd.Series - we will return 2 pands series that is (y_train, y_test)
    """
    logging.info("Data spltting has started")

    # load the raw data
    df = pd.DataFrame(pd.read_csv(path))

    split = SelectSplitter(test_size=0.2,random_state=42,target_var='price')
    split.set_splitter(TestTrainSplit)
    split.execute_splitter(df)

    logging.info("Data splitting has ended")


_split_data('data/raw_data/raw.csv')