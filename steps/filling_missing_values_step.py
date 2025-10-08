import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy
import pandas as pd
import logging
from _src.fill_missing_data import selectFiller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    we will handle missing values y dropping rows involved or filling in the missing values
    args:
        df: pd.Dataframe - this will hold the unprocessed dataframe
    returns:
        pd.Dataframe - this will return a dataframe with no or little missing values
    """
    logging.info("Handling missing values has begun...")

    filler = selectFiller()
    columns = df.columns.tolist()

    df = filler.execute_handler(df=df,columns=columns,method='fill',n_strategy='mean',c_strategy='most_frequent') #dad the rest of params

    logging.info("Handing missing values has finished...")

    return df

