import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy
import pandas as pd
from _src.data_ingestion import SelectDataIngestor, ZipfileDataIngestor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _data_ingestion_step(file_path: str):
    """
    we will ingest the data from the provided file_path
    args:
        file_path: str - this holds the file path of the data we want to ingest
    returns:
        pd.Dataframe - it returns a pandas data frame
    """
    logging.info("Data ingestion has started.")

    ingestor = SelectDataIngestor()
    ingestor.set_ingestor(ZipfileDataIngestor)

    ingestor.execute_ingestor(file_path)

    logging.info("Data ingestion has ended.")

_data_ingestion_step('data/zip_data/archive.zip')
    
