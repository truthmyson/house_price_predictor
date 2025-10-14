import os
import numpy
import pandas as pd
from abc import ABC, abstractmethod
import zipfile

# create a base ingestor class
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str):
        """
        we will ingest a fle path to extract data from it
        args:
            file_path: str - this contains the file path we would be ingesting
        returns:
            none
        """
        
        pass


# we will create a zip file ingestor to extract data from zip files
class ZipfileDataIngestor(DataIngestor):
    def ingest(self, file_path: str):
        """
        we will extract all the files in the zip file into a folder called(extracted_data). after we will go throught the list of extrcted files and select our csv source of data and save it into the fo;der (raw_data)
        args:
            file_path: str - this contains the file path we would be ingesting
        returns:
            none
        """
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall("data/extracted_data")

        all_extracted_files = os.listdir("data/extracted_data")

        csv_files = [file for file in all_extracted_files if file.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("There is no csv file (0 csv files).")
        if len(csv_files) > 1:
           raise ValueError("there are many csv files, select one.")
        
        csv_path = os.path.join('data/extracted_data', csv_files[0])
            
        df = pd.DataFrame(pd.read_csv(csv_path))

        # check if the location exists
        path = "data/raw_data/raw.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # save the data into a folder
        df.to_csv(path, index=False)


# create our ingestor selector
class SelectDataIngestor():
    def __init__(self, ingestor: DataIngestor=DataIngestor):
        """
        we will set a default ingestor with the base ingestor (DataIngestor)
        args:
            ingestor: DataIngestor - stores the base ingestor(DataIngestor)
        returns:
            none - we will set the default ingestor to be DataIngestor
        """
        self._ingestor = ingestor
        
    def set_ingestor(self, ingestor: DataIngestor):
        """
        we will set our preferred ingestor
        args:
            ingestor: DataIngestor - stores our preferred ingestor
        returns:
            none - we will set the ingestor to our preferred ingestor
        """
        self._ingestor = ingestor()

    def execute_ingestor(self,file_path_: str):
        """
        we will validate the extension of the file path to be (.zip) and then clll the righ ingestor
        args:
            file_path: str - this contains the file path of the zip file we want t ingest its contents
        returns:
            none - after validating we will call the righ ingestor with respect to the file extensio of our data source
        """
        file_path_extension = os.path.splitext(file_path_)[1]
        if file_path_extension == ".zip":
            return self._ingestor.ingest(file_path_)
        else:
            raise ValueError("The file path does not contain a zip file.")
        



# # example use case
# if __name__ == "__main__":
#     ingestor = SelectDataIngestor()
#     ingestor.set_ingestor(ZipfileDataIngestor())
#     df = ingestor.execute_ingestor('data/archive.zip')
#     print(df.head())

