import os
import sys
import numpy
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from typing import Tuple

# this the base class for the splitting the data into train test sets
class SplittingDataset(ABC):
    @abstractmethod
    def split(self, df: pd.DataFrame):
        """
        this split our data into our desired ratio and return the test and train sets
        args:
            df: pd.Dataframe - this will contain or dataframe
        returns:
            none
        """
        pass

# this is the slitter class
class TestTrainSplit(SplittingDataset):
    def __init__(self, test_size: float, random_state: int, target_var: str, shuffle: bool=True):
        """
        we will initialize our slitter with some values like test_size, etc
        args:
            test_size: float - this will determine the size of the test data set(multiply by 100 to gfet a percentage)
            random_state: int - this will help us get a consitent result when ever we run the program
            target_var: str - this will hold the target column name
            shuffle: bool - this will help shuffle our data before splitting
        returns:
            None
        """
        self._test_size = test_size
        self._random_state = random_state
        self._target_var = target_var
        self._shuffle = shuffle

    def split(self, df: pd.DataFrame):
        """
        this split our data into our desired ratio and return the test and train sets
        args:
            df: pd.Dataframe - this will contain or dataframe
        returns:
            none
        """
        x = df.drop(columns=[self._target_var])
        y = df[self._target_var]

        x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=self._test_size, random_state=self._random_state, shuffle=self._shuffle)

        # save all the tests and train data into  folder(splitted_data)
        path = "data/splitted_data/"

        # check if the path exist if not create it
        os.makedirs(os.path.dirname(os.path.join(path,"x_train.csv")), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(path,"x_test.csv")), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(path,"y_train.csv")), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.join(path,"y_test.csv")), exist_ok=True)

        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        # save all the tests and trins into their respective files
        x_train.to_csv(path + 'x_train.csv', index=False)
        x_test.to_csv(path + 'x_test.csv', index=False)
        y_train.to_csv(path + 'y_train.csv', index=False)
        y_test.to_csv(path + 'y_test.csv', index=False)

    
class SelectSplitter():
    def __init__(self, test_size: float, random_state: int, target_var: str, shuffle: bool=True):
        """
        we will initialize our slitter with some values like test_size, etc
        args:
            test_size: float - this will determine the size of the test data set(multiply by 100 to gfet a percentage)
            random_state: int - this will help us get a consitent result when ever we run the program
            target_var: str - this will hold the target column name
            shuffle: bool - this will help shuffle our data before splitting
        returns:
            None
        """
        self._test_size = test_size
        self._random_state = random_state
        self._target_var = target_var
        self._shuffle = shuffle

    def set_splitter(self, splitter: SplittingDataset=SplittingDataset):
        """
        we will change the slitter from the default one to our prefered splitter
        """
        self._splitter = splitter

    def execute_splitter(self, df: pd.DataFrame):
        """
        this will execute our splitter
        args:
            df: pd.Dataframe - this will hold the incoming data
        returns:
            none
        """
        self._splitter = self._splitter(test_size=self._test_size, random_state=self._random_state, shuffle=self._shuffle, target_var=self._target_var)
        self._splitter.split(df)
    

# # example use case
# if __name__ == "__main__":
#     data = numpy.transpose([[1,1,1,12,23],[4,5,6,7,8]])
#     df = pd.DataFrame(data, columns=['num','cat'])
#     splitter = SelectSplitter(test_size=0.2, random_state=42,target_var='cat')
#     splitter.set_splitter(TestTrainSplit)
#     xtr,xte,ytr,yte = splitter.execute_splitter(df=df)
#     print(f"{xtr}\n{xte}\n{ytr}\n{yte}")