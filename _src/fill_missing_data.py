import numpy
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer

# a base class to handle missing values
class HandleMissigingValues(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        this handles all missing values
        args:
            df: pd.Dataframe - this contains the dataframe
        returns:
            pd.Dataframe - returns a pandas dataframe
        """
        pass

# this class removes all rows with missing values
class RemoveMissingValues(HandleMissigingValues):
    def __init__(self, axis: int=0):
        """
        we set our threshold to drop rows with missing values
        args:
            axis: int - this decides where we want to drop row wise (0) or column wise (1)
        returns:
            none - we will set the threshold and axis
        """
        self._axis = int(axis)

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        we will drop all the missing values if the missing_data_threshold is less than self._threshold
        args:
            df: pd.Dataframe - it holds the dataframe
        returns:
            pd.Dataframe - it returns a pandas dataframe
        """
        df = df.dropna(axis=self._axis)
        
        return df
    

# this will fill missing values for numerical columns
class FillNumericColumnsMissingValues(HandleMissigingValues):
    def __init__(self, strategy: str="mean", columns: str=[]):
        """
        we will set the strategy to fill numeric columns
        args:
            strategy: str - this determines the strategy we will use to fill in the missing values
            columns: str - ts is a list of all the columns we will want to fill its missing values
        returns:
            none - we will set the strategy to a default strategy and columns to an empty list
        """
        self._strategy = strategy
        self._columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        we will fill all the missing values
        args:
            df: pd.Dataframe - this contains the dataframe 
        returns:
            pd.Dataframe - it returns a pandas datframe
        """
        imputer = SimpleImputer(strategy=self._strategy)

        for column in self._columns:
            df[column] = imputer.fit_transform(df[[column]])

        return df
    

# this class fills missing values for categorical columns
class FillingCategoricalColumnsMissingValues(HandleMissigingValues):
    def __init__(self, strategy: str=None, columns: str=[]):
        """
        we will set the strategy to fill numeric columns
        args:
            strategy: str - this determines the strategy we will use to fill in the missing values
            columns: str - ts is a list of all the columns we will want to fill its missing values
        returns:
            none - we will set the strategy to a default strategy and columns to an empty list
        """
        self._strategy = str(strategy)
        self._columns = columns

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        we will fill all the missing values
        args:
            df: pd.Dataframe - this contains the dataframe 
        returns:
            pd.Dataframe - it returns a pandas datframe
        """
        imputer = SimpleImputer(strategy=self._strategy)
        
        for column in self._columns:
            df[column] = imputer.fit_transform(df[[column]]).ravel()
            
        return df
    

# this class will select the right filler
class selectFiller():
    def __init__(self, handler: HandleMissigingValues=HandleMissigingValues):
        """
        we will set our handler to a default handler
        args:
            handler: HandleMissingVales - this will set our handler to the 'HandleMissigingValues'
        returns:
            none - we will set the handler to the base handler
        """
        self._handler = handler

    def execute_handler(self, df: pd.DataFrame, method: str=['fill','drop'], axis: int=0, n_strategy: str=['mean','median'],c_strategy: str='most_frequent', columns: str=[]):
        if method == 'drop':
            self._handler = RemoveMissingValues(axis=axis)
            df = self._handler.handle(df)
            return df
        elif method == 'fill':
            for column in columns:
                col_type = df[column].dtype
                if col_type == object:
                    self._handler = FillingCategoricalColumnsMissingValues(strategy=c_strategy,columns=[column])
                    df = self._handler.handle(df)
                else:
                    self._handler = FillNumericColumnsMissingValues(strategy=n_strategy,columns=[column])
                    df = self._handler.handle(df)
            return df
        else:
            raise ValueError("Method unknown....")
        

# # example use case
# if __name__ == "__main__":
#     data = {
#     'Name': [None, None, 'Charlie','Kofi','Ama', 'Ama'],
#     'Age': [None, 30, 40, 20, 23,None],
#     'City': ['Accra', 'Kumasi', None, 'Dodowa', 'Accra', 'Accra'],
#     'Target': [1,0,1,None,0,1]
# }
#     df = pd.DataFrame(pd.read_csv('train.csv'))
#     print(df.info())
#     columns = df.columns.tolist()

#     filler = selectFiller()
#     df = filler.execute_handler(df=df,method='drop',columns=columns,c_strategy='most_frequent',n_strategy='mean')
#     df.info()
