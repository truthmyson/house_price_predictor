import pandas as pd
from abc import ABC, abstractmethod


# we will create a base dataframe inspector
class DataInspector(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        this is the base inspector function
        args:
            df: pd.Dataframe - this holds the dataframe containg our data
        returns:
            none - it will return a description on the dataframe(numeriacla and categorical cloumns)
        """
        pass


# create a strategy class to summarise the data
class SummaryDataInspectionStrategy(DataInspector):
    def inspect(self, df: pd.DataFrame):
        """
        this will give us a summary of the data on both numerical and categorical columns
        args:
            df: pd.Dataframe - this contains our data
        returns:
            none - it displays summary on the dataframe 
        """
        print('data summary')
        print(df.info(), '\n')
        print('Description on numeric columns....')
        print(df.describe(), '\n')

        print('Description on categorical columns...')
        print(df.describe(include=['object','category']), '\n')



# we will use this class to select our data strategy inspector
class DataInspectorSelector():
    def __init__(self, strategy: DataInspector=DataInspector):
        """
        we will set our strategy to the default strategy(DataInspector)
        args:
            strategy: DataInspector - this willset a default strategy to be DataInspector
        returns:
            none - it initializes the default strategy 
        """
        self._strategy = strategy
        

    def set_strategy(self, strategy: DataInspector):
        """
        we will set our preferred strategy
        args:
            strategy: DataInspector - this will set our new strategy
        returns:
            none - it sets the strategy to our prefeerred strategy
        """
        self._strategy = strategy()


    def execute_stratgy(self, df: pd.DataFrame):
        """
        we will execute our strategy
        args:
            df: pd.Dataframe - this contains our dataframe for our strategy
        returns:
            none - it executes the strategy
        """
        self._strategy.inspect(df)




# # example use case
# if __name__ == "__main__":
#     strategy = DataInspectorSelector()
#     strategy.set_strategy(SummaryDataInspectionStrategy())
#     data = {
#         'Name': ['Alice', 'Bob', 'Charlie'],
#         'Age': [25, 30, 35],
#         'City': ['Accra', 'Kumasi', 'Tamale']
#     }
#     df = pd.DataFrame(data)
#     strategy.execute_stratgy(df)