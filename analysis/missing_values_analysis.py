import pandas as pd
from abc import ABC, abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt

# create base class for missing data
class MissingValues(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        we will analyze our date for missing values
        args:
            df: pd.Dataframe - this holds the dataframe containg our data
        returns:
            none - it will display a summary and graph plotting on missing values for each columns
        """
        self.identify_missing_values(df)
        self.visualiZe_missing_values(df)

    # a function to identify allmissing values column wise
    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame):
        """
        we will identify and summarize all missing values
        args:
            df: pd.Dataframe - this holds the dataframe containg our data
        returns:
            none - it will print a summary on missing values
        """
        pass

    # a function to visualize the missing values
    @abstractmethod
    def visualiZe_missing_values(self, pd: pd.DataFrame):
        """
        we will visualize all missing values
        args:
            df: pd.Dataframe - this holds the dataframe containg our data
        returns:
            none - it will visualize a summary on missing values
        """
        pass



# create a class to identify and visualize the missing values
class MissingValuesAnalysis(MissingValues):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        we will identify and summarize all missing values
        args:
            df: pd.Dataframe - this holds the dataframe containg our data
        returns:
            none - it will print a summary on missing values
        """
        df = df
        print('Missing values count column wise...')
        print(df.isnull().sum(), '\n')
        print('Duplicates count in the dataset')
        print(df.duplicated().sum(), '\n')

    def visualiZe_missing_values(self, df: pd.DataFrame):
        """
        we will visualize all missing values
        args:
            df: pd.Dataframe - this holds the dataframe containg our data
        returns:
            none - it will visualize a summary on missing values using heatmap
        """
        missing_values = df.isnull() # change to a boolean dataframe

        plt.figure(figsize=(12,12))
        sns.heatmap(missing_values, cmap='viridis', cbar=False)
        plt.title('Visualizing Missing Values')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
        


# create a function to select ur prefared missing values analyzer
class MissingValueAnalyzer():
    def __init__(self, analyzer: MissingValues=MissingValues):
        """
        we will set our analyzer to the default analyzer(MissingValues)
        args:
            analyzer: MissingValues - this willset a default analyzer to be MissingValues
        returns:
            none - it initializes the default analyzer 
        """
        self._analyzer = analyzer

    def set_analyzer(self, analyzer: MissingValues):
        """
        we will set our preferred analyzer
        args:
            analyzer: MissingVlues - this will set our new analyzer
        returns:
            none - it sets the analyzer to our prefeerred analyzer
        """
        self._analyzer = analyzer()

    def execute_analyzer(self, df: pd.DataFrame):
        """
        we will execute our analyzer
        args:
            df: pd.Dataframe - this contains our dataframe for our analyzer
        returns:
            none - it executes the analyzer
        """
        self._analyzer.analyze(df)



# # example use case
# if __name__ == "__main__":
#     analyzer = MissingValueAnalyzer()
#     analyzer.set_analyzer(MissingValuesAnalysis())
#     data = {
#         'Name': ['Alice', None, 'Charlie'],
#         'Age': [None, None, 35],
#         'City': ['Accra', 'Kumasi', 'Tamale']
#     }
#     missing_df = pd.DataFrame(data)
#     analyzer.execute_analyzer(missing_df)
