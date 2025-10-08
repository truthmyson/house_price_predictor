import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# base class to plot Bivariante graphs
class BivarianteAnlysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, x_column: str, y_column: str):
        """
        we graph 2  columns for visual details
        args:
            df: pd.datafrme - this is the dataframe containing the data
            x_column: str - this column is the x_coordinate for the graph
            y_column: str - this column is the y_coordinate for the graph
        returns:
            none - it returns a graph on the columns
        """
        pass


# a class to graph numerical-numerical columns
class NumericalNumericalBivarianteAnalysis(BivarianteAnlysis):
    def analyze(self, df: pd.DataFrame, x_column: str, y_column: str):
        """
        we graph 2  numerical columns for visual details
        args:
            df: pd.datafrme - this is the dataframe containing the data
            x_column: str - this column is the x_coordinate for the graph
            y_column: str - this column is the y_coordinate for the graph
        returns:
            none - it returns a graph on the columns using scatter diagram
        """
        plt.figure(figsize=(8,6))
        sns.scatterplot(data=df, x=x_column, y=y_column)
        plt.title(f'relationship between {x_column} & {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()


# a class to graph categorical-categorical columns
class CategoricalNumericalBivarianteAnalysis(BivarianteAnlysis):
    def analyze(self, df: pd.DataFrame, x_column: str, y_column: str):
        """
        we graph 2 categorical columns for visual details
        args:
            df: pd.datafrme - this is the dataframe containing the data
            x_column: str - this column is the x_coordinate for the graph
            y_column: str - this column is the y_coordinate for the graph
        returns:
            none - it returns a graph on the columns using boxplot
        """
        plt.figure(figsize=(8,6))
        sns.boxplot(data=df, x=x_column, y=y_column, color="skyblue")
        plt.title(f'relationship between {x_column} & {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()


# automatic Bivariante analyzer selctor 
class SelectorBivarianeAnalyzer():
    def __init__(self, analyzer: BivarianteAnlysis=BivarianteAnlysis):
        """
        we wll set the analyzer to be the base analyser
        args:
            analyzer: BIvarianteAnalyzer - this is the default analyzer
        returns:
            none - we will set the analyzer to be the base analyzer(BIvarianteAnalyzer)
        """
        self._analyzer = analyzer

    def execute_analyzer(self, df: pd.DataFrame, x_column: str, y_column:str, x_col_type: str):
        """
        we will set the preferred analyzer base on the datatype of the column automatically
        args:
            df: pd.Dataframe - this will hold our dataframe for the analyzer
            x_column: str - this will contain the name of the column for the x-axis
            y_column: str - this will contain the name of the column for the y_axis
            x_col_type: str - this will contain the datatype of the x-axis column, which will determine the type of analyzer to use
        returns:
            none - we will check the datatype of x_column and execute the right analyzer
        """
        if x_col_type in ["object", "category"]:
            self._analyzer = CategoricalNumericalBivarianteAnalysis()
            self._analyzer.analyze(df,x_column,y_column)
        else:
            self._analyzer = NumericalNumericalBivarianteAnalysis()
            self._analyzer.analyze(df,x_column,y_column)



# # example use case
# if __name__ == "__main__":
#     analyzer = SelectorBivarianeAnalyzer()
#     data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['Accra', 'Kumasi', 'Tamale'],
#     'Target': [1,0,1]
# }
#     df = pd.DataFrame(data)
#     columns = df.columns
#     y = 'Target'
#     for i in columns:
#         if i != "Target":
#             x_col_type = df[i].dtype
#             analyzer.execute_analyzer(df,i,y,x_col_type)