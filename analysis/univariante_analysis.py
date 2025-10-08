import pandas as pd
import numpy
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# base class to plot univariante graphs
class UnivarianteAnlysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, column):
        """
        we graph all our columns for visual details
        args:
            df: pd.datafrme - this is the dataframe containing the data
            column: str - this is the column we would be visualizing
        returns:
            none - it returns a graph on the column only
        """
        pass


# a class to graph numerical columns
class NumericalUnivarianteAnalysis(UnivarianteAnlysis):
    def analyze(self, df: pd.DataFrame, column: str):
        """
        we graph a numerical column for visual details
        args:
            df: pd.datafrme - this is the dataframe containing the data
            column: str - this is the column we would be visualizing
        returns:
            none - it returns a graph on the column only using histogram
        """
        print(f"{column} skewness : {df[column].skew()} \n")
        plt.figure(figsize=(12,12))
        sns.histplot(df, x=column, bins=30, color='blue', kde=True)
        plt.title(f'distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


# a class to graph categorical columns
class CategoricalUNivarianteAnalysis(UnivarianteAnlysis):
    def analyze(self, df: pd.DataFrame, column: str):
        """
        we graph a categorical column for visual details
        args:
            df: pd.datafrme - this is the dataframe containing the data
            column: str - this is the column we would be visualizing
        returns:
            none - it returns a graph on the column only using countplot
        """
        plt.figure(figsize=(12,12))
        sns.countplot(data=df, x=column, color="skyblue")
        plt.title(f"Distribution for {column}")
        plt.xlabel(column)
        plt.ylabel("frequency")
        plt.show()


# automatic univariante analyzer selctor 
class SelectorUnivarianeAnalyzer():
    def __init__(self, analyzer: UnivarianteAnlysis=UnivarianteAnlysis):
        """
        we wll set the analyzer to be the base analyser
        args:
            analyzer: UnivarianteAnlysis - this is the default analyzer
        returns:
            none - we will set the analyzer to be the base analyzer(UnivarianteAnlysis)
        """
        self._analyzer = analyzer

    def execute_analyzer(self, df: pd.DataFrame, column: str, col_type: str):
        """
        we will set the preferred analyzer base on the datatype of the column automatically
        args:
            df: pd.Dataframe - this will hold our dataframe for the analyzer
            column: str - this will contain the name of the column
            col_type: str - this will contain the datatype of the column, which will determine the type of analyzer to use
        returns:
            none - we will check the datatype of column and execute the right analyzer
        """
        if col_type in ["object", "category"]:
            self._analyzer = CategoricalUNivarianteAnalysis()
            self._analyzer.analyze(df,column)
        else:
            self._analyzer = NumericalUnivarianteAnalysis()
            self._analyzer.analyze(df,column)



# # example use case
# if __name__ == "__main__":
#     analyzer = SelectorUnivarianeAnalyzer()
#     data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['Accra', 'Kumasi', 'Tamale']
# }
#     df = pd.DataFrame(data)
#     columns = df.columns
#     for i in columns:
#         col_type = df[i].dtype
#         analyzer.execute_analyzer(df,i,col_type)