import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# this is the base class to plot correlation map
class MultivarianteAnalysis(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        """
        this will print a correlation map on the dataset(numerical columns)
        args:
            df: pd.datafrme - this is the dataframe containing the data
        returns:
            none - it returns a correlation map on the data
        """
        pass

# this class will create a correlation heatmap on the data frame(numerical col)
class NumericalMultivarianteAnalysis(MultivarianteAnalysis):
    def analyze(self, df: pd.DataFrame):
        """
        this will print a correlation map on the dataset(numerical columns)
        args:
            df: pd.datafrme - this is the dataframe containing the data
        returns:
            none - it returns a correlation map on the data
        """
        plt.figure(figsize=(12,12))
        _df = df.select_dtypes(include=[float, int])  # Only numeric columns
        sns.heatmap(_df.corr(),annot=True,cmap="coolwarm")
        plt.title("correlation map")
        plt.show()

        # return pair plot
        plt.figure(figsize=(12,12))
        sns.pairplot(df, palette="set2")
        plt.title("Pair plot on the dataset")
        plt.show()


# this class will help us select the analyzer we want
class SelectMUltivarianteAnalyzer():
    def __init__(self, analyzer: MultivarianteAnalysis=MultivarianteAnalysis):
        """
        we wll set the analyzer to be the base analyser
        args:
            analyzer: MultivarianteAnalyzer - this is the default multianalyzer
        returns:
            none - we will set the analyzer to be the base analyzer
        """
        self._analyzer = analyzer

    def set_analyzer(self, analyzer: MultivarianteAnalysis):
        """
        this will allow us to select our prefered analyzer
        args:
            analyzer: MultivarianteAnalyzer = it will store the name of the prefered analyzer provided
        returns:
            none - this will change the default analyzer to our prefered one
        """
        self._analyzer = analyzer()

    def execute_nalyzer(self, df: pd.DataFrame):
        """
        this will execute our prefered multianalyzer on the provided dtaframe
        args:
            df: pd.datafrme - this is the dataframe containing the data we will send to our prefered analyzer
        returns:
            none - it ill call and execute our analyzer on the dataframe providered
        """
        df = df
        self._analyzer.analyze(df)


# # exmple use case
# if __name__ == "__main__":
#     analyzer = SelectMUltivarianteAnalyzer()
#     analyzer.set_analyzer(NumericalMultivarianteAnalysis())
#     data = {
#     'Name': ['Alice', 'Bob', 'Charlie'],
#     'Age': [25, 30, 35],
#     'City': ['Accra', 'Kumasi', 'Tamale'],
#     'Target': [1,0,1]
# }
#     df = pd.DataFrame(data)
#     analyzer.execute_nalyzer(df)