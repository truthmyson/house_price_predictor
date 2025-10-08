import os
import sys
import numpy
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# this is the base class for feature engineering
class FeatureEngineering(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame):
        """
        we will take the data frame and perform feature engineering on the data and save as file
        args:
            df: pd.Dataframe - this contains the dataframe
        returns:
            none
        """ 
        pass

class x_FetureEngineering(FeatureEngineering):
    def __init__(self, col_onehotencode: str=[], col_skewed: str=[], col_outliers: str=[],  method: str=['train','test']):
        """
        we will initialize all the necessary varibles needed for the transformation
        args:
            col_onehotencoding: Str - this si all the columns ee want to perform the one hot transformation on
            col_skewed: Str - this is the coluomns we want correct skewness on
            col_outliers: Str - this are the columns we want to reshape to remov outliers
            method: str - this will determine whether we fit and or just transform
        returns:
            none
        """
        self._col_onehotencode = col_onehotencode
        self._col_skewed = col_skewed
        self._col_outliers = col_outliers
        self._method = method

    def apply_transformation(self, df: pd.DataFrame):
        """
        we will transfor the data 
        args:
            df: pd.Dtaframe - this will hold the dataframe of the data
        returns:
            none
        """
        # we will use power transformer to handle skewness
        # we will use robust transformer to handle outliers
        Skew_tranform = PowerTransformer()
        outlier_transform = RobustScaler()
        onehot_transform = OneHotEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ("remove skewness", Skew_tranform, self._col_skewed),
                ("deal with outliers", outlier_transform, self._col_outliers),
                ("one hot encode cat cols", onehot_transform, self._col_onehotencode)
            ],
            remainder='passthrough'
        )

        path = "data/arr_data"

        if isinstance(df, pd.DataFrame) and self._method == 'train':
            x_train = preprocessor.fit_transform(df)
            # we vrify if the path exist else we will create it
            os.makedirs(os.path.dirname(os.path.join(path, 'x_train.np')), exist_ok=True)
            numpy.save(path + 'x_train.np',x_train)

        elif isinstance(df, pd.DataFrame) and self._method == 'test':
            x_test = preprocessor.transform(df)
            # we vrify if the path exist else we will create it
            os.makedirs(os.path.dirname(os.path.join(path, 'x_test.np')), exist_ok=True)
            numpy.save(path + 'x_test.np',x_train)
            



class y_FetureEngineering(FeatureEngineering):
    def __init__(self, method: str=['train','test']):
        """
        we will initialize the method for or transformation
        args:
            none
        returns:
            none
        """
        self._method = method

    def apply_transformation(self, df: pd.DataFrame):
        """
        we will transfor the data 
        args:
            df: pd.Dtaframe - this will hold the dataframe of the data
        returns:
            none
        """
        # we will use power transformer to handle skewness
        # we will use robust transformer to handle outliers
        Skew_tranform = PowerTransformer()
        outlier_transform = RobustScaler()
        
        # because is a pandans DataFrame we cant use coumnTransformer
        pipline = Pipeline(
            steps=[
                ("remove skewness", Skew_tranform),
                ("deal with outliers", outlier_transform),
            ],
        )

        path = "data/arr_data"

        if isinstance(df, pd.DataFrame) and self._method == 'train':
            y_train = pipline.fit_transform(df)
            # we vrify if the path exist else we will create it
            os.makedirs(os.path.dirname(os.path.join(path, 'y_train.np')), exist_ok=True)
            numpy.save(path + 'y_train.np',y_train)

        elif isinstance(df, pd.DataFrame) and self._method == 'test':
            y_test = pipline.transform(df)
            # we vrify if the path exist else we will create it
            os.makedirs(os.path.dirname(os.path.join(path, 'y_test.np')), exist_ok=True)
            numpy.save(path + 'y_test.np',y_test)
        

# this class helps us to select the preferred style of data engineering
class selectFeatureEngineeringStrtegy():
    def __init__(self, strategy: FeatureEngineering=FeatureEngineering):
        """
        we will initialize our strategy selector with the base feature engineering class
        args:
            strategy: FeatureEngineering - we will set the strategy to the base feature engineering strategy
        returns:
            None - we will jsut set the strategy to the base strategy
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineering):
        """
        we will set the strategy to our preferred strategy
        args:
            strategy: FeatureEngineering - creates an instance of our preferred strategy
        returns:
            None - we will just set the strategy to thepreferred strategy
        """
        self._strategy = strategy

    def execute_strategy(self,df: pd.DataFrame, col_onehotencode: str=[], col_skewed: str=[], col_outliers: str=[],  method: str=['train','test']):
        """
        we will execute the srategy we have set
        args:
            df: pd.Dataframe - this will hold our datafrane
            columns: str - this is a list of all the columns we want perform feature engineering on
        returns:
            pd.Dataframe - returns a pandas dataframe
        """
        if self._strategy == x_FetureEngineering:
            self._strategy = self._strategy(col_onehotencode=col_onehotencode,col_skewed=col_skewed,col_outliers=col_outliers,method=method)
            self._strategy.apply_transformation(df)
        elif self._strategy == y_FetureEngineering:
            self._strategy = self._strategy(method=method)
            self._strategy.apply_transformation(df)
        else:
            raise ValueError('the strategy is invalid..')
    

# # example use case
# if __name__ == "__main__":
#     data = [2,1,1,12,23]
#     df = pd.DataFrame(data)
#     print(df)
#     strategy = selectFeatureEngineeringStrtegy()
#     strategy.set_strategy(y_FetureEngineering)
#     df = strategy.execute_strategy(df=df,method='test')
#     print(df)