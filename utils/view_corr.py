import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union

from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression

# Uncomment to enable logging for debugging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class  CorrelationViewer:
    """
    A class that shows the correlations between continuous and categorical features in a DataFrame.
    It provides methods for computing Pearson correlation (continuous-continuous), correlation ratio/point-biserial (continuous-categorical), 
    and Cramer's V (categorical-categorical).
    Additionally, it offers plotting utilities for visualizing these correlations.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    continuous_cols : list of str
        List of column names in df that are continuous.
    categorical_cols : list of str
        List of column names in df that are categorical.

    Attributes
    ----------
    df : pd.DataFrame
        The input DataFrame.
    continuous_cols : list of str
        Continuous columns.
    categorical_cols : list of str
        Categorical columns.
    """

    def __init__(self, 
                 df: pd.DataFrame, 
                 continuous_cols: Optional[List[str]]=None, 
                 categorical_cols: Optional[List[str]]=None):
        self.df = df.copy()
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols

        # Validate columns
        self._validate_columns()

        # Initialize caches
        self._continuous_continuous_cache = {}
        self._continuous_categorical_cache = None
        self._categorical_categorical_cache = None

    def _validate_columns(self) -> None:
        """
        Validate that all specified continuous and categorical columns exist in the DataFrame.
        Raise a ValueError if any columns are not found.
        """
        missing_continuous = set(self.continuous_cols) - set(self.df.columns)
        missing_categorical = set(self.categorical_cols) - set(self.df.columns)

        if missing_continuous or missing_categorical:
            err_msg = []
            if missing_continuous:
                err_msg.append(f"Missing continuous columns: {missing_continuous}")
            if missing_categorical:
                err_msg.append(f"Missing categorical columns: {missing_categorical}")
            raise ValueError(" ".join(err_msg))
        
        overlap = set(self.continuous_cols).intersection(self.categorical_cols)
        if overlap:
            raise ValueError(f"The following columns are listed as both continuous and categorical: {overlap}")
        
        if self.continuous_cols or self.categorical_cols:
            raise ValueError("Set both `continuous_cols` and `categorical_cols` to `None` to auto infer column types")
        
    @staticmethod
    def infer_column_types(df: pd.DataFrame , categorical_threshold: int = 10) -> Dict[str, List]:
        """
        Infer which columns are continuous and which are categorical based on unique counts and dtype.

        Parameters
        -----------
        df : pd.DataFrame

        Returns
        -------
        dict of list
            Dictionary of list of column types.
        """
        continuous = []
        categorical = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > categorical_threshold:
                continuous.append(col)
            else:
                categorical.append(col)

        col_types = {'continuous': continuous, 'categorical': categorical}

        logger.info("\n\tThreshold for Categorical Columns: %d unique values", categorical_threshold)
        return col_types
    
    def _auto_assign_column_types(self):
        
        
        col_types = self.infer_column_types(df=self.df)
        self.continuous_cols = col_types['continuous']
        self.categorical_cols = col_types['categorical']

