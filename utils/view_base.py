import pandas as pd
import numpy as np
from IPython.display import display
import logging
from typing import Union, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicPropLoader:
    """
    A class to load the basic properties of a file for preliminary EDA.
    """

    def __init__(self, df:pd.DataFrame):
        """
        Initialize the BasicPropLoader with the given file (or filepath). 
        If no file is loaded, an existing df should be passed.

        Parameters
        ----------.
        df : pandas.DataFrame, optional
            The dataframe to use instead of loading a file.
        """
        self.df = df

    def __getattr__(self, name:str):
        """
        Delegate attribute access to the underlying DataFrame if not found on BasicPropLoader.

        Raises
        ------
        AttributeError
            If the attribute is not found on both BasicPropLoader and DataFrame.
        """
        if self.df is not None and hasattr(self.df, name):
            return getattr(self.df, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __dir__(self):
        """
        Include DataFrame attributes in directory listing for user convenience.
        """
        attrs = set(super().__dir__()) | set(dir(pd.DataFrame))
        return list(attrs)
    
    def _no_df_error(self):
        if self.df is None:
            raise ValueError("No DataFrame found.")

    def show_summary(self, auto: bool = True, display_df : bool = True,
                     num_cols: List[str] = [], cat_cols: List[str] = []) -> Dict[str, pd.DataFrame]:
        """
        Shows the shape, info, and summary statistics of the data.

        Parameters
        ----------
        auto : bool, default True
            If True, uses df.describe() to show general statistics.
            If False, user must provide `num_cols` and `cat_cols`.
        num_cols : list of str, optional
            List of numeric columns to describe if auto=False.
        cat_cols : list of str, optional
            List of categorical columns to describe if auto=False.

        Returns
        -------
        dict of str -> pd.DataFrame
            A dictionary containing 'shape', 'info', and 'describe' DataFrames.

        Raises
        ------
        ValueError
            If no DataFrame is loaded.
        TypeError
            If auto=False and no `num_cols` or `cat_cols` are provided.
        """
        self._no_df_error()

        shape_df = pd.DataFrame({'Rows': [self.df.shape[0]], 'Columns': [self.df.shape[1]]}, index=['Dataset'])

        non_null = self.df.count()
        dtypes = self.df.dtypes.astype(str)
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / self.df.shape[0]) * 100
        info_df = pd.DataFrame({
            'Non-Null Count': non_null,
            'Dtype': dtypes,
            'Missing Count': missing_counts,
            'Missing %': np.round(missing_pct, 2)
        }).reset_index(names='Columns')

        if auto:
            desc = self.df.describe(include='all')
        else:
            if not num_cols or not cat_cols:
                raise TypeError('If auto=False, please provide `num_cols` and `cat_cols`.')
            num_desc = self.df[num_cols].describe()
            cat_desc = self.df[cat_cols].describe()
            desc = pd.concat([num_desc, cat_desc], axis=1)

        if display_df:
            display(shape_df)
            display(info_df)
            display(desc.T)
        else:
            return {'shape': shape_df, 'info': info_df, 'describe': desc.T.reset_index(names='Columns')}
        
    def check_duplicates(self, **kwargs) -> Dict[str, int]:
        """
        Checks for duplicate rows and duplicate columns.

        Returns
        -------
        dict of str -> int
            {
                'duplicate_rows': <count_of_duplicate_rows>,
                'duplicate_column_names': <count_of_duplicate_column_names>
            }

        Raises
        ------
        ValueError
            If no DataFrame is found.
        """

        self._no_df_error()
        dup_rows = self.df.duplicated(**kwargs).sum()

        # Check duplicate columns
        #dup_col_names = self.df.columns.duplicated().sum()
        duplicates_list = []
        columns_hashes = {col:hash(tuple(self.df[col])) for col in self.df.columns}
        hash_groups = {}

        for col, col_hash in columns_hashes.items():
            if col_hash not in hash_groups:
                hash_groups[col_hash] = [col]
            else:
                hash_groups[col_hash].append(col)

        for group in hash_groups.values():
            if len(group) > 1:
                original_col = group[0]
                for duplicate_col in group[1:]:
                    duplicates_list.append({'Original Columns': original_col, 'Duplicate Columns': duplicate_col})

        if not duplicates_list:
            duplicates_list = [{"Duplicate Columns": 0}]

        result = {
                'Duplicate Rows': dup_rows,
                'Duplicate Column Names': duplicates_list
        }

        return result
    
    def check_missing_data(self) -> pd.DataFrame:
        """
        Check for the number and percentage of missing values in each column of the DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame with 'missing_count' and 'missing_percentage' for each column.

        Raises
        ------
        ValueError
            If no DataFrame is found.
        """
        self._no_df_error()

        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df)) * 100
        missing_summary = pd.DataFrame({
            'Missing Counts': missing_counts,
            'Missing %': missing_pct
        })

        #rows_with_missing = self.df.isnull().any(axis=1).sum()
        return missing_summary.reset_index(names='Columns')
    
    def check_value_counts(self, cols: Union[str, List[str]] = 'all', transpose: bool = False) -> pd.DataFrame:
        """
        Show the number of unique values and their counts for given columns.

        Parameters
        ----------
        cols : str or list of str, default 'all'
            - 'all': consider all columns in the DataFrame.
            - list: consider only the specified columns.
        transpose : bool, default False
            If True, transpose the final DataFrame before returning.

        Returns
        -------
        pd.DataFrame
            A DataFrame summarizing unique values and distributions for each column.

        Raises
        ------
        ValueError
            If the DataFrame is not loaded.
        TypeError
            If `cols` is not 'all' or a list.
        """
        self._no_df_error()

        if cols == 'all':
            columns = self.df.columns
        elif isinstance(cols, list) and len(cols) > 0:
            columns = cols
        else:
            raise TypeError("`cols` must be 'all' or a list of column names.")

        uniques = (self.df.nunique()
                   .rename_axis('Columns')
                   .reset_index(name='Unique Values'))

        # Build a concatenated distribution DataFrame
        dist_frames = []
        for col in columns:
            count_df = (self.df.value_counts(subset=[col])
                        .rename('Count')
                        .to_frame()
                        .assign(Distribution=lambda x:
                                np.round((x['Count'] / x['Count'].sum()) * 100, 2))
                        .reset_index()
                        .assign(Columns=col)
                        .rename(columns={col: 'Values'}))
            dist_frames.append(count_df)

        value_dist = pd.concat(dist_frames, ignore_index=True)

        fdf = (uniques.merge(value_dist, on='Columns')
               .set_index(['Columns', 'Unique Values', 'Values']))
        fdf.rename(columns={'Distribution':'Distribution, %'}, inplace=True)

        if transpose:
            fdf = fdf.T

        return fdf