import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pointbiserialr
import plotly.graph_objects as go
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
                 categorical_cols: Optional[List[str]]=None,
                 category_threshold: Optional[int]=10,
                 **kwargs):
        self.df = df
        self.kwargs = kwargs or {}

        self.continuous_cols = self.kwargs.get('continuous_cols', continuous_cols)
        self.categorical_cols = self.kwargs.get('categorical_cols', categorical_cols)
        self.category_threshold = self.kwargs.get('categorical_threshold', category_threshold)
        self.columns = self.df.columns

        self._auto_assign_column_types()
        self._validate_columns()
        
        # Initialize caches
        self._continuous_continuous_cache = {}
        self._continuous_categorical_cache = None
        self._categorical_categorical_cache = None
        self._mutual_information_cache = None
        
    def _validate_columns(self) -> None:
        """
        Validate that all specified continuous and categorical columns exist in the DataFrame.
        Raise a ValueError if any columns are not found.
        """
        missing_continuous = set(self.continuous_cols) - set(self.columns)
        missing_categorical = set(self.categorical_cols) - set(self.columns)

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
        
    @staticmethod
    def infer_column_types(df: pd.DataFrame , category_threshold: int = 10) -> Dict[str, List]:
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
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > category_threshold:
                continuous.append(col)
            else:
                categorical.append(col)

        col_types = {'continuous': continuous, 'categorical': categorical}

        logger.info("\n\tThreshold for Categorical Columns: %d unique values", category_threshold)
        return col_types
    
    def _auto_assign_column_types(self):
        """
        Automatically infer column types
        """
        if self.continuous_cols and self.categorical_cols:
            return 

        if self.categorical_cols and self.continuous_cols is None:
            self.continuous_cols = list(
                set(self.columns).difference(set(self.categorical_cols))
            )
            return

        if self.continuous_cols and self.categorical_cols is None:
            self.categorical_cols = list(
                set(self.columns).difference(set(self.continuous_cols))
            )
            return
        
        if self.continuous_cols is None and self.categorical_cols is None:
            col_types = CorrelationViewer.infer_column_types(
                df=self.df, category_threshold=self.category_threshold
            )
            self.continuous_cols = col_types['continuous']
            self.categorical_cols = col_types['categorical']
            return

    def _correlation_ratio(self, categories: pd.Series, values: pd.Series) -> float:
        """
        Calculate correlation ratio (eta) or point-biserial correlation 
        depending on the number of unique categories.

        If the categorical variable has exactly two categories, use point-biserial correlation.
        Otherwise, use the correlation ratio (eta).

        Parameters
        ----------
        categories : pd.Series
            Categorical variable.
        values : pd.Series
            Continuous variable.

        Returns
        -------
        float
            If two categories: point-biserial correlation value.
            If more than two categories: correlation ratio (eta) value.
            Returns NaN if insufficient data.
        """
        mask = ~pd.isnull(categories) & ~pd.isnull(values)
        if mask.sum() == 0:
            return np.nan

        categories = categories[mask]
        values = values[mask]

        unique_cats = pd.unique(categories)
        if len(unique_cats) == 0:
            return np.nan

        # If exactly two categories, use point-biserial correlation
        if len(unique_cats) == 2:
            cat_map = {unique_cats[0]: 0, unique_cats[1]: 1}
            binary_cats = categories.map(cat_map)
            if binary_cats.nunique() < 2:
                return np.nan
            r, _ = pointbiserialr(binary_cats.values, values.values)
            return r

        # Otherwise, use correlation ratio (eta)
        category_groups = {cat: values[categories == cat] for cat in unique_cats if len(values[categories == cat]) > 0}
        if len(category_groups) == 0:
            return np.nan

        overall_mean = values.mean()
        ss_between = sum([len(vals) * (vals.mean() - overall_mean)**2 for vals in category_groups.values()])
        ss_total = ((values - overall_mean)**2).sum()

        if ss_total == 0:
            return np.nan
        eta = np.sqrt(ss_between / ss_total)
        return eta

    def _cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate Cramer's V for categorical-categorical association.

        Parameters
        ----------
        x, y : pd.Series
            Categorical variables.

        Returns
        -------
        float
            Cramer's V value. Returns NaN if not computable.
        """
        # Drop missing values
        mask = ~pd.isnull(x) & ~pd.isnull(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0 or len(y) == 0:
            return np.nan

        contingency = pd.crosstab(x, y)
        if contingency.size == 0:
            return np.nan

        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        if n == 0:
            return np.nan
        r, k = contingency.shape
        if min(r, k) == 1:
            return np.nan
        return np.sqrt((chi2 / n) / (min(r, k) - 1))
    
    def continuous_continuous_corr(self, method: str='pearson') -> pd.DataFrame:
        """
        Compute Pearson correlation among continuous columns.
        
        Parameters
        ----------
        method: {'pearson', 'spearman', 'kendall'}, default 'pearson'

        Returns
        -------
        pd.DataFrame
            Correlation matrix (continuous vs continuous).
            Returns an empty DataFrame if no continuous columns are available.
        """
        if not self.continuous_cols:
            return pd.DataFrame()
        
        # Use caching to avoid recomputation
        if method not in self._continuous_continuous_cache.keys():
            self._continuous_continuous_cache[method] = self.df[self.continuous_cols].corr(method=method)
        
        return self._continuous_continuous_cache[method]

    def continuous_categorical_corr(self) -> pd.DataFrame:
        """
        Compute correlation (point-biserial or correlation ratio) for continuous vs categorical columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with correlation for each (categorical, continuous) pair.
            Columns: ['Category', 'Continuous', 'Correlation']
            Sorted by correlation (descending).
        """
        if self._continuous_categorical_cache is not None:
            return self._continuous_categorical_cache
        
        rows = []
        for cat_col in self.categorical_cols:
            for cont_col in self.continuous_cols:
                eta = self._correlation_ratio(self.df[cat_col], self.df[cont_col])
                rows.append((cat_col, cont_col, eta))

        df_corr = pd.DataFrame(rows, columns=['Category', 'Continuous', 'Correlation'])
        df_corr = df_corr.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
        self._continuous_categorical_cache = df_corr
        return df_corr
    
    def categorical_categorical_corr(self) -> pd.DataFrame:
        """
        Compute Cramer's V for categorical vs categorical columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with Cramer's V for each pair of categorical columns.
            Columns: ['Category_1', 'Category_2', 'CramersV']
            Sorted by CramersV (descending).
        """
        if self._categorical_categorical_cache is not None:
            return self._categorical_categorical_cache
        
        rows = []
        cat_count = len(self.categorical_cols)
        for i in range(cat_count):
            for j in range(i+1, cat_count):
                col1 = self.categorical_cols[i]
                col2 = self.categorical_cols[j]
                cv = self._cramers_v(self.df[col1], self.df[col2])
                rows.append((col1, col2, cv))

        df_corr = pd.DataFrame(rows, columns=['Category_1', 'Category_2', 'CramersV'])
        df_corr= df_corr.sort_values(by='CramersV', ascending=False).reset_index(drop=True)
        self._categorical_categorical_cache = df_corr
        return df_corr
    
    def combined_correlations(self, cc_method:str = 'pearson') -> Dict[str, pd.DataFrame]:
        """
        Combine all correlation computations into a single dictionary.

        Parameters:
        -----------
        cc_method : {'pearson', 'spearman', 'kendall'}, default 'pearson'

        Returns
        -------
        dict
            A dictionary with keys:
            - 'continuous_continuous': DataFrame of Pearson correlations
            - 'continuous_categorical': DataFrame of correlation ratio/point-biserial
            - 'categorical_categorical': DataFrame of Cramer's V values
        """
        logger.info("Method of correlation used : %s", cc_method)
        return {
            'continuous_continuous': self.continuous_continuous_corr(method=cc_method),
            'continuous_categorical': self.continuous_categorical_corr(),
            'categorical_categorical': self.categorical_categorical_corr()
        }
    
    def partial_correlation(self, x_col: str, y_col: str, control_cols: List[str]) -> float:
        """
        Compute the partial correlation between x_col and y_col controlling for control_cols.

        Parameters:
        -----------
        x_col:str
            Target variable 1.
        y_col:str
            Target variable 2.
        control_cols: list of str
            Controlling variable/s.

        Returns:
        --------
        float
            Correlation of residuals
        """

        all_cols = [x_col, y_col] + control_cols
        df_sub = self.df[all_cols].dropna()

        if df_sub[x_col].nunique() < 2 or df_sub[y_col].nunique() < 2:
            return np.nan
        
        X_ctrl = df_sub[control_cols].values
        X_ctrl = np.column_stack([X_ctrl, np.ones(X_ctrl.shape[0])])  # Add intercept

        model_x = LinearRegression().fit(X_ctrl, df_sub[x_col])
        model_y = LinearRegression().fit(X_ctrl, df_sub[y_col])

        resid_x = df_sub[x_col] - model_x.predict(X_ctrl)
        resid_y = df_sub[y_col] - model_y.predict(X_ctrl)

        # Compute correlation of residuals
        return pd.Series(resid_x).corr(pd.Series(resid_y))
    
    def mutual_information_continuous(self) -> pd.DataFrame:
        """
        Compute mutual information between continuous features.
        Uses sklearn's `mutual_info_regression`.

        Returns
        -------
        pd.DataFrame
            A symmetric DataFrame of MI values.
        """
        if self._mutual_information_cache is not None:
            return self._mutual_information_cache
        
        if len(self.continuous_cols) < 2:
            return pd.DataFrame()

        cols = self.continuous_cols
        n = len(cols)
        mi_matrix = np.zeros((n, n))

        data = self.df[cols].dropna()
        for i in range(n):
            for j in range(i+1, n):
                x = data[cols[i]].values.reshape(-1, 1)
                y = data[cols[j]].values
                mi = mutual_info_regression(x, y, discrete_features=False)
                mi_val = mi[0] if len(mi) > 0 else np.nan
                mi_matrix[i, j] = mi_val
                mi_matrix[j, i] = mi_val
            mi_matrix[i, i] = 1.0

        mi_df = pd.DataFrame(mi_matrix, index=cols, columns=cols)
        self._mutual_information_cache = mi_df

        return mi_df
    
    def plot_continuous_continuous_corr(self, method='pearson') -> Optional[go.Figure]:
        """
        Plot a heatmap for continuous vs continuous correlation.

        Parameters
        ----------
        method : {'pearson', 'spearman', 'kendall'}, default 'pearson'

        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object of the heatmap, or None if no continuous columns are available.
        """
        corr = self.continuous_continuous_corr(method=method)
        if corr.empty:
            # logger.info("No continuous columns available to plot.")
            return None
        values = corr.values
        # Create a heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='rdylgn',
                zmin=-1,  # Minimum correlation value
                zmax=1,   # Maximum correlation value
                colorbar=dict(title='Correlation', tickvals=[-1, 0, 1]),
                showscale=True,
                texttemplate='%{text}',
                text=np.round(values, 2).astype(str)
            )
        )

        # Update layout for better appearance
        fig.update_layout(
            title=f"Continuous vs Continuous Correlation ({method})",
            xaxis=dict(title="Variables"),
            yaxis=dict(title="Variables", autorange="reversed"),  # Reverse y-axis for a proper heatmap look
            template="simple_white",
        )

        return fig
    
    def plot_mutual_inf(self) -> Optional[go.Figure]:
        """
        Plot a heatmap for continuous vs continuous correlation (Pearson).

        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object of the heatmap, or None if no continuous columns are available.
        """
        corr = self.mutual_information_continuous()
        if corr.empty:
            # logger.info("No continuous columns available to plot.")
            return None
        values = corr.values

        min_val = np.min(values)
        max_val = np.max(values)
        med_val = np.median(values)
        # Create a heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='rdylgn',
                zmin= min_val,  # Minimum correlation value
                zmax= max_val,   # Maximum correlation value
                colorbar=dict(title='Mutual Information', tickvals=[np.round(min_val, 2), np.round(med_val, 2), np.round(max_val, 2)]),
                showscale=True,
                texttemplate='%{text}',
                text=np.round(values, 2).astype(str)
            )
        )

        # Update layout for better appearance
        fig.update_layout(
            title="Mutual Information Heatmap",
            xaxis=dict(title="Variables"),
            yaxis=dict(title="Variables", autorange="reversed"),  # Reverse y-axis for a proper heatmap look
            template="simple_white",
        )

        return fig

    def plot_pairwise_correlations(self, 
                                   df_corr: pd.DataFrame,
                                   category_cols: Tuple[str, ...],
                                   val_col: str, 
                                   sort: bool=True,
                                   title: str = None) -> Optional[go.Figure]:
        """
        Helper method to plot horizontal bar charts for pairwise correlations.

        Parameters
        ----------
        df_corr : pd.DataFrame
            DataFrame containing pairwise correlation data.
        category_cols : tuple of str
            Column names in df_corr that represent the feature pair. Must be 1 or 2 columns.
        val_col : str
            The column name of the correlation value in df_corr.
        sort : bool, optional
            If True, sort by the correlation value.
        title : str, optional
            The title of the chart.

        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object of the bar plot, or None if df_corr is empty.
        """
        if df_corr.empty:
            # logger.info("No correlation data available to plot.")
            return None

        df_corr = df_corr.copy()
        # Create a 'Pair' column for plotting
        if len(category_cols) == 2:
            df_corr['Pair'] = df_corr[category_cols[0]] + ' vs ' + df_corr[category_cols[1]]
        elif len(category_cols) == 1:
            df_corr['Pair'] = df_corr[category_cols[0]]
        else:
            raise ValueError("category_cols must be a tuple of length 1 or 2.")

        if sort:
            df_corr = df_corr.sort_values(by=val_col, ascending=True)

        values = np.round(df_corr[val_col], 2)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=values, y=df_corr['Pair'], orientation='h', text=values, textposition='outside',
                marker=dict(color=values, cmin=0, cmax=1, colorscale=[(0, 'gray'), (1, 'green')], showscale=True)
            )
        )
        fig.update_layout(title=f'Pairwise Correlation ({title})',
                          xaxis_title='Correlation',
                          yaxis_title='Pair')

        return fig

    def plot_continuous_categorical_corr(self) -> Optional[go.Figure]:
        """
        Plot a horizontal bar chart for continuous vs categorical correlations.

        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object if data is available, otherwise None.
        """
        df_corr = self.continuous_categorical_corr()
        return self.plot_pairwise_correlations(
            df_corr=df_corr,
            category_cols=('Category', 'Continuous'),
            val_col='Correlation',
            title='Continuous vs Categorical'
        )

    def plot_categorical_categorical_corr(self) -> Optional[go.Figure]:
        """
        Plot a horizontal bar chart for categorical vs categorical correlations (Cramer's V).

        Returns
        -------
        matplotlib.axes.Axes or None
            Axes object if data is available, otherwise None.
        """
        df_corr = self.categorical_categorical_corr()
        return self.plot_pairwise_correlations(
            df_corr=df_corr,
            category_cols=('Category_1', 'Category_2'),
            val_col='CramersV',
            title='Categorical vs Categorical (Cramer\'s V)',
        )