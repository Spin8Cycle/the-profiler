import os
from typing import Optional, List

import pandas as pd

class DataLoader:
    """
    A class for loading data dynamically.
    """

    def __init__(self, file: Optional[str]=None, file_repo: Optional[str]=None, df: Optional[pd.DataFrame]=None, **kwargs):
        """
        Initialize the DataLoader with the given file (or filepath). 
        If no file is loaded, an existing df should be passed.

        Parameters
        ----------
        file : str
            The filename or filepath of the data to load.
        file_repo : str, optional
            The repository of data to work (if working with multiple data).
        df : pandas.DataFrame, optional
            The dataframe to use instead of loading a file.
        kwargs
            Additional keyword arguments passed to `_read_file()`.
        """
        if file and df is not None:
            raise ValueError(
                "You cannot pass both 'file' and 'df'. "
                "Please provide only one of these arguments."
            )
        if not file and df is None:
            raise ValueError(
                "You must provide either a 'file' or a 'df'."
            )
        
        self.file = file
        self.file_repo = file_repo
        self.file_extension = ""
        self.df = df
        if file:
            self.file_extension = os.path.splitext(file)[1]
            if not file_repo:
                self.file_repo = os.path.dirname(file)
            self._read_file(**kwargs)

    def __getattr__(self, name:str):
        """
        Delegate attribute access to the underlying DataFrame if not found on DataLoader.

        Raises
        ------
        AttributeError
            If the attribute is not found on both DataLoader and DataFrame.
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
    
    def _read_file(self, **kwargs) -> pd.DataFrame:
        """
        Infer the file type based on the extension and read it into a pandas DataFrame.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the underlying pandas reader function.

        Returns
        -------
        pd.DataFrame
            The DataFrame read from the file.

        Raises
        ------
        ValueError
            If the file cannot be read.
        """

        if self.file is None:
            raise ValueError("No filepath provided.")
        ext = self.file_extension.lower()
        if ext in ['.xls', '.xlsx', '.xlsm']:
            self.df = pd.read_excel(self.file, **kwargs)
        elif ext == '.csv':
            self.df = pd.read_csv(self.file, **kwargs)
        elif ext == '.json':
            self.df = pd.read_json(self.file, **kwargs)
        else:
            self.df = self._try_all_formats(**kwargs)
        return self.df
    
    def _try_all_formats(self, **kwargs) -> pd.DataFrame:
        tried = []
        for func, fmt in [(self._read_excel, 'Excel'), (self._read_csv, 'CSV'), (self._read_json, 'JSON')]:
            try:
                return func(**kwargs)
            except Exception:
                tried.append(fmt)

        raise ValueError(f"Could not determine the file format or read the file '{self.file}'. "
                         f"Tried formats: {', '.join(tried)}.")
    
    def repo_files(self) -> List[str]:
        """
        Returns the list of files available in the file repository provided. 
        If no file repository is provided, it will be extracted from the base directory of the file.
        """
        return os.listdir(self.file_repo)