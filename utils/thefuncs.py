from typing import Optional, Union, List, Dict
import importlib

import pandas as pd
from . import view_base, view_corr, viz_corr

importlib.reload(view_base)
importlib.reload(viz_corr)


class Viewers(view_base.BasicPropLoader,
              view_corr.CorrelationViewer
              ):
    
    def __init__(self, df:pd.DataFrame=None,**kwargs):
        super().__init__(df=df, **kwargs)
        
    def __getattr__(self, name:str):
        """
        Delegate attribute access to the underlying DataFrame.

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
        

class Modifiers():
    pass