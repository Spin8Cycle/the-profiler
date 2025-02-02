from typing import Optional, Dict, List
import importlib

import pandas as pd


from . import (data_utils, theviz, thefuncs, viz_base, viz_corr)

importlib.reload(data_utils)
importlib.reload(thefuncs)
importlib.reload(theviz)
importlib.reload(viz_base)
importlib.reload(viz_corr)


class TheProfiler(data_utils.DataLoader, 
                  thefuncs.Viewers):
    """
    Combines data loading and property loading capabilities. 
    Also offers dashboard creation methods.
    """
    
    def __init__(self, 
                 file: Optional[str]=None, 
                 file_repo: Optional[str]=None, 
                 df: Optional[pd.DataFrame]=None,
                 **kwargs):
        super().__init__(
                file=file, 
                file_repo=file_repo, 
                df=df,
                **kwargs
            )
        
        self.kwargs = kwargs
        
    def main_dashboard(self, **kwargs):
        main_db = theviz.MainDB(**kwargs)
        main_db._register_dashboard(self.summary_tables(main=False), 'Summary')
        main_db._register_dashboard(self.value_tables(main=False), 'Value Properties')
        main_db._register_dashboard(self.corr_tables(main=False), 'Correlations')

        return main_db.run()

    def summary_tables(self, main: bool=True, **kwargs): #viz_base
        summary = self.show_summary(display_df=False)
        inst = viz_base.SummaryTables(summary, **kwargs)
        if main:
            dashboard = inst.run()
        else:
            dashboard = inst 
        return dashboard
    
    def value_tables(self, main: bool=True, **kwargs): #viz_base
        dups = self.check_duplicates()
        miss = self.check_missing_data()
        vals = self.check_value_counts().reset_index()
        inst = viz_base.ValueTables(self.df, dups, miss, vals, **kwargs)
        if main:
            dashboard = inst.run()
        else:
            dashboard = inst
        return dashboard
    
    def corr_tables(self, main: bool=True, **kwargs): #viz_corr
        
        inst = viz_corr.CorrViz(
            df = self.df,
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            category_threshold=self.category_threshold
        )

        if main:
            dashboard = inst.run()
        else:
            dashboard = inst
        return dashboard

        




    

