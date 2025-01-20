from typing import Dict
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import html, dash_table, Input, Output
import dash_ag_grid as dag

class CorrViz:
    def __init__(self, concat_df: pd.DataFrame, concon_df: pd.DataFrame, catcat_df: pd.DataFrame, mut_inf: pd.DataFrame,
                 port: int = 8080, jupyter: bool=True, debug: bool=True):
        """Initialize the Dash app and set up layouts.

        Parameters
        ----------
        port : int, default 8080
            Port to run the Dash app on.
        jupyter : bool, default True
            If True, runs the app in jupyter_mode='inline'.
        debug : bool, default True
            If True, runs the Dash app with debug mode enabled.
        """
        self.concat_df = concat_df
        self.concon_df = concon_df
        self.catcat_df = catcat_df
        self.mut_inf = mut_inf

        self.port = port
        self.jupyter = jupyter
        self.debug = debug
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.LUX]
        )

    def _layout(self):
        concat = dbc.Row([
            dbc.Col([
                dag.AgGrid(
                    rowData=self.concat_df.to_dict('records'),
                    columnDefs=[{'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}} 
                                if i == 'Correlation' else {'field':i} 
                                for i in self.concat_df.columns],
                    columnSize='responsiveSizeToFit',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            ], width=6)
        ], justify='center')

        concon = dbc.Row([
            dbc.Col([
                dag.AgGrid(
                    rowData=self.concon_df.reset_index(names='Columns').to_dict('records'),
                    columnDefs=[{'field':i, 'pinned': True}  if i == 'Columns' else 
                                {'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}}
                                for i in self.concon_df.reset_index(names='Columns').columns],
                    columnSize='sizeToFit',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            ])
        ], justify='center')

        catcat = dbc.Row([
            dbc.Col([
                dag.AgGrid(
                    rowData=self.catcat_df.to_dict('records'),
                    columnDefs=[{'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}} 
                                if i == 'CramersV' else {'field':i}
                                for i in self.catcat_df.columns],
                    columnSize='responsiveSizeToFit',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            ], width=7)
        ], justify='center')

        mui = dbc.Row([
            dbc.Col([
                dag.AgGrid(
                    rowData=self.mut_inf.reset_index(names='Columns').to_dict('records'),
                    columnDefs=[{'field':i, 'pinned': True}  if i == 'Columns' else 
                                {'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}}
                                for i in self.mut_inf.reset_index(names='Columns').columns],
                    columnSize='sizeToFit',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            ])
        ], justify='center')

        layout = dbc.Container([
            html.Br(),
            dbc.Accordion([
                dbc.AccordionItem(concat, title='Continuous-Category'),
                dbc.AccordionItem(concon, title='Continuous-Continuous'),
                dbc.AccordionItem(catcat, title='Category-Category'),
                dbc.AccordionItem(mui, title='Mutual Information'),

            ], flush=True)
        ], fluid=True)

        return layout
    
    def _build_layout(self):
        """
        Build the layout and attach it to `self.app`.
        """
        layout = self._layout()
        self.app.layout = layout

    def run(self):
        """
        Run the Dash server.
        """
        self._build_layout()

        if self.jupyter:
            self.app.run(jupyter_mode='inline', port=self.port)
        else:
            self.app.run_server(debug=self.debug, port=self.port)