from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, dcc
import dash_ag_grid as dag

import importlib
from . import view_corr

importlib.reload(view_corr)


class CorrViz:
    def __init__(
            self, 
            df: pd.DataFrame, port: int = 8080, jupyter: bool=True, debug: bool=True,
            continuous_cols: Optional[List[str]]=None, 
            categorical_cols: Optional[List[str]]=None,
            category_threshold: Optional[int]=10,
            **kwargs 
):
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

        self.port = port
        self.jupyter = jupyter
        self.debug = debug
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.LUX]
        )

        self.corrview = view_corr.CorrelationViewer(
            df=df, 
            continuous_cols=continuous_cols, 
            categorical_cols=categorical_cols,
            category_threshold=category_threshold
        )


    def _layout(self):
        # CONTINUOUS-CATEGORY
        concat = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Table / Graph View", color="dark", className="border rounded-pill", n_clicks=0, id='concat-button', size='sm')
                ], align='center')
            ], justify='center'),
            html.Br(),
            dbc.Row(id='concat-display',justify='center')
        ],  fluid=True)

        # CONTINUOUS-CONTINUOUS
        concon = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Table / Graph View", color="dark", className="border rounded-pill", n_clicks=0, id='concon-button', size='sm')
                ], align='center'),
                dbc.Col([
                    dbc.Select(
                        id='corr-method-selector', value='pearson', size='sm',
                        options=[
                            {'label': 'pearson', 'value':'pearson'},
                            {'label': 'spearman', 'value':'spearman'},
                            {'label': 'kendall', 'value':'kendall'},])
                ], align='center', width='auto')
            ], justify='center'),
            html.Br(),
            dbc.Row(id='concon-display',justify='center')
        ],  fluid=True)

        # CATEGORY-CATEGORY
        catcat = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Table / Graph View", color="dark", className="border rounded-pill", n_clicks=0, id='catcat-button', size='sm')
                ], align='center'),
            ], justify='center'),
            html.Br(),
            dbc.Row(id='catcat-display',justify='center')
        ],  fluid=True)

        # MUTUAL INFORMATION
        mui = dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Table / Graph View", color="dark", className="border rounded-pill", n_clicks=0, id='mui-button', size='sm')
                ], align='center')
            ], justify='center'),
            html.Br(),
            dbc.Row(id='mui-display',justify='center')
        ],  fluid=True)

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
    
    def _register_callbacks(self, app_inst=None):
        """
        Register Dash callbacks for interactive updates.
        """

        if not app_inst:
            app = self.app
        else:
            app = app_inst

        @app.callback(
                Output('concat-display', 'children'),
                [Input('concat-button', 'n_clicks')]
        )
        def concat_toggle(n):
            df = self.corrview.continuous_categorical_corr()
            if n % 2 ==0:
                return dag.AgGrid(
                    rowData=df.to_dict('records'),
                    columnDefs=[{'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}} 
                                if i == 'Correlation' else {'field':i} 
                                for i in df.columns],
                    columnSize='responsiveSizeToFit',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            else:
                return dcc.Graph(figure=self.corrview.plot_continuous_categorical_corr(), 
                                 responsive=True)
            
        @app.callback(
                Output('concon-display', 'children'),
                [Input('concon-button', 'n_clicks'),
                Input('corr-method-selector', 'value')]
        )
        def concon_toggle(n, val):
            cc = self.corrview.continuous_continuous_corr(method=val)
            if n % 2 ==0:
                return dag.AgGrid(
                    rowData=cc.reset_index(names='Columns').to_dict('records'),
                    columnDefs=[{'field':i, 'pinned': True}  if i == 'Columns' else 
                                {'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}}
                                for i in cc.reset_index(names='Columns').columns],
                    columnSize='auto',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            else:
                return dcc.Graph(figure=self.corrview.plot_continuous_continuous_corr(method=val), 
                                 responsive=True)
            
        @app.callback(
                Output('catcat-display', 'children'),
                [Input('catcat-button', 'n_clicks')]
        )
        def catcat_toggle(n):
            cc = self.corrview.categorical_categorical_corr()
            if n % 2 ==0:
                return dag.AgGrid(
                    rowData=cc.to_dict('records'),
                    columnDefs=[{'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}} 
                                if i == 'CramersV' else {'field':i}
                                for i in cc.columns],
                    columnSize='responsiveSizeToFit',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            else:
                return dcc.Graph(figure=self.corrview.plot_categorical_categorical_corr(), 
                                 responsive=True)
            
        @app.callback(
                Output('mui-display', 'children'),
                [Input('mui-button', 'n_clicks')]
        )
        def mui_toggle(n):
            mudf = self.corrview.mutual_information_continuous()
            if n % 2 ==0:
                return dag.AgGrid(
                    rowData=mudf.reset_index(names='Columns').to_dict('records'),
                    columnDefs=[{'field':i, 'pinned': True}  if i == 'Columns' else 
                                {'field':i, 'valueFormatter': {"function": "d3.format(',.2f')(params.value)"}}
                                for i in mudf.reset_index(names='Columns').columns],
                    columnSize='auto',
                    defaultColDef={'filter':True, 'sortable':True}
                )
            else:
                return dcc.Graph(figure=self.corrview.plot_mutual_inf(), responsive=True)
    
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

    def corrvizz(self):
        inst = CorrViz(df=self.df)
        return inst