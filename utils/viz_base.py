from typing import Dict
import pandas as pd

import dash
import dash_bootstrap_components as dbc
from dash import html, dash_table, Input, Output
import dash_ag_grid as dag  # If you're using dash-ag-grid

class SummaryTables:
    def __init__(self, summary: Dict, port: int = 8080, jupyter: bool=True, debug: bool=True):
        """Initialize the Dash app and set up layouts.

        Parameters
        ----------
        summary : dict
            A dictionary (or similar) containing dataframes under keys 
            like 'shape', 'info', 'describe', etc.
        port : int, default 8080
            Port to run the Dash app on.
        jupyter : bool, default True
            If True, runs the app in jupyter_mode='inline'.
        debug : bool, default True
            If True, runs the Dash app with debug mode enabled.
        """
        self.df = summary
        self.port = port
        self.jupyter = jupyter
        self.debug = debug
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.LUX]
        )

    def _layout(self):
        """
        Constructs the layout for displaying summary tables:
        - Shape
        - Info (using dash-ag-grid)
        - Summary statistics (using dash-ag-grid)
        """

        # Header
        header = html.Div([html.H2('Data Summary')], style={'textAlign': 'center'})

        # Shape data table
        shape = dbc.Row([
            html.Div([html.H5('Shape')], style={'textAlign': 'center'}),
            dbc.Col([
                dash_table.DataTable(
                    data=self.df['shape'].to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in self.df['shape'].columns],
                    style_header={'textAlign': 'center', 'fontWeight': 'bold'},
                    style_cell={'textAlign': 'center'},
                    fill_width=True
                )
            ], width=2)
        ], justify='center')

        # Info grid
        info_grid = dbc.Row([
            html.Div([html.H5('Info')], style={'textAlign': 'center'}),
            dbc.Col([
                dag.AgGrid(
                    rowData=self.df['info'].to_dict('records'),
                    columnDefs=[{'field': i, 'filter': True, 'sortable': True}
                                for i in self.df['info'].columns],
                    columnSize='responsiveSizeToFit',
                    # dashGridOptions={"domLayout": "autoHeight"},
                )
            ])
        ], justify='center')

        # Summary statistics grid
        no_fmt = ['count', 'unique', 'top', 'freq']
        desc_grid = dbc.Row([
            html.Div([html.H5('Summary Statistics')], style={'textAlign': 'center'}),
            dbc.Col([
                dag.AgGrid(
                    rowData=self.df['describe'].to_dict('records'),
                    columnDefs=[
                        {'field': i, 'filter': True, 'sortable': True, 'pinned': 'left'} if i == 'Columns' else 
                        {'field': i, 'sortable': True} if i in no_fmt else 
                        {'field': i, 'sortable': True, "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}}
                        for i in self.df['describe'].columns
                    ],
                    columnSize='autoSize',
                )
            ])
        ], justify='center')

        layout = dbc.Container([
            html.Br(),
            header,
            html.Br(),
            shape,
            html.Br(),
            info_grid,
            html.Br(),
            desc_grid
        ], fluid=False)

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

class ValueTables:
    def __init__(self, df:pd.DataFrame, dups:Dict[str, int], miss: pd.DataFrame, vals: pd.DataFrame, 
                 port: int=8080, jupyter: int=True, debug: int=True):
        """Initialize the Dash app and set up layouts.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to use for analysis.
        dups : dict
            Contains information about duplicates (e.g. {'Duplicate Rows': int, 'Duplicate Column Names': [...]})
        miss : pd.DataFrame
            DataFrame with missing data info.
        vals : pd.DataFrame
            DataFrame with value distribution info (columns, values, counts, distribution).
        port : int, default 8080
            Port to run the Dash app on.
        jupyter : bool, default True
            If True, runs the app in jupyter_mode='inline'.
        debug : bool, default True
            If True, runs the Dash app with debug mode enabled.
        """
        self.df = df 
        self.dups = dups
        self.miss = miss
        self.vals = vals
        self.port = port
        self.jupyter = jupyter
        self.debug = debug
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.LUX]
        )

    def _layout(self):
        """
        Constructs the layout for displaying:
        - Duplicate information
        - Missing data info (dash-ag-grid)
        - Value distributions (dash-ag-grid with callbacks for column selection)
        """

        # Duplicates
        nr = 'Duplicate Rows'
        nc = 'Duplicate Column Names'
        sh = {'textAlign': 'center', 'fontWeight': 'bold'}
        sc = {'textAlign': 'center'}

        duplicates = dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    data=[{nr:self.dups[nr]}],
                    columns=[{'name': nr, 'id': nr}],
                    style_header=sh, style_cell= sc, fill_width=True
                )
            ], width=2),
            dbc.Col([
                dash_table.DataTable(
                    data=self.dups[nc],
                    columns=[{'name': i, 'id': i} for i in self.dups[nc][0].keys()],
                    style_header= sh, style_cell= sc, fill_width=True
                )
            ], width=3)
        ], justify='center')

        missing = dbc.Row([
            dbc.Col([
                dag.AgGrid(
                    rowData=self.miss.to_dict('records'),
                    columnDefs=[
                        {'field': i, 'sortable': True, 'filter': True, 
                        "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}} if i == 'Missing %' else
                        {'field': i, 'sortable': True,  'filter': True,}
                        for i in self.miss.columns
                    ],
                    columnSize='responsiveSizeToFit',
                )
            ], width=7),
        ], justify='center')

        value_dist = dbc.Container([
            dbc.Row([
                        dbc.Col([
                            dbc.Select(id='select-columns', placeholder='Select Columns...', 
                                    value= self.vals['Columns'].unique()[0],
                                    options=[{'label': i, 'value': i} for i in self.vals['Columns'].unique()]
                            )
                        ], width=3),
                        dbc.Col([
                            html.P(id='unique-values')
                        ], width=4, align='end'),
                ], justify='center'),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dag.AgGrid(
                        id='value-counts',
                        columnDefs=[
                            { 'field': 'Values' , 'cellDataType': False},
                            { 'field': 'Count'},
                            { 'field': 'Distribution, %' },
                        ],
                        columnSize='responsiveSizeToFit',)
                ],width=7)
            ], justify='center')
        ],fluid=True)

        layout = dbc.Container([
            html.Br(),
            dbc.Accordion([
                dbc.AccordionItem(duplicates, title='Duplicates'),
                dbc.AccordionItem(missing, title='Missing'),
                dbc.AccordionItem(value_dist, title='Value Distributions'),
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
            Output('unique-values', 'children'),
            [Input('select-columns', 'value')]
        )
        def unique_vals(value):
            val_counts = self.df[value].nunique()
            val_text = f'Unique Value Counts: {val_counts}'
            return val_text


        @app.callback(
            Output('value-counts', 'rowData'),
            [Input('select-columns', 'value')]
        )
        def value_counts(value):
            val_dist = self.vals[self.vals['Columns']==value][['Values', 'Count', 'Distribution, %']]
            return val_dist.to_dict('records')

    def _build_layout(self):
        """
        Build the layout and attach it to `self.app`, then register callbacks.
        """
        layout = self._layout()
        self.app.layout = layout
        self._register_callbacks()
     
    def run(self):
        """
        Run the Dash server.
        """
        self._build_layout()
        if self.jupyter:
            self.app.run(jupyter_mode='inline', port=self.port)
        else:
            self.app.run_server(debug=self.debug, port=self.port)
