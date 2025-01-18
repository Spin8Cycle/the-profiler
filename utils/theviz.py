import importlib

import dash
import dash_bootstrap_components as dbc

from . import viz_base

importlib.reload(viz_base)


class MainDB:
    """
    A combined Dash application that can hold multiple dashboards in a tabbed layout.
    """
    def __init__(self, port:int=8080, jupyter:bool=True, debug:bool=True, external_stylesheets:list=[dbc.themes.LUX]):
        """
        Parameters
        ----------
        port : int, default 8080
            The port to run the Dash server on.
        jupyter : bool, default True
            If True, runs in jupyter_mode='inline' (for notebooks).
        debug : bool, default True
            If True, runs the Dash server in debug mode.
        external_stylesheets : list, optional
            A list of external stylesheets, e.g., [dbc.themes.LUX].
        """

        self.port = port
        self.jupyter = jupyter
        self.debug = debug
        self.app = dash.Dash(
            __name__, 
            external_stylesheets=external_stylesheets
        )
        self.dashboards = {}

    def _register_dashboard(self, dashboard_instance, tab_name):
        """
        Create and register a new dashboard from a dashboard class.
        
        Parameters
        ----------
        dashboard_instance : Protocol
            An instance of dash apps.

        """
        self.dashboards[tab_name] = dashboard_instance

    def _build_layout(self):
        """
        Build a set of dbc.Tab elements for each registered dashboard, 
        then wrap them in a dbc.Tabs container.
        """
        tabs = []
        for tabname, dashboard in self.dashboards.items():
            tabs.append(dbc.Tab(dashboard._layout(), label=tabname))

        self.app.layout=dbc.Tabs(tabs)
    
    def _register_callbacks(self):
        """
        If a dashboard has a '_register_callbacks' method, call it here 
        and pass in `self.app`. This consolidates callback wiring.
        """
        for dashboard in self.dashboards.values():
            if hasattr(dashboard, '_register_callbacks'):
                dashboard._register_callbacks(app_inst=self.app)

    def run(self):
        """
        Final method to run the Dash application. 
        Build layout, register callbacks, and then run the server.
        """

        self._build_layout()
        self._register_callbacks()

        if self.jupyter:
            self.app.run(jupyter_mode='inline', port=self.port)
        else:
            self.app.run_server(debug=self.debug, port=self.port)