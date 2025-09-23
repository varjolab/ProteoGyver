"""Main application module for ProteoGyver.

This module initializes and configures the Dash application with Celery for long callbacks,
sets up logging, creates the navigation bar, and defines the main layout structure.

Attributes:
    celery_app (Celery): Celery application instance for handling long callbacks
    long_callback_manager (CeleryLongCallbackManager): Manager for Celery long callbacks
    app (Dash): Main Dash application instance
    server (Flask): Flask server instance from Dash app
    logger (Logger): Application logger instance
"""

from dash import Dash, html, page_registry, page_container
import dash_bootstrap_components as dbc
from dash_bootstrap_components.themes import FLATLY
from dash.dependencies import Input, Output, State
import logging
import os
from celery import Celery
from dash.long_callback import CeleryLongCallbackManager
from datetime import datetime
from components.tools import utils
from celery.schedules import crontab

celery_app = Celery(
    __name__, broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)
long_callback_manager = CeleryLongCallbackManager(celery_app, expire=300)

app = Dash(__name__, use_pages=True, external_stylesheets=[
           FLATLY], suppress_callback_exceptions=True, long_callback_manager=long_callback_manager)

app.title = 'ProteoGyver beta version'
app.enable_dev_tools(debug=True)


def main() -> None:
    """Main entry point for running the application.

    """
    logger.info(f'Proteogyver started: {datetime.now()}')
    app.run(debug=True)

def create_navbar(parameters: dict) -> dbc.Navbar:
    """Creates the application navigation bar.
    
    Args:
        parameters (dict): Application parameters containing navbar configuration
        
    Returns:
        dbc.Navbar: Bootstrap navbar component with configured pages and styling
        
    Notes:
        - Creates NavItems for each registered page
        - Orders pages according to parameters['Navbar page order']
        - Includes logo, brand name and collapsible navigation
    """
    LOGO = 'assets/images/proteogyver.png'
    for page in page_registry.values():
        logger.info(page['name'])
    pages = {
        page['name'].lower(): dbc.NavItem(
            dbc.NavLink(
                page['name'].upper(),
                href=page['relative_path'],
                style={
                    'padding': '10px',
                    'color': 'white'
                }
            )
        ) for page in page_registry.values()
    }

    pages_in_order = parameters['Navbar page order']
    pages_in_order.extend(sorted([p for p in pages.keys() if p not in pages_in_order]))

    navbar_items: list = []
    navbar_items.extend([pages[p] for p in pages_in_order if p in pages])

    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.Img(src=LOGO, height='100px',id='proteogyver-logo'),
                dbc.NavbarBrand('ProteoGyver', className='ms-2', style = {'paddingRight': '50px','font-size': '30px'} ),
                dbc.NavbarToggler(id="proteogyver-navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    navbar_items,
                    id="proteogyver-navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ]
        ),
        id='proteogyver-navbar',
        color='primary',
        style={'zIndex':2147483647, 'position': 'fixed', 'width': '100%', 'height': '85px', 'Top': 0},
        dark=True,
    )
    return navbar

@app.callback(
    Output("proteogyver-navbar-collapse", "is_open"),
    [Input("proteogyver-navbar-toggler", "n_clicks")],
    [State("proteogyver-navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n: int, is_open: bool) -> bool:
    """Callback to toggle the navbar collapse state.
    
    Args:
        n (int): Number of clicks on the toggle button
        is_open (bool): Current collapse state
        
    Returns:
        bool: New collapse state
    """
    if n:
        return not is_open
    return is_open

parameters = utils.read_toml('parameters.toml')
server = app.server
if not os.path.isdir('logs'):
    os.makedirs('logs')

logfilename:str = os.path.join(
    parameters['Config']['LogDir'], f'{datetime.now().strftime("%Y-%m-%d")}_proteogyver.log')
logging.basicConfig(filename=logfilename, level=parameters['Config']['LogLevel'])
logger = logging.getLogger(__name__)
logger.info('Site pages:')
app.layout = html.Div([
    create_navbar(parameters),
    page_container,
],id='proteogyver-layout')
logger.info('End app.')

celery_app.conf.beat_schedule = {
    'cleanup-cache-daily': {
        'task': 'components.cleanup_tasks.cleanup_cache_folders',
        'schedule': crontab(hour=0, minute=30),
    },
}

if __name__ == '__main__':
    main()