"""Main application module for ProteoGyver.

This module initializes and configures the Dash application with Celery for long callbacks,
sets up logging, creates the navigation bar, and defines the main layout structure.

Attributes:
    celery_app (Celery): Celery application instance for handling long callbacks
    app (Dash): Main Dash application instance
    server (Flask): Flask server instance from Dash app
    logger (Logger): Application logger instance
"""

from dash import Dash, html, page_registry, page_container, CeleryManager
import dash_bootstrap_components as dbc
from dash_bootstrap_components.themes import FLATLY
from dash.dependencies import Input, Output, State
import logging
import os
from pathlib import Path
from celery import Celery
from datetime import datetime
from components.tools import utils
from celery.schedules import crontab
from _version import __version__

celery_app = Celery(
    __name__,
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=[
        'components.MS_run_json_parser',
        'components.cleanup_tasks',
        'pipeline_module.pipeline_input_watcher'
    ]
)
background_callback_manager = CeleryManager(celery_app, expire=300)

app = Dash(__name__, use_pages=True, external_stylesheets=[
           FLATLY], suppress_callback_exceptions=True, background_callback_manager=background_callback_manager)

app.title = f'ProteoGyver {__version__}'
# Only enable dev tools when not running in Docker
if not os.path.exists('/.dockerenv'):
    app.enable_dev_tools(debug=True)
    

def main() -> None:
    """Run the Dash application (main entry point)."""
    logger.info(f'Proteogyver {__version__} started: {datetime.now()}')
    # Only run in debug mode when not in Docker
    debug_mode = not os.path.exists('/.dockerenv')
    logger.info(f'Debug mode: {debug_mode}')
    app.run(debug=debug_mode)

def create_navbar(parameters: dict) -> dbc.Navbar:
    """Create the application navigation bar.

    :param parameters: App parameters containing navbar configuration.
    :returns: Bootstrap Navbar with pages and branding.
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
                dbc.NavbarBrand(f'Proteogyver {__version__} ', className='ms-2', style = {'paddingRight': '50px','font-size': '30px'} ),
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
    """Toggle the navbar collapse state.

    :param n: Number of clicks on the toggle button.
    :param is_open: Current collapse state.
    :returns: New collapse state.
    """
    if n:
        return not is_open
    return is_open

parameters = utils.read_toml(Path('config/parameters.toml'))
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

celery_app.conf.ONCE = {
    "backend": "celery_once.backends.Redis",
    "settings": {
        "url": "redis://localhost:6379/2",  # a separate DB is nice, but not required
        "default_timeout": 60 * 60 * 24,         # lock expiry (seconds) – tune to your longest run
    },
}
celery_app.conf.beat_schedule = {
    'cleanup-cache-daily': {
        'task': 'components.cleanup_tasks.cleanup_cache_folders',
        'schedule': crontab(hour=0, minute=30),
    },
    'rotate-logs-daily': {
        'task': 'components.cleanup_tasks.rotate_logs',
        'schedule': crontab(hour=15, minute=45),
    },
    'parse-MS-runs': {
        'task': 'components.MS_run_json_parser.parse_json_files',
        'schedule': crontab(minute='*/5'),
    },
    'watch-pipeline-input': {
        'task': 'pipeline_module.pipeline_input_watcher.watch_pipeline_input',
        'schedule': crontab(minute='*'),
        'args': (parameters['Pipeline module']['Input watch directory'],),
    }
}

if __name__ == '__main__':
    main()