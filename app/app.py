
from dash import Dash, html, page_registry, page_container
import dash_bootstrap_components as dbc
from dash_bootstrap_components.themes import FLATLY
import logging
import os
import json
from celery import Celery
from dash.long_callback import CeleryLongCallbackManager
from datetime import datetime

celery_app = Celery(
    __name__, broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)
long_callback_manager = CeleryLongCallbackManager(celery_app, expire=300)

app = Dash(__name__, use_pages=True, external_stylesheets=[
           FLATLY], suppress_callback_exceptions=True, long_callback_manager=long_callback_manager)

app.title = 'Data analysis alpha version'
app.enable_dev_tools(debug=True)
#app.config.from_pyfile('app_config.py')
server = app.server
if not os.path.isdir('logs'):
    os.makedirs('logs')
with open('parameters.json') as fil:
    parameters = json.load(fil)

# Logging levels:
#    NOTSET=0.
#    DEBUG=10.
#    INFO=20.
#    WARN=30.
#    ERROR=40.
#    CRITICAL=50.
logfilename:str = os.path.join(
    parameters['Config']['LogDir'], f'{datetime.now().strftime("%Y-%m-%d")}_proteogyver.log')
logging.basicConfig(filename=logfilename, level=parameters['Config']['LogLevel'])

#log_handler = logging.handlers.RotatingFileHandler(os.path.join(
    #'logs', f'{datetime.now().strftime("%Y-%m-%d")}_proteogyver.log'), maxBytes=1000000000, backupCount=1)
#log_handler.setLevel(logging.WARNING)
#logger = logging.getLogger(__name__)
#logger.warning(f'Proteogyver started: {datetime.now()}')

logger = logging.getLogger(__name__)

logger.warning('Site pages:')
for page in page_registry.values():
    logger.warning(page['name'])
pages = {
    page['name'].lower(): dbc.NavItem(
        dbc.NavLink(
            page['name'].upper(),
            href=page['relative_path']
        )
    ) for page in page_registry.values()
}

pages_in_order = [
    'qc and data analysis',
    'ms analytics dashboard',
    'windowmaker',
]
pages_in_order.extend(sorted([p for p in pages.keys() if p not in pages_in_order]))
navbar_items = [pages[p] for p in pages_in_order]

def main() -> None:
    logger.warning(f'Proteogyver started: {datetime.now()}')
    app.run(debug=True)

#navbar_items.append(
#    dbc.NavItem(
#        dbc.NavLink('JupyterHub',href='pg-23.biocenter.helsinki.fi:8090/')
#    )
#)
navbar: dbc.NavbarSimple = dbc.NavbarSimple(
    id='main-navbar',
    children=navbar_items,
    brand='ProteoGyver',
    color='primary',
    dark=True,
    style={'zIndex':2147483647, 'position': 'fixed', 'width': '100%', 'Top': 0},
    fluid=True
)

app.layout = html.Div([
    navbar,
    page_container,
])
logger.warning('End app.')
if __name__ == '__main__':
    main()