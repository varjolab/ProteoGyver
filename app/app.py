
from dash import Dash, html, callback, Input, Output, State  # dcc,DiskcacheManager, CeleryManager
import dash_bootstrap_components as dbc
import dash
from DbEngine import DbEngine
import logging
import logging

db: DbEngine = DbEngine()

app: Dash = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.SANDSTONE
    ],
    suppress_callback_exceptions=True
)

app.title = 'Data analysis alpha version'
app.enable_dev_tools(debug=True)
#app.config.from_pyfile('app_config.py')
server = app.server


logger = logging.getLogger('applogger')
log_handler = logging.handlers.RotatingFileHandler('dashlog.log', maxBytes=1000000000, backupCount=1)
log_handler.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.warning('Start app.')
server.logger.addHandler(log_handler)

logger.warning('Initialize app.')
logger.info('Site pages:')
for page in dash.page_registry.values():
    logger.info(page['name'])
navbar_items = [
        dbc.NavItem(
            dbc.NavLink(page['name'], href=page['relative_path'])
        )
        for page in dash.page_registry.values()
    ]
#navbar_items.append(
#    dbc.NavItem(
#        dbc.NavLink('JupyterHub',href='pg-23.biocenter.helsinki.fi:8090/')
#    )
#)
navbar: dbc.NavbarSimple = dbc.NavbarSimple(
    id='main-navbar',
    children=navbar_items,
    brand='Quick analysis',
    color='primary',

    dark=True
)


@callback(
    Output('session-data-clear-output', 'children'),
    Input('button-clear-session-data', 'n_clicks'),
    State('session-uid', 'children'),
    prevent_initial_call=True,
)
def clear_session_data(_, uid) -> str:
    db.clear_session_data(uid)
    return 'cleared'


app.layout = html.Div([
    html.Div(id='session-uid', children='placeholder', hidden=True),
    navbar,
    dash.page_container,
    #    dbc.Button('Clear session data',\
    #        id='button-clear-session-data'),
    html.Div(id='session-data-clear-output', hidden=True)

])
logger.warning('End app.')
if __name__ == '__main__':
    app.run(debug=True)
# <iframe src="https://www.chat.openai.com/chat" title="ChatGPT Embed"></iframe>
