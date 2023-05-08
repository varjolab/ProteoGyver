
from uuid import uuid4, UUID
from dash import Dash, html, callback, Input, Output, State  # dcc,DiskcacheManager, CeleryManager
import dash_bootstrap_components as dbc
import dash
from DbEngine import DbEngine


launch_uid: UUID = str(uuid4())
db: DbEngine = DbEngine()

app: Dash = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.SANDSTONE
    ],
    suppress_callback_exceptions=True)
app.title = 'Data analysis alpha version'
app.enable_dev_tools(debug=True)
#app.config.from_pyfile('app_config.py')
#app.enable_devtools
server = app.server
print('Site pages:')
for page in dash.page_registry.values():
    print(page['name'])
navbar: dbc.NavbarSimple = dbc.NavbarSimple(
    id='main-navbar',
    children=[
        dbc.NavItem(
            dbc.NavLink(page['name'], href=page['relative_path'])
        )
        for page in dash.page_registry.values()
    ],
    brand='Quick analysis',
    color='primary',

    dark=True
)


@callback(
    Output('notused', 'children'),
    Input('button-clear-session-data', 'n_clicks'),
    State('session-uid', 'children'),
    prevent_initial_call=True,
)
def clear_session_data(_, uid) -> str:
    db.clear_session_data(uid)
    return 'cleared'


app.layout = html.Div([
    html.Div(id='session-uid', children=launch_uid, hidden=True),
    navbar,
    dash.page_container,
    #    dbc.Button('Clear session data',\
    #        id='button-clear-session-data'),
    html.Div(id='notused', hidden=True)

])

if __name__ == '__main__':
    app.run_server()#debug=True)
# <iframe src="https://www.chat.openai.com/chat" title="ChatGPT Embed"></iframe>
