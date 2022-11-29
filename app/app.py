
from dash.dependencies import Input, Output, State
from dash import Dash, html, callback# dcc,DiskcacheManager, CeleryManager
import dash_bootstrap_components as dbc
import dash
from uuid import uuid4, UUID
from DbEngine import DbEngine


launch_uid: UUID = str(uuid4())
db: DbEngine = DbEngine()

app: Dash = Dash(__name__, use_pages=True, external_stylesheets = [dbc.themes.DARKLY])

navbar: dbc.NavbarSimple = dbc.NavbarSimple(
    id = 'main-navbar',
    children = [
        dbc.NavItem(
            dbc.NavLink(page['name'], href = page['relative_path'])
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
    State('session-uid','children'),
    prevent_initial_call=True,
)
def sample_table_download(_,uid) -> str:
    db.clear_session_data(uid)
    return 'cleared'

app.layout = html.Div([
    html.Div(id='session-uid',children=launch_uid,hidden=True),
    navbar,
	dash.page_container,
    dbc.Button('Clear session data',\
        id='button-clear-session-data'),
    html.Div(id='notused',hidden=True)
                
])

if __name__ == '__main__':
	app.run_server(debug=True)
