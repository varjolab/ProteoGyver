import json
import dash
from dash import Dash, html, dcc,dash_table, callback
import pandas as pd
from DbEngine import DbEngine
from dash.dependencies import Input, Output, State

dash.register_page(__name__)
db: DbEngine = DbEngine()

@callback(
        Output('workflow-qc-figures', 'children'),
        Input('analysis-page','pathname'),
        State('session-uid','children')
        )
def generate_workflow_figures(_, session_uid) -> str:
    workflow: str = ''
    with open(db.get_cache_file(session_uid + '_workflow-choice.txt')) as fil:
        workflow = fil.read().strip()
    with open(db.get_cache_file(session_uid + '_data-dict.json')) as fil:
        data: dict = json.load(fil)
        sample_groups: dict = data['sample groups']
        table: pd.DataFrame = pd.read_json(data['table'],orient='split')
    if workflow == 'proteomics':
        pass
    if session_uid:
        return str(session_uid)
    return ''

layout:html.Div = html.Div(children=[
    dcc.Location(id='analysis-page'),
    html.H1(children='Workflow-specific QC'),
    html.Div(id='workflow-qc-figures')

])
