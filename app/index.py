"""Dash app for data upload"""

import json
from dash import Dash, html, dcc
from dash import dash_table
import pandas as pd
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from DbEngine import DbEngine
import uuid

# db will house all data, keep track of next row ID, and validate any new data
db: DbEngine = DbEngine()
app: Dash = Dash(__name__)
server: app.server = app.server
"""
parameter_file: str = 'parameters.json'
with open(parameter_file, encoding='utf-8') as fil:
    parameters: dict = json.load(fil)

known_types: dict = parameters['Known types']

data: pd.DataFrame = pd.read_csv(parameters['Run data'],sep='\t')
last_id: int = data['id'].max() + 1
upload_table_data_columns: list = []
upload_table_data_dropdowns: dict = {}
"""
def get_next_n_ids(number:int) -> list:
    """Returns a list of next number ids for the run data table
    """
    return list(range(db.last_id, db.last_id + number))

def QC_tab() -> dcc.Tab:
    session_id: str = str(uuid.uuid4())
    return dcc.Tab(
            label='QC',
            children = [
                dbc.Container([
                    html.Div(session_id, id='session-id', style={'display': 'none'}),
                    html.Div(dcc.Input(id="input_session_id",type="text",value=session_id))
                ])
            ]
        )

def generate_rows(how_many:int) -> list:
    """This function will generate new empty input rows for the data input datatable.
    """
    retlist: list = []
    columns: pd.Index = db.data.columns
    nums: list = get_next_n_ids(how_many)
    for number in nums:
        retlist.append({})
        for column_name in columns:
            if column_name == 'id':
                val: str = str(number)
            else:
                val: str = ''
            retlist[-1][column_name] = val
    return retlist

#graph = px.bar(data, x='Fruit', y='Amount', color='City', barmode='group')


upload_tabs: list = [
    dcc.Tab(
        label='Upload data',
        children = [
            dbc.Container([
                dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('select'),
                            ' Data file'
                        ]),
                        style={
                            'width': '40%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': 'auto',
                            'float': 'left'
                        },
                        multiple=False
                ),
                dcc.Upload(
                        id='upload-expdesign',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('select'),
                            ' ExpDesign file'
                        ]),
                        style={
                            'width': '40%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': 'auto',
                            'float': 'right'
                        },
                        multiple=False
                ),
                html.Button('Download ExpDesign template',\
                    id='button-download-expdesign-template'),
                html.Div(id='output-data-upload'),
            ], style={
                'margin': '0px',
                'float': 'center'
            }),
            dbc.Container([
                dbc.Alert(id='analysis-table-problems'),
                dash_table.DataTable(
                    page_size=50,
                    #fixed_rows = {'headers': True, 'data': 0},
                    #fixed_rows = {'headers': True, 'data': 0},
                    active_cell = {'column': 1, 'row': 0},
                    is_focused = True,
                    filter_action = 'none',
                    #hidden_columns = ['id_upload'],
                    id = 'analysis-data-upload-table',
                    data = generate_rows(20),
                    columns = db.upload_table_data_columns,
                    dropdown = db.upload_table_data_dropdowns,
                    row_deletable=True,
                    editable = True,
                ),
                html.P(id='analysis-table-hidden', style={'display':'none'}),

                html.Button('Add',id='add-rows-button',n_clicks=0),
                dcc.Input(value=10,type='number',id='add-rows-input'),
                'Rows',
            ], style = {
                'margin': '10px',
                'float': 'center'
            })
        ]
    ),
    QC_tab(),
    dcc.Tab(
        label='Full runlist',
        children = [
            dash_table.DataTable(
                db.data.to_dict('records'), [{'name': i, 'id': i} for i in db.data.columns]
            )
        ]
    )
]

app.layout = html.Div([
    html.H1(children='Quick data analysis'),
    dcc.Tabs(children = upload_tabs),
    html.Div(id='data-app-tabs'),
    html.Div(id='button-area')
])

def validate_row(active_cell) -> str:
    """Validates a data row"""
    return str(active_cell)

@app.callback(
    Output('analysis-table-hidden', 'children'),
    [Input('analysis-data-upload-table', 'data')])
def update_table(rows) -> str:
    df: pd.DataFrame = pd.DataFrame(rows)
    problems: str = 'No problems!'
    df.to_csv(str(uuid.uuid4()) + '.tsv',sep='\t')
    return problems

"""
@app.callback(Output('analysis-table-hidden', 'figure'),
              Input('datatable-upload-container', 'data'))
def display_graph(rows):
    df = pd.DataFrame(rows)

    if (df.empty or len(df.columns) < 1):
        return {
            'data': [{
                'x': [],
                'y': [],
                'type': 'bar'
            }]
        }
    return {
        'data': [{
            'x': df[df.columns[0]],
            'y': df[df.columns[1]],
            'type': 'bar'
        }]
    }



"""



@app.callback(Output('analysis-table-problems', 'children'), \
    [Input('analysis-data-upload-table', 'data'), \
        Input('analysis-data-upload-table', 'active_cell')])
def input_row_changed(data_table, active_cell):
    """Will do tricks to any changed input data row.
    """
    return str(validate_row(active_cell))
    #return str(active_cell) if active_cell else "Click the table"



if __name__ == '__main__':
    app.run(port='8050', debug=True)
    