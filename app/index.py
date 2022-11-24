"""Dash app for data upload"""

import io
import json
import base64
from random import sample
from dash import Dash, html, dcc
from dash import dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from plotly import express as px
from dash_bootstrap_templates import load_figure_template
from DbEngine import DbEngine
import uuid

# db will house all data, keep track of next row ID, and validate any new data
db: DbEngine = DbEngine()
app: Dash = Dash(__name__)
server: app.server = app.server

upload_style: dict = {
                    'width': '40%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': 'auto',
                    'float': 'right'
}


def get_next_n_ids(number:int) -> list:
    """Returns a list of next number ids for the run data table
    """
    return list(range(db.last_id, db.last_id + number))

def QC_tab() -> dcc.Tab:
    return dcc.Tab(
            label='QC',
            id='QCtab',
            children = [
                dcc.Graph(id='qc-protein-count-graph',figure=barchart()),
                dcc.Graph(id='qc-sum-value-graph',figure=barchart()),
                dcc.Graph(id='qc-protein-count-compared-to-previous',figure=barchart()), 
                dcc.Graph(id='qc-sum-value-compared-to-previous',figure=barchart())
            ]
        )

#@app.callback(
#    Output('graph','figure'),
#    input('QCtab', 'value'))
def barchart():
    df:pd.DataFrame = pd.DataFrame(data=[['sample1',3000],['sample2',4000],['sample5',3500]],columns=['Sample name','Protein count'])
    fig:px.bar = px.bar(df,x='Sample name',y='Protein count')
    return fig

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

#@app.callback(Output('intermediate-value','data'), Input(''))

def read_df_from_content(content, filename) -> pd.DataFrame:
    _, content_string = content.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    f_end:str = filename.rsplit('.',maxsplit=1)[-1]
    data = None
    if f_end=='csv':
        data:pd.DataFrame = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    elif f_end in ['tsv','tab','txt']:
        data:pd.DataFrame = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')),sep='\t')
    elif f_end in ['xlsx','xls']:
        data:pd.DataFrame = pd.read_excel(io.StringIO(decoded_content))
    return data


def parse_data(data_content, data_name, expdes_content, expdes_name) -> list:
    table: pd.DataFrame = read_df_from_content(data_content,data_name)
    expdesign: pd.DataFrame = read_df_from_content(expdes_content,expdes_name)

    column_map: dict = {}
    sample_groups: dict = {}
    for col in table.columns[1:]:
        if col not in expdesign['Sample name']: continue
        newname: str = expdesign[expdesign['Sample name']==col].iloc[0]['Sample group']
        i: int = 0
        while f'{newname}_{i}' in column_map:
            i+=1
        newname_to_use: str = f'{newname}_{i}'
        if newname not in sample_groups:
            sample_groups[newname] = []
        sample_groups[newname].append(newname_to_use)
        column_map[newname_to_use] = col
    table: pd.DataFrame = table.rename(columns={v:k for k,v in column_map.items()})

    return [table, column_map, sample_groups]

@app.callback([
                Output('output-data-upload', 'data'),
                Output('output-data-upload-problems','children')
                ],
              Input('upload-data-file', 'contents'),
              State('upload-data-file', 'filename'),
              Input('upload-expdesign-file', 'contents'),
              State('upload-expdesign-file', 'filename'),)
def process_input_tables(
                        data_file_contents,
                        data_file_name,
                        expdesign_file_contents,
                        expdesign_file_name
                        ) -> tuple:
    return_message: list = []
    return_dict: dict = {}
    if data_file_contents is None:
        return_message.append('Missing data table')
    if expdesign_file_contents is None:
        return_message.append('Missing expdesign')
    if len(return_message)==0:
        table, column_map, sample_groups = parse_data(
            data_file_contents,
            data_file_name,
            expdesign_file_contents,
            expdesign_file_name
            )
        return_dict['table'] = table.to_json(orient='split')
        return_dict['column map'] = column_map
        return_dict['sample_groups'] = sample_groups
        
        return_message.append('Upload successful')
    return return_dict, ' ; '.join(return_message)
#{'all_names': all_names, 'data': data}



def upload_tab():
    return dcc.Tab(
        label='Upload data',
        children = [
            dbc.Container([
                dcc.Upload(
                        id='upload-data-file',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('select'),
                            ' Data file'
                        ]),
                        style=upload_style,
                        multiple=False
                ),
                dcc.Upload(
                        id='upload-expdesign-file',
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('select'),
                            ' ExpDesign file'
                        ]),
                        style=upload_style,
                        multiple=False
                ),
                html.Button('Download ExpDesign template',\
                    id='button-download-expdesign-template'),
                html.Div(id='output-data-upload-problems'),
                dcc.Store(id='output-data-upload'),
            ], style={
                'margin': '0px',
                'float': 'center'
            }),
            dbc.Container([
                dbc.Alert(id='analysis-table-problems'),
                dash_table.DataTable(
                    page_size=50,
                    active_cell = {'column': 1, 'row': 0},
                    is_focused = True,
                    filter_action = 'none',
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
    )
upload_tabs: list = [
    upload_tab(),
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
    