"""Dash app for data upload"""

import io
import json
import base64
from matplotlib.pyplot import legend
import numpy as np
from random import sample
import plotly.graph_objects as go
from dash import Dash, html, dcc,dash_table, callback
from dash.dependencies import Input, Output, State
import dash
import pandas as pd
import dash_bootstrap_components as dbc
from plotly import express as px
from dash_bootstrap_templates import load_figure_template
from DbEngine import DbEngine
import os
from utilitykit import plotting
import plotly.io as pio
import data_functions
import figure_generation
import plotly.express as px
from plotly import tools as tls

# db will house all data, keep track of next row ID, and validate any new data
db: DbEngine = DbEngine()
#app: Dash = Dash(__name__)
#server: app.server = app.server

upload_style: dict = {
                    'width': '40%',
                    'height': '60px',
                    #'lineHeight': '20px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'alignContent': 'center',
                    'margin': 'auto',
                    #'float': 'right'
}

upload_a_style: dict = {
    'color': '#1EAEDB',
    'cursor': 'pointer',
    'text-decoration': 'underline'
}
figure_templates = [
            'plotly',
            'plotly_white',
            'plotly_dark',
            'ggplot2',
            'seaborn',
            'simple_white'
            ]
dash.register_page(__name__)

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

@callback(
    Output('workflow-choice','data'),
    Input('workflow-dropdown','value'),
    Input('upload-data-file', 'contents'),
    Input('upload-sample_table-file', 'contents'),
    State('session-uid','children')
)
def set_workflow(workflow_setting_value, _,__,session_uid) -> str:
    with open(db.get_cache_file(session_uid + '_workflow-choice.txt'),'w',encoding='utf-8') as fil:
        fil.write(workflow_setting_value)
    return str(workflow_setting_value)

@callback([
        Output('output-data-upload', 'data'),
        Output('output-data-upload-problems','children'),
        Output('figure-template-choice', 'data'),
        Output('placeholder', 'children')
    ],
    Input('figure-template-dropdown', 'value'),
    Input('upload-data-file', 'contents'),
    State('upload-data-file', 'filename'),
    Input('upload-sample_table-file', 'contents'),
    State('upload-sample_table-file', 'filename'),
    State('session-uid','children')
    )
def process_input_tables(
                        figure_template_dropdown_value,
                        data_file_contents,
                        data_file_name,
                        sample_table_file_contents,
                        sample_table_file_name,
                        session_uid
                        ) -> tuple:
    if figure_template_dropdown_value:
        pio.templates.default = figure_template_dropdown_value
    return_message: list = []
    return_dict: dict = {}
    if data_file_contents is None:
        return_message.append('Missing data table')
    if sample_table_file_contents is None:
        return_message.append('Missing sample_table')
    if len(return_message)==0:
        table, column_map, sample_groups = data_functions.parse_data(
            data_file_contents,
            data_file_name,
            sample_table_file_contents,
            sample_table_file_name
            )
        return_dict['table'] = table.to_json(orient='split')
        return_dict['column map'] = column_map
        return_dict['sample groups'] = sample_groups
        with open(db.get_cache_file(session_uid + '_data_dict.json'),'w',encoding='utf-8') as fil:
            json.dump(return_dict,fil)

        return_message.append('Upload successful')
    return return_dict, ' ; '.join(return_message), figure_template_dropdown_value, ''

def add_replicate_colors(data_df, column_to_replicate) -> None:

    need_cols: int = list(
            {
                column_to_replicate[sname] for sname in \
                data_df.index.unique() \
                if sname in column_to_replicate
            }
        )
    colors: list = plotting.get_cut_colors(number_of_colors = len(need_cols))
    colors = plotting.cut_colors_to_hex(colors)
    colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
    color_column:list = []    
    for sn in data_df.index.values:
        color_column.append(colors[column_to_replicate[sn]])
    data_df.loc[:,'Color'] = color_column

@callback(
    Output('qc-plot-container','children'),
    Input('placeholder', 'children'),
    State('output-data-upload', 'data')
    )
def quality_control_charts(_:str, data_dictionary:dict)->list:
    figures:list = []
    if 'table' in data_dictionary:
        data_table: pd.DataFrame = pd.read_json(data_dictionary['table'],orient='split')
        count_data: pd.DataFrame = data_functions.get_count_data(data_table)
        add_replicate_colors(count_data, data_dictionary['sample groups'])
        rep_colors: dict = {}
        for sname,sample_color_row in count_data[[ 'Color']].iterrows():
            rep_colors[sname] = sample_color_row['Color']
        data_dictionary['Replicate colors'] = rep_colors
        figures.append(figure_generation.protein_count_figure(count_data))

        na_data: pd.DataFrame = data_functions.get_na_data(data_table)
        na_data['Color'] = [rep_colors[sample_name] for sample_name in na_data.index.values]
        figures.append(figure_generation.missing_figure(na_data))

        sumdata: pd.DataFrame = data_functions.get_sum_data(data_table)
        sumdata['Color'] = [rep_colors[sample_name] for sample_name in sumdata.index.values]
        figures.append(figure_generation.sum_value_figure(sumdata))

        avgdata: pd.DataFrame = data_functions.get_avg_data(data_table)
        avgdata['Color'] = [rep_colors[sample_name] for sample_name in avgdata.index.values]
        figures.append(figure_generation.avg_value_figure(avgdata))

        figures.append(figure_generation.distribution_figure(data_table, rep_colors,data_dictionary['sample groups']))
        # Long-term: 
        # - protein counts compared to previous similar samples
        # - sum value compared to previous similar samples
        # - Person-to-person comparisons: protein counts, intensity/psm totals
    return figures

@callback(
    Output('download-sample_table-template', 'data'),
    Input('button-download-sample_table-template', 'n_clicks'),
    prevent_initial_call=True,
)
def sample_table_download(n_clicks):
    return dcc.send_file(db.request_file('assets','example-sample_table'))

@callback(
    Output('download-datafile-example', 'data'),
    Input('button-download-datafile-example', 'n_clicks'),
    prevent_initial_call=True,
)
def download_data_table(n_clicks):
    return dcc.send_file(db.request_file('assets','example-data_file'))

layout: html.Div = html.Div(
    children=[        
        html.Div(
            [dbc.Row(
                [
                    dbc.Col(
                        html.Div('Workflow:'),
                        width=4
                    ),
                    dbc.Col(
                        html.Div('Figure Theme:'),
                        width=4
                    ),
                    dbc.Col(
                        dbc.Button('Download sample_table template',\
                            id='button-download-sample_table-template'),
                        width=4
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Select(
                            value = db.default_workflow,
                            options = [
                                {'label': item, 'value': item} for item in  db.implemented_workflows
                            ],
                            id='workflow-dropdown',
                        ),
                        #width=4
                    ),
                    dbc.Col(
                        dbc.Select(
                            value = figure_templates[0],
                            options = [
                                {'label': item, 'value': item} for item in figure_templates
                            ],
                            id='figure-template-dropdown',
                        ),
                        #width=4
                    ),
                    dbc.Col(
                        dbc.Button(
                            'Download Datafile example',
                            id='button-download-datafile-example'
                        ),
                        #width=4
                    )
                ]
            ),
            
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            id='upload-data-file',
                            children=html.Div([
                                'Drag and drop or ',
                                html.A(
                                    'select',
                                    style = upload_a_style
                                    ),
                                ' Data file'
                            ]
                            ),
                            style=upload_style,
                            multiple=False
                        )
                    ),
                    dbc.Col(
                        dcc.Upload(
                            id='upload-sample_table-file',
                            children=html.Div([
                                'Drag and drop or ',
                                html.A(
                                    'select', 
                                    style = upload_a_style
                                    ),
                                ' sample_table file'
                            ]),
                            style=upload_style,
                            multiple=False
                        )
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(id='output-data-upload-problems')
                )
            )]
        ),
        
        dcc.Download(id='download-sample_table-template'),
        dcc.Download(id='download-datafile-example'),
        html.Div(id='placeholder'),
        dcc.Store(id='output-data-upload'),
        dcc.Store(id='figure-template-choice'),
        dcc.Store(id='workflow-choice'),
        dbc.Container(id='qc-plot-container', style={
            'margin': '0px',
            'float': 'center'
            }
        ),
        html.Div(id='data-app-tabs'),
        html.Div(id='button-area')
        ]
    )