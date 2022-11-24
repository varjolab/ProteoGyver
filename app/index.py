"""Dash app for data upload"""

import io
import json
import base64
from matplotlib.pyplot import legend
import numpy as np
from random import sample

from dash import Dash, html, dcc
from dash import dash_table
import pandas as pd
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from plotly import express as px
from dash_bootstrap_templates import load_figure_template
from DbEngine import DbEngine

from utilitykit import plotting
import plotly.io as pio
import plotly.express as px

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


@app.callback(
    Output('placeholder', 'children'),
    Input('figure-template-dropdown', 'value'),
    #State('output-data-upload','data')
)
def update_output(value) -> str:
    if value:
        pio.templates.default = value
    #if data:
    #    quality_control_charts(data)
    return ''


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

    table = table.replace(0,np.nan)
    column_map: dict = {}
    sample_groups: dict = {}
    rename_columns: dict = {oldname: oldname.rsplit('\\',maxsplit=1)[-1].rsplit('/')[-1] for oldname in table.columns}
    table = table.rename(columns=rename_columns)
    expdesign['Sample name'] = [
            oldvalue.rsplit('\\',maxsplit=1)[-1].rsplit('/')[-1] for oldvalue in expdesign['Sample name'].values
        ]
    protein_id_column: str = 'Protein.Group'
    keep_columns: set = set()
    for col in table.columns:
        if col not in expdesign['Sample name'].values: 
            continue
        sample_group = expdesign[expdesign['Sample name']==col].iloc[0]['Sample group']
        # If no value is available for sample in the expdesign (but sample column name is there for some reason), discard column
        if pd.isna(sample_group):
            continue
        newname: str = str(sample_group)
        # We expect replicates to not be specifically named; they will be named here.
        i: int = 0
        if newname[0].isdigit():
            newname = f'Sample_{newname}'
        while f'{newname}_Rep_{i}' in column_map:
            i+=1
        newname_to_use: str = f'{newname}_Rep_{i}'
        if newname not in sample_groups:
            sample_groups[newname] = []
        sample_groups[newname].append(newname_to_use)
        sample_groups[newname_to_use] = newname
        column_map[newname_to_use] = col
        keep_columns.add(newname_to_use)
    table: pd.DataFrame = table.rename(columns={v:k for k,v in column_map.items()})
    table.index = table[protein_id_column]
    table = table[[c for c in table.columns if c in keep_columns]]

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
        return_dict['sample groups'] = sample_groups
        
        return_message.append('Upload successful')
    return return_dict, ' ; '.join(return_message)

def add_replicate_colors(data_df, column_to_replicate):

    need_cols: int = list(
            {
                column_to_replicate[sname] for sname in \
                data_df['Sample name'].unique() \
                if sname in column_to_replicate
            }
        )
    colors: list = plotting.get_cut_colors(number_of_colors = len(need_cols))
    colors = plotting.cut_colors_to_hex(colors)
    colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
    color_column:list = []    
    for sn in data_df['Sample name'].values:
        color_column.append(colors[column_to_replicate[sn]])
    data_df.loc[:,'Color'] = color_column



def missing_figure(na_data) -> dcc.Graph:
    na_figure: px.bar = px.bar(
        na_data,
        x='Sample name',
        y='Missing value %',
        height=500,
        width=750,
        title='Missing values per sample',
        color='Color',
        color_discrete_map='identity')
    na_figure.update_xaxes(type='category')
    return dcc.Graph(id='protein-count-figure',figure=na_figure)

def protein_count_figure(count_data) -> dcc.Graph:
    count_figure: px.bar = px.bar(
            count_data, 
            x='Sample name',
            y='Protein count',
            height=500,
            width=750,
            title='Proteins per sample',
            color='Color',
            color_discrete_map='identity'
            )
    count_figure.update_xaxes(type='category')
    return dcc.Graph(id='protein-count-figure',figure=count_figure)


def get_na_data(data_table) -> pd.DataFrame:
    return ((data_table.\
    isna().sum() / data_table.shape[0]) * 100).\
    to_frame(name='Missing value %').\
    reset_index().\
    rename(columns={'index': 'Sample name'})

def get_count_data(data_table) -> pd.DataFrame:
    return data_table.\
           notna().sum().\
           to_frame(name='Protein count').\
           reset_index().\
           rename(columns={'index': 'Sample name'})

@app.callback(
    Output('qc-plot-container','children'),
    Input('output-data-upload','data'),
    Input('placeholder', 'children'),
    )
def quality_control_charts(data_dictionary:dict, _:str)->list:
    figures:list = []
    if 'table' in data_dictionary:
        data_table: pd.DataFrame = pd.read_json(data_dictionary['table'],orient='split')
        count_data: pd.DataFrame = get_count_data(data_table)
        add_replicate_colors(count_data, data_dictionary['sample groups'])
        rep_colors: dict = {}
        for _,sample_color_row in count_data[['Sample name', 'Color']].drop_duplicates().iterrows():
            rep_colors[sample_color_row['Sample name']] = sample_color_row['Color']
        data_dictionary['Replicate colors'] = rep_colors
        figures.append(protein_count_figure(count_data))
        
        na_data: pd.DataFrame = get_na_data(data_table)
        na_data['Color'] = [rep_colors[sample_name] for sample_name in na_data['Sample name'].values]
        figures.append(missing_figure(na_data))
        
        # proteins per sample plot
        # missing per sample plot

        # Intensity/specs per sample plot
        # Average per sample plot
        # Next: imputation

        # Long-term: 
        # - protein counts compared to previous similar samples
        # - sum value compared to previous similar samples
        # - Person-to-person comparisons: protein counts, intensity/psm totals


    return figures

def upload_tab():
    return dcc.Tab(
        label='Upload data',
        children = [
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
            'Figure theme: ',
            dcc.Dropdown([
                'plotly',
                'plotly_white',
                'plotly_dark',
                'ggplot2',
                'seaborn',
                'simple_white'],
                id='figure-template-dropdown'),
            html.Div(id='placeholder'),
            html.Div(id='output-data-upload-problems'),
            dcc.Store(id='output-data-upload'),
            dbc.Container(id='qc-plot-container', style={
                'margin': '0px',
                'float': 'center'
                }
            )
        ]
    )
upload_tabs: list = [
    upload_tab(),
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

if __name__ == '__main__':
    app.run(port='8050', debug=True)
    