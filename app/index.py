"""Dash app for data upload"""

import io
import json
import base64
from matplotlib.pyplot import legend
import numpy as np
from random import sample
import plotly.graph_objects as go
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
from plotly import tools as tls

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
        i: int = 1
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
                Output('output-data-upload-problems','children'),
                Output('figure-template-choice', 'data'),
                Output('placeholder', 'children')
              ],
              Input('figure-template-dropdown', 'value'),
              Input('upload-data-file', 'contents'),
              State('upload-data-file', 'filename'),
              Input('upload-expdesign-file', 'contents'),
              State('upload-expdesign-file', 'filename'),)
def process_input_tables(
                        figure_template_dropdown_value,
                        data_file_contents,
                        data_file_name,
                        expdesign_file_contents,
                        expdesign_file_name
                        ) -> tuple:
    if figure_template_dropdown_value:
        pio.templates.default = figure_template_dropdown_value
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
    return return_dict, ' ; '.join(return_message), figure_template_dropdown_value, ''

def add_replicate_colors(data_df, column_to_replicate):

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


def bar_plot(value_df,title,y=0) -> px.bar:
    figure: px.bar = px.bar(
            value_df,
            x=value_df.index,#'Sample name',
            y=value_df.columns[y],
            height=500,
            width=750,
            title=title,
            color='Color',
            color_discrete_map='identity'
            )
    figure.update_xaxes(type='category')
    return figure

def sum_value_figure(sum_data) -> dcc.Graph:
    sum_figure = bar_plot(sum_data,title='Value sum per sample')
    return dcc.Graph(id='value-sum-figure', figure=sum_figure)

def avg_value_figure(avg_data) -> dcc.Graph:
    avg_figure = bar_plot(avg_data,title='Value mean per sample')
    return dcc.Graph(id='value-sum-figure', figure=avg_figure)

def missing_figure(na_data) -> dcc.Graph:
    na_figure: px.bar = bar_plot(na_data,title='Missing values per sample')
    return dcc.Graph(id='protein-count-figure',figure=na_figure)

def protein_count_figure(count_data) -> dcc.Graph:
    count_figure: px.bar = bar_plot(count_data,title='Proteins per sample')
    return dcc.Graph(id='protein-count-figure',figure=count_figure)

def get_na_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = ((data_table.\
    isna().sum() / data_table.shape[0]) * 100).\
    to_frame(name='Missing value %')
    data.index.name = 'Sample name'
    return data

def distribution_figure(data_table, color_dict, sample_groups) -> dcc.Graph:
    data: list = []
    for col in data_table.columns:
        data.append(
            go.Violin(
                    x=np.log2(data_table[col]),
                    line_color = color_dict[col],
                    name=col,
                    legendgroup = sample_groups[col],
                    orientation = 'h',
                    side = 'positive',
                    width = 3,
                    points = False
                )
            )
    fheight = 40*len(data)
    layout: go.Layout = go.Layout(
        title = 'Value distribution',
        xaxis={
            'title': 'Log2 value',
            'showgrid': False,
            'zeroline': False
            },
        yaxis={
            'title': 'Sample',
            'showgrid': True,
            },
        legend_traceorder='grouped+reversed',
        height=fheight,
        width=750
    )

    fig: go.Figure = go.Figure(
            layout = layout,
            data=data,
            #height=500,
            #width=750,
        )

    return dcc.Graph(id='distribution-figure',figure=fig)

def get_count_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.\
           notna().sum().\
           to_frame(name='Protein count')
    data.index.name = 'Sample name'
    return data

def get_sum_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.sum().\
           to_frame(name='Value sum')
    data.index.name = 'Sample name'
    return data

def get_avg_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.mean().\
           to_frame(name='Value mean')
    data.index.name = 'Sample name'
    return data

@app.callback(
    Output('qc-plot-container','children'),
    Input('placeholder', 'children'),
    State('output-data-upload', 'data')
    )
def quality_control_charts(_:str, data_dictionary:dict)->list:
    figures:list = []
    if 'table' in data_dictionary:
        data_table: pd.DataFrame = pd.read_json(data_dictionary['table'],orient='split')
        count_data: pd.DataFrame = get_count_data(data_table)
        add_replicate_colors(count_data, data_dictionary['sample groups'])
        rep_colors: dict = {}
        for sname,sample_color_row in count_data[[ 'Color']].iterrows():
            rep_colors[sname] = sample_color_row['Color']
        data_dictionary['Replicate colors'] = rep_colors
        figures.append(protein_count_figure(count_data))

        na_data: pd.DataFrame = get_na_data(data_table)
        na_data['Color'] = [rep_colors[sample_name] for sample_name in na_data.index.values]
        figures.append(missing_figure(na_data))

        sumdata: pd.DataFrame = get_sum_data(data_table)
        sumdata['Color'] = [rep_colors[sample_name] for sample_name in sumdata.index.values]
        figures.append(sum_value_figure(sumdata))

        avgdata: pd.DataFrame = get_avg_data(data_table)
        avgdata['Color'] = [rep_colors[sample_name] for sample_name in avgdata.index.values]
        figures.append(avg_value_figure(avgdata))

        figures.append(distribution_figure(data_table, rep_colors,data_dictionary['sample groups']))


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
                id='figure-template-dropdown',
                value='plotly_white'),
            html.Div(id='placeholder'),
            html.Div(id='output-data-upload-problems'),
            dcc.Store(id='output-data-upload'),
            dcc.Store(id='figure-template-choice'),
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
    