"""Dash app for inspection and analysis of MS performance based on TIC graphs"""

import os
from typing import Any
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash import callback, dcc, html, Input, Output, State, ALL, MATCH, ctx
import plotly.io as pio
from datetime import datetime, date
import json

dash.register_page(__name__, path='/')

with open('parameters.json', encoding = 'utf-8') as fil:
    parameters: dict = json.load(fil)


data_dir: str = os.path.join(*parameters['files']['data']['TIC information'])
data_file: str = os.path.join(data_dir, 'rundata.tsv')
# Read data
df: pd.DataFrame = pd.read_csv(data_file,sep='\t')
ms_list: list = sorted(list(df['MS'].unique()))
sample_list: list = sorted(list(df['sample_type'].unique()))
max_x: int = df['max_time'].max()
max_y: float = df['tic_max_intensity'].max()
df['run_time'] = df['run_time'].apply(lambda x: datetime.strptime(x,parameters['Config']['Time format']))

ticfile_dir: str = os.path.join(data_dir, 'TIC_files')
color = 'rgb(56, 8, 35)'
num_of_traces_visible = 7
traces: dict = {}
for dfname in os.listdir(ticfile_dir):
    if 'ipynb' in dfname:
        continue
    if 'unsmoothed' in dfname:
        continue
    runid: int = int(dfname.replace('.tsv',''))
    rundf: pd.DataFrame = pd.read_csv(os.path.join(ticfile_dir, f'{runid}.tsv'),sep='\t')
    traces[runid] = {}
    for color_i in range(num_of_traces_visible):
        traces[runid][color_i] = go.Scatter(
            x=rundf['Time'],
            y=rundf['SumIntensity'],
            mode = 'lines',
            opacity=(1/num_of_traces_visible)*(num_of_traces_visible - color_i),#(len(colors) - color_i)*(1/len(colors)),
            line = {'color' : color, 'width': 1},
            name = runid,
            #fill='tozeroy',
            #fillcolor='rgba(50,255,255,0)'
            )
df = df[df['run_id'].isin(traces.keys())]
df.sort_values(by='run_id',ascending=True)
#df['Run index'] = list(range(df.shape[0]))

def description_card() -> html.Div:
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("TIC visualizer and analysis"),
            html.H4("Visualize and assess MS performance."),
            html.Div(
                id="intro",
                children="Explore TICs of any MS run or sample set. Choose runs based on times, run IDs, or sample types.",
            ),
        ],
    )

def round_time(d) -> date:
    return date(d.year,d.month, d.day)

def generate_control_card() -> html.Div:
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id='control-card',
        children=[
            html.P('Select MS'),
            dcc.Dropdown(
                id='ms-select',
                options=[{'label': i, 'value': i} for i in ms_list],
                value=ms_list[0],
            ),
            html.Br(),
            html.P('Select Run time'),
            dcc.DatePickerRange(
                id='date-picker-select',
                display_format='YYYY.MM.DD',
                start_date = round_time(df['run_time'].min()),
                end_date = round_time(df['run_time'].max()),
                min_date_allowed = round_time(df['run_time'].min()),
                max_date_allowed = round_time(df['run_time'].max()),
                initial_visible_month=round_time(df['run_time'].max())
            ),
            html.Br(),
            html.Br(),
            html.P('Select sample type'),
            dcc.Dropdown(
                id='sampletype-select',
                options=[{'label': i, 'value': i} for i in sample_list],
                value=sample_list[:],
                multi=True,
            ),
            html.Br(),
            html.Div(
                id='reset-btn-div',
                children=dbc.Button(id='reset-btn', children='Reset', n_clicks=0),
            ),
        ],
    )


@callback(
    Output('tic-analytics-interval-component','disabled'),
    Output('start-stop-btn', 'children'),
    Input('start-stop-btn','n_clicks'),
    State('start-stop-btn','children'),
    prevent_initial_call=True
)
def toggle_graphs(n_clicks, current) -> tuple:
    if (int(n_clicks)==0) or (n_clicks is None) or (current == 'Stop'):
        text = 'Start'
        disabled = True
    elif current == 'Start':
        text = 'Stop'
        disabled = False
    return (disabled,text)

@callback(
    Output('reset-tics','children'),
    Input('reset-animation-button','n_clicks'),
)
def reset_graphs(_) -> int:
    return 0

@callback(
    Output('tic-analytics-tic-graphs', 'figure'),
    Output('auc-graph', 'figure'),
    Output('mean-intensity-graph', 'figure'),
    Output('max-intensity-graph', 'figure'),
    Output('tic-analytics-current-tic-idx','children'),
    Input('tic-analytics-interval-component', 'n_intervals'),
    Input('prev-btn','n_clicks'),
    Input('next-btn','n_clicks'),
    Input('reset-animation-button','n_clicks'),
    State('tic-analytics-current-tic-idx','children'),
    State('chosen-tics','children')
    )
def update_tic_graph(_,prev_btn_nclicks, next_btn_nclicks, __, tic_index: int, ticlist:list) -> tuple:
    ticlist.sort()
    next_offset: int = 0
    #tic_index = tic_index - prev_btn_nclicks + next_btn_nclicks
    if ctx.triggered_id == 'reset-animation-button':
        tic_index = 0# + prev_btn_nclicks - next_btn_nclicks
    elif ctx.triggered_id == 'prev-btn':
        tic_index -= 1
    elif ctx.triggered_id == 'next-btn':
        tic_index += 1
    else:
        next_offset = 1
    return_tic_index: int = tic_index + next_offset

    if tic_index < 0:
        tic_index = len(ticlist)-1
    elif tic_index > (len(ticlist)-1):
        tic_index = 0
    if return_tic_index >= len(ticlist):
        return_tic_index = 0# + prev_btn_nclicks - next_btn_nclicks
    these_tics: list = [traces[t] for t in ticlist][:tic_index+1]
    these_tics = these_tics[-num_of_traces_visible:]
    tic_figure: go.Figure = go.Figure()
    these_tics = these_tics[:num_of_traces_visible]
    for i, trace_dict in enumerate(these_tics[::-1]):
        tic_figure.add_traces(trace_dict[i])
    data_to_use: pd.DataFrame = df[df['run_id'].isin(ticlist)]
    supp_graph_max_x: int = data_to_use.shape[0]
    auc_graph_max_y: int = data_to_use['AUC'].max()
    max_intensity_graph_max_y: int = data_to_use['max_intensity'].max()
    mean_intensity_graph_max_y: int = data_to_use['mean_intensity'].max()
    auc_graph_max_y += auc_graph_max_y/20
    max_intensity_graph_max_y += max_intensity_graph_max_y/20
    mean_intensity_graph_max_y += mean_intensity_graph_max_y/20
    data_to_use = data_to_use.head(tic_index+1).copy()

    data_to_use['Run index'] = list(range(data_to_use.shape[0]))
    auc_figure: go.Figure = go.Figure(
        go.Scatter(
            x = data_to_use['Run index'],
            y = data_to_use['AUC'],
            mode = 'lines+markers',
            opacity = 1,
            line={'width': 1, 'color': color},
            showlegend=False
        )
    )
    max_intensity: go.Figure = go.Figure(
        go.Scatter(
            x = data_to_use['Run index'],
            y = data_to_use['max_intensity'],
            mode = 'lines+markers',
            opacity = 1,
            line={'width': 1, 'color': color},
            showlegend=False
        ))
    mean_intensity_figure: go.Figure = go.Figure(
        go.Scatter(
            x = data_to_use['Run index'],
            y = data_to_use['mean_intensity'],
            mode = 'lines+markers',
            opacity = 1,
            line={'width': 1, 'color': color},
            showlegend=False
        )
    )

    tic_figure.update_layout(
        #title=setname,
        height=400,
        xaxis_range=[0,max_x],
        yaxis_range=[0,max_y],
        #template='plotly_dark',
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        #paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=5, r=5, t=20, b=5)
    )
    auc_figure.update_layout(
        #title=setname,
        height=150,
        xaxis_range=[0,supp_graph_max_x],
        yaxis_range=[0,auc_graph_max_y],
        #template='plotly_dark',
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        #paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=5, r=5, t=20, b=5)
    )
    max_intensity.update_layout(
        #title=setname,
        height=150,
        xaxis_range=[0,supp_graph_max_x],
        yaxis_range=[0,max_intensity_graph_max_y],
        #template='plotly_dark',
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        #paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=5, r=5, t=20, b=5)
    )
    mean_intensity_figure.update_layout(
        #title=setname,
        height=150,
        xaxis_range=[0,supp_graph_max_x],
        yaxis_range=[0,mean_intensity_graph_max_y],
        #template='plotly_dark',
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        #paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=5, r=5, t=20, b=5)
    )


    return tic_figure, auc_figure, mean_intensity_figure, max_intensity, return_tic_index

@callback(
    Output('chosen-tics', 'children'),
    [
        Input('date-picker-select', 'start_date'),
        Input('date-picker-select', 'end_date'),
        Input('sampletype-select', 'value'),
    ],
)
# untested
def update_run_choices(start, end, sample_types) -> list:
    start: datetime = datetime.strptime(start+'T00:00:00+0200',parameters['Config']['Time format'])
    end: datetime = datetime.strptime(end+'T23:59:59+0200',parameters['Config']['Time format'])
    chosen_runs: pd.DataFrame = df[(df['run_time']>=start) & (df['run_time']<=end)]
    chosen_runs = chosen_runs[chosen_runs['sample_type'].isin(sample_types)]
    return list(chosen_runs['run_id'].values)


layout = html.Div(
    id="app-container",
    children=[
        html.Div(id='utilities',children = [
            dcc.Interval(
                id='tic-analytics-interval-component',
                interval=2*1000, # in milliseconds
                n_intervals=0,
                disabled=True
            ),
            html.Div(id='reset-tics', style={'display': 'none'}),
            html.Div(id='prev-btn-notifier', style={'display': 'none'}),
            html.Div(id='chosen-tics', children = [], style={'display': 'none'}),
            html.Div(id='tic-analytics-current-tic-idx', children = 0, style={'display': 'none'}),
        ]),
        dbc.Row([
            dbc.Col([
                description_card(),
                generate_control_card()
            ],
            width = 4),
            dbc.Col([
                dbc.Row([
                    html.H4('TICs'),
                    html.Hr(),
                    dcc.Graph(id='tic-analytics-tic-graphs'),
                    html.Div([
                        dbc.Button(id='prev-btn', children='Previous', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                        dbc.Button(id='start-stop-btn', children='Start', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                        dbc.Button(id='next-btn', children='Next', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                        dbc.Button(id='reset-animation-button', children='Reset', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                    ])
                ]),
                dbc.Row([
                        html.H4('Supplementary metrics'),
                        html.Hr(),
                        html.B('Area under the curve'),
                        dcc.Graph(id='auc-graph'),
                        html.B('Mean intensity'),
                        dcc.Graph(id='mean-intensity-graph'),
                        html.B('Max intensity'),
                        dcc.Graph(id='max-intensity-graph'),
                ])
            ],
            width = 8)
        ])
    ])
