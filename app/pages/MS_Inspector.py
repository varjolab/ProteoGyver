"""Dash app for inspection and analysis of MS performance based on TIC graphs.

This module provides a web interface for visualizing and analyzing Mass Spectrometry (MS) 
performance through Total Ion Current (TIC) graphs and related metrics. It allows users to 
explore MS runs based on time periods, run IDs, or sample types.

Features:
    - Interactive TIC visualization with animation controls
    - Multiple trace types support (TIC, MSn_unfiltered)
    - Supplementary metrics tracking (AUC, mean intensity, max intensity)
    - Sample type filtering and date range selection
    - Data export functionality in multiple formats (HTML, PNG, PDF, TSV)

Components:
    - Main TIC graph with adjustable opacity for temporal comparison
    - Three supplementary metric graphs (AUC, mean intensity, max intensity)
    - Control panel for MS selection, date ranges, and sample types
    - Animation controls for TIC visualization
    - Data download functionality

Dependencies:
    - dash: Web application framework
    - plotly: Interactive plotting library
    - pandas: Data manipulation and analysis
    - dash_bootstrap_components: Bootstrap components for Dash

Attributes:
    num_of_traces_visible (int): Maximum number of traces visible at once
    trace_color (str): RGB color code for traces
    trace_types (list): List of supported trace types (TIC, MSn_unfiltered)
    run_limit (int): Maximum number of runs that can be loaded at once

Notes:
    - The application enforces a run limit to maintain performance
    - Traces are displayed with decreasing opacity for temporal comparison
    - All graphs are synchronized for consistent data visualization
"""

import os
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash import callback, dcc, html, Input, Output, State, ctx, no_update
from datetime import datetime, date
from components import parsing
import re
from plotly import io as pio
from components import db_functions
import numpy as np
from element_styles import GENERIC_PAGE
import json
from io import StringIO
import logging
import zipfile
import uuid

pio.templates.default = 'plotly_white'
logger = logging.getLogger(__name__)
dash.register_page(__name__, path=f'/MS_inspector')
logger.warning(f'{__name__} loading')

def description_card() -> html.Div:
    """Creates the description card component for the dashboard.

    Returns:
        html.Div: A Dash HTML Div containing the dashboard title and descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("TIC visualizer and analysis"),
            html.H4("Visualize and assess MS performance."),
            html.Div(
                id="intro",
                children=[
                    html.P("Explore TICs of any MS run or sample set. Choose runs based on times, run IDs, or sample types."),
                    html.P(f"NOTE: only {RUN_LIMIT} runs can be loaded at once. If more than {RUN_LIMIT} runs are chosen, only the {RUN_LIMIT} most recent ones will be loaded."),
                    html.P("NOTE: If you want to switch to a different runset, you might want to reload the page. Otherwise it will load a bunch of runs right from the start. This will be fixed at some point, and there will be a note about it on the announcements page.")
                ]
            ),
        ],
    )

def generate_control_card() -> html.Div:
    """Creates the control card component with input widgets.

    Returns:
        html.Div: A Dash HTML Div containing controls for MS selection, date range,
            sample type selection, and run ID input.
    """
    ms_list: list = sorted(list(MASS_SPECS.keys()))
    return html.Div(
        id='control-card',
        children=[
            html.H5('Select MS'),
            dcc.Dropdown(
                id='ms-select',
                options=[{'label': i, 'value': i} for i in ms_list],
                value=ms_list[0],
            ),
            html.Br(),
            html.H5('Select Run time'),
            dcc.DatePickerRange(
                id='date-picker-select',
                display_format='YYYY.MM.DD',
                start_date = MINTIME,
                end_date = MAXTIME,
                min_date_allowed = MINTIME,
                max_date_allowed = MAXTIME,
                initial_visible_month=MAXTIME
            ),
            html.Br(),
            html.Br(),
            html.H5('Select data type'),
            dcc.Dropdown(
                id='ddadia-select',
                options=[{'label': i, 'value': i} for i in DATA_TYPES],
                value=DATA_TYPES[:],
                multi=True,
            ),
            html.Br(),
            html.H5('Or Input a list of run numbers'),
            dcc.Textarea(
                id='load-runs-from-runids',
                placeholder='Enter run ID numbers. Numbers can be separated by space, tab, or any of these symbols: ,;:',
                style={'width': '100%', 'height': 150},
            ),
            html.Br(),
            html.Div(
                id='button-div',
                children=[
                    dbc.Button(id='reset-btn', children='Reset', n_clicks=0),
                    dbc.Button(
                        id='load-runs-button',
                        children=dcc.Loading(
                            html.Div(
                                id='load-runs-spinner-div',
                                children='Load runs by selected parameters'
                            ),
                        )
                    )
                ]
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
    """Toggles the graph animation state between running and stopped.

    Args:
        n_clicks (int): Number of times the button has been clicked
        current (str): Current button text ('Start' or 'Stop')

    Returns:
        tuple: (disabled state, button text)
    """
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
    State('chosen-tics','children'),
    State('datatype-dropdown','value'),
    State('datatype-supp-dropdown','value'),
    State('trace-dict','data'),
    State('plot-data','data'),
    State('plot-max-y','data'),
    prevent_initial_call=True
    )
def update_tic_graph(_,__, ___, ____, tic_index: int, ticlist:list, datatype:str, supp_datatype:str, traces:dict, plot_data: str, max_y:dict) -> tuple:
    """Updates the TIC graphs and supplementary metrics plots.

    Args:
        prev_btn_nclicks (int): Number of clicks on previous button
        next_btn_nclicks (int): Number of clicks on next button
        tic_index (int): Current TIC index being displayed
        ticlist (list): List of TIC IDs to display
        datatype (str): Type of data to display (TIC or MSn)
        supp_datatype (str): Type of data for supplementary plots
        traces (dict): Dictionary containing trace data
        plot_data (str): JSON string containing plot data
        max_y (dict): Dictionary of maximum y-values for each trace type

    Returns:
        tuple: (tic_figure, auc_figure, mean_intensity_figure, max_intensity, return_tic_index)
            Contains updated Plotly figures and next TIC index

    Notes:
        The number of times the previous, next, or reset button is clicked is not used.
    """
    data_to_use: pd.DataFrame = pd.read_json(StringIO(plot_data),orient='split')
    ticlist.sort()
    next_offset: int = 0
    if ctx.triggered_id == 'reset-animation-button':
        tic_index = 0
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
        return_tic_index = 0
    these_tics: list = [traces[str(t)] for t in ticlist][:tic_index+1]
    these_tics = these_tics[-num_of_traces_visible:]
    tic_figure: go.Figure = go.Figure()
    these_tics = these_tics[:num_of_traces_visible]
    for i, trace_dict in enumerate(these_tics[::-1]):
        tic_figure.add_traces(trace_dict[datatype][str(i)])
    max_x = data_to_use[f'{datatype}_maxtime'].max()
    supp_graph_max_x: int = data_to_use.shape[0]
    auc_graph_max_y: int = data_to_use[f'{supp_datatype}_auc'].max()
    max_intensity_graph_max_y: int = data_to_use[f'{supp_datatype}_max_intensity'].max()
    mean_intensity_graph_max_y: int = data_to_use[f'{supp_datatype}_mean_intensity'].max()
    auc_graph_max_y += int(auc_graph_max_y/20)
    max_intensity_graph_max_y += int(max_intensity_graph_max_y/20)
    mean_intensity_graph_max_y += int(mean_intensity_graph_max_y/20)
    data_to_use = data_to_use.head(tic_index+1).copy()
    data_to_use['Run index'] = list(range(data_to_use.shape[0]))
    auc_figure: go.Figure = go.Figure(
        go.Scatter(
            x = data_to_use['Run index'],
            y = data_to_use[f'{supp_datatype}_auc'],
            mode = 'lines+markers',
            opacity = 1,
            line={'width': 1, 'color': trace_color},
            showlegend=False
        )
    )
    max_intensity: go.Figure = go.Figure(
        go.Scatter(
            x = data_to_use['Run index'],
            y = data_to_use[f'{supp_datatype}_max_intensity'],
            mode = 'lines+markers',
            opacity = 1,
            line={'width': 1, 'color': trace_color},
            showlegend=False
        ))
    mean_intensity_figure: go.Figure = go.Figure(
        go.Scatter(
            x = data_to_use['Run index'],
            y = data_to_use[f'{supp_datatype}_mean_intensity'],
            mode = 'lines+markers',
            opacity = 1,
            line={'width': 1, 'color': trace_color},
            showlegend=False
        )
    )
    tic_figure.update_layout(
        #title=setname,
        height=400,
        xaxis_range=[0,max_x],
        yaxis_range=[0,max_y[datatype]],
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

def sort_dates(date1, date2):
    """Sorts two dates in ascending order.

    Args:
        date1 (str): First date in YYYY-MM-DD format
        date2 (str): Second date in YYYY-MM-DD format

    Returns:
        tuple[str, str]: A tuple containing (earlier_date, later_date)
    """
    if datetime.strptime(date1, '%Y-%m-%d') > datetime.strptime(date2, '%Y-%m-%d'):
        return (date2,date1)
    return (date1, date2)

def delim_runs(runs):
    """Parses a string of run IDs into a list of valid run identifiers.

    Args:
        runs (str): String containing run IDs separated by spaces, tabs, commas, 
            semicolons, or colons

    Returns:
        list[str]: Sorted list of valid run IDs
    """
    retruns = []
    for run in sorted([
                s for s in re.split('|'.join(['\n',' ','\t',',',';',':']), runs) if len(s.strip())>0
            ]):
        try:
            retruns.append(run)
        except ValueError:
            continue
    return retruns
    

@callback(
    Output('chosen-tics', 'children'),
    Output('trace-dict', 'data'),
    Output('plot-data','data'),
    Output('plot-max-y','data'),
    Output('load-runs-spinner-div','children'),
    Output('start-stop-btn','n_clicks'),
    Output('run-ids-not-found','children'),
    Input('load-runs-button','n_clicks'),
    State('date-picker-select', 'start_date'),
    State('date-picker-select', 'end_date'),
    State('ddadia-select', 'value'),
    State('load-runs-from-runids','value'),
    State('load-runs-spinner-div','children'),
    prevent_initial_call=True
) 
def update_run_choices(_, start_date, end_date, sample_types, run_id_list, button_text) -> list:
    """Updates the list of runs based on selected criteria.

    Args:
        start_date (str): Start date
        end_date (str): End date
        sample_types (list): List of selected sample types
        run_id_list (str): String of run IDs to load
        button_text (str): Current button text

    Returns:
        list: Contains [chosen_tics, trace_dict, plot_data, max_y, button_text, button_clicks]
    """
    db_conn = db_functions.create_connection(database_file)
    not_found_ids = []
    if (run_id_list is None ) or (run_id_list.strip() == ''):
        start:str
        end: str
        start, end = sort_dates(start_date,end_date)
        start = start+' 00:00:00'
        end = end+' 23:59:59'
        #start: datetime = datetime.strptime(start+' 00:00:00',parameters['Config']['Time format'])
        #end: datetime = datetime.strptime(end+' 23:59:59',parameters['Config']['Time format'])
        chosen_runs: pd.DataFrame = db_functions.get_from_table(
            db_conn, # type: ignore
            'ms_runs',
            'run_time',
            (start, end),
            select_col=', '.join(REQUIRED_MAINCOLS),
            as_pandas=True,
            operator = 'BETWEEN',
            index_col = 'internal_run_id'
        )
        chosen_runs = chosen_runs[chosen_runs['sample_type'].isin(sample_types)]
        chosen_runs.sort_values(by='run_time',ascending=True, inplace=True)
        #chosen_runs.index = chosen_runs.index.astype(str)# And flip back to make passing trace_dict easier. Keys of the dict will be converted to strings when passed through data store.
        if chosen_runs.shape[0] > RUN_LIMIT:
            chosen_runs = chosen_runs.tail(RUN_LIMIT)
    else:
        run_ids = delim_runs(run_id_list)
        # Filter run_ids to only those found in KNOWN_SAMPLES
        not_found_ids = [rid for rid in run_ids if rid not in IDMAP]
        run_ids = [IDMAP[i] for i in run_ids if i in IDMAP]
        chosen_runs = db_functions.get_from_table_by_list_criteria(
            db_conn, 
            'ms_runs',
            'internal_run_id',
            run_ids,
            select_col=', '.join(REQUIRED_MAINCOLS),
            index_col = 'internal_run_id'
        )
        chosen_runs.sort_values(by='run_time',ascending=True, inplace=True)
    run_plots = db_functions.get_from_table_by_list_criteria(
        db_conn, 
        'ms_plots',
        'internal_run_id',
        list(chosen_runs.index),
        index_col = 'internal_run_id'
    )
    db_conn.close() # type: ignore
    not_found_text = None
    if len(not_found_ids) > 0:
        not_found_text: list[str] = ['IDs NOT FOUND IN DATABASE:']+not_found_ids+['\n']
    max_y = {}
    for t in trace_types:
        max_y[t] = run_plots[f'{t}_max_intensity'].max()
    trace_dict: dict = {}
    for runid, rundata in run_plots.iterrows():
        runid = str(runid)
        trace_dict[runid] = {}
        for tracename in trace_types:
            tracename = tracename
            trace_dict[runid][tracename] = {}
            for color_i in range(num_of_traces_visible):
                d = json.loads(rundata[f'{tracename}_trace'])
                d['line'] = {'color': trace_color,'width': 1}
                d['opacity'] = (1/num_of_traces_visible)*(num_of_traces_visible - color_i)
                trace_dict[runid][tracename][color_i] = pio.from_json(json.dumps(d))['data'][0]

    return (
        sorted(list(chosen_runs.index)),
        trace_dict,
        run_plots.to_json(orient='split'),
        max_y,button_text,1, not_found_text
    )

def ms_analytics_layout():
    """Creates the main layout for the MS analytics dashboard.

    The layout includes:
    - A description card explaining the dashboard's purpose
    - Control panel for selecting MS instruments, date ranges, and sample types
    - TIC visualization area with animation controls
    - Supplementary metrics graphs (AUC, mean intensity, max intensity)

    Returns:
        html.Div: Main container with all dashboard components organized in a 
            responsive grid layout
    """
    return html.Div(
        id="app-container",
        children=[
            html.Div(id='utilities',children = [
                dcc.Interval(
                    id='tic-analytics-interval-component',
                    interval=1.5*1000, # in milliseconds
                    n_intervals=0,
                    disabled=True
                ),
                html.Div(id='reset-tics', style={'display': 'none'}),
                html.Div(id='prev-btn-notifier', style={'display': 'none'}),
                html.Div(id='chosen-tics', children = [], style={'display': 'none'}),
                dcc.Store('trace-dict'),
                dcc.Store('plot-data'),
                dcc.Store('plot-max-y'),
                html.Div(id='tic-analytics-current-tic-idx', children = 0, style={'display': 'none'})
            ],style={'display':'none'}),
            dbc.Row([
                dbc.Col([
                    description_card(),
                    generate_control_card()
                ],
                width = 4),
                dbc.Col([
                    dbc.Row([
                        html.Div(
                            id = 'run-ids-not-found'
                        )
                    ]),
                    dbc.Row([
                        html.H4('TICs'),
                        html.Hr(),
                        dcc.Graph(id='tic-analytics-tic-graphs'),
                        html.Div([
                            html.P('Choose metric:    ',style={'float': 'left', 'margin': 'auto'}),
                            dcc.Dropdown(trace_types, 'TIC', id='datatype-dropdown'),
                            html.Br(),
                            dbc.Button(id='prev-btn', children='Previous', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                            dbc.Button(id='start-stop-btn', children='Start', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                            dbc.Button(id='next-btn', children='Next', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                            dbc.Button(id='reset-animation-button', children='Reset', n_clicks=0,style={'float': 'left','margin': 'auto'}),
                            dbc.Button(id='download-graphs-btn', children='Download Data', n_clicks=0, style={'float': 'left','margin': 'auto'}),
                            dcc.Download(id='download-graphs'),
                        ])
                    ]),
                    dbc.Row([
                            html.H4('Supplementary metrics'),
                            html.Hr(),
                            html.P('Choose metric for supplementary plots:    ',style={'float': 'left', 'margin': 'auto'}),
                            dcc.Dropdown(trace_types, 'TIC', id='datatype-supp-dropdown',style={'float': 'left', 'margin': 'auto'}),
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
        ],style=GENERIC_PAGE)

# Add new callback for download functionality
@callback(
    Output('download-graphs', 'data'),
    Input('download-graphs-btn', 'n_clicks'),
    State('tic-analytics-tic-graphs', 'figure'),
    State('auc-graph', 'figure'),
    State('mean-intensity-graph', 'figure'),
    State('max-intensity-graph', 'figure'),
    State('plot-data', 'data'),
    prevent_initial_call=True
)
def download_graphs(n_clicks, tic_fig, auc_fig, mean_fig, max_fig, plot_data):
    """Creates a ZIP file containing graphs and data for download.

    Saves the current graphs in multiple formats (HTML, PNG, PDF) and the underlying
    data as a TSV file. All files are bundled into a ZIP archive for download.

    Args:
        n_clicks (int): Number of clicks on download button
        tic_fig (plotly.graph_objects.Figure): TIC plot figure
        auc_fig (plotly.graph_objects.Figure): AUC plot figure
        mean_fig (plotly.graph_objects.Figure): Mean intensity plot figure
        max_fig (plotly.graph_objects.Figure): Max intensity plot figure
        plot_data (str): JSON string containing plot data

    Returns:
        dcc.send_bytes: ZIP file containing:
            - HTML, PNG, and PDF versions of all plots
            - TSV file with the underlying data
            - Named with timestamp prefix
            
    Raises:
        Exception: If there's an error during file creation or zipping process
    """
    if not n_clicks:
        return no_update
    
    # Use cache directory from parameters
    str_uuid = str(uuid.uuid4())
    temp_dir = os.path.join(*parameters['Data paths']['Cache dir'],'ms inspector',str_uuid)
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    zip_filename = f"{timestamp} MS Inspector.zip"
    
    try:
        # Save figures as HTML, PNG and PDF
        figs = {
            'Chromatogram': tic_fig,
            'AUC': auc_fig,
            'Mean Intensity': mean_fig,
            'Max Intensity': max_fig
        }
        
        for name, fig in figs.items():
            # Save as HTML
            pio.write_html(fig, os.path.join(temp_dir, f"{name}.html"))
            # Save as PNG
            pio.write_image(fig, os.path.join(temp_dir, f"{name}.png"))
            # Save as PDF
            pio.write_image(fig, os.path.join(temp_dir, f"{name}.pdf"))
        
        # Save data as TSV
        df = pd.read_json(StringIO(plot_data), orient='split')
        df.to_csv(os.path.join(temp_dir, 'Data.tsv'), sep='\t', index=True)
        
        # Create ZIP file
        with zipfile.ZipFile(os.path.join(temp_dir, zip_filename), 'w') as zipf:
            for file in os.listdir(temp_dir):
                if file != zip_filename:
                    zipf.write(os.path.join(temp_dir, file), file)
        
        # Read ZIP file and encode for download
        with open(os.path.join(temp_dir, zip_filename), 'rb') as f:
            zip_data = f.read()
        
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return dcc.send_bytes(zip_data, zip_filename)
    
    except Exception as e:
        logger.error(f"Error creating download package: {str(e)}")
        return no_update
    
parameters = parsing.parse_parameters('parameters.toml')
database_file = os.path.join(*parameters['Data paths']['Database file'])    

num_of_traces_visible = 7
trace_color = 'rgb(56, 8, 35)'
trace_types: list = ['TIC','BPC','MSn']
samplecols = ['file_name', 'file_name_clean', 'sample_id', 'sample_name' ]
REQUIRED_MAINCOLS = [
    'internal_run_id',
    'inst_model',
    'inst_serial_no',
    'run_date',
    'data_type'
] + samplecols
required_plot_cols = ['internal_run_id']
for tracename in trace_types:
    required_plot_cols.append(f'{tracename} trace')
    required_plot_cols.append(f'{tracename} mean_intensity')
    required_plot_cols.append(f'{tracename} auc')
    required_plot_cols.append(f'{tracename} max_intensity')

db_conn = db_functions.create_connection(database_file)
data = db_functions.get_from_table(
    db_conn,
    'ms_runs', 
    select_col = REQUIRED_MAINCOLS,
    as_pandas = True,
    pandas_index_col = 'internal_run_id'
).replace('',np.nan)
db_conn.close() # type: ignore

MASS_SPECS = {}
for row in data[['inst_model','inst_serial_no']].drop_duplicates().values:
    if not row[0]:
        continue
    if not row[1]:
        continue
    MASS_SPECS.setdefault(row[0], [])
    MASS_SPECS[row[0]].append(row[1])
DATA_TYPES = list(data['data_type'].unique())

IDMAP: dict = {}
# TODO: handle better runs with names or IDs that are repeated. e.g. two files with the same sample_id
del_runs = set()
for internal_run_id, row in data[samplecols].iterrows():
    for c in samplecols:
        if row[c] in IDMAP:
            del_runs.add(row[c]) # For now we just delete ambiguous entries
        else:
            IDMAP[row[c]] = internal_run_id
for d in del_runs:
    del IDMAP[d]

if data.shape[0] > 0:
    data['run_date'] = data.apply(lambda x: datetime.strptime(x['run_date'],parameters['Config']['Time format']),axis=1)
    d = data['run_date'].max()
    MAXTIME = date(d.year,d.month, d.day)
    d = data['run_date'].min()
    MINTIME = date(d.year,d.month, d.day)
    del data
    logger.warning(f'{__name__} preliminary data loaded')
    RUN_LIMIT = 100
    use_layout = ms_analytics_layout()
else:
    use_layout = html.Div('No MS runs in database.')
layout = use_layout