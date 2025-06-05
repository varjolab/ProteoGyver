"""Dash app for visualization and analysis of microscopy images.

This module provides functionality to:
1. Load and parse .lif microscopy files
2. Display individual channels with customizable colormaps
3. Generate merged views of selected channels
4. Allow synchronized zooming across all views

The app uses Dash callbacks to handle user interactions and update visualizations dynamically.
"""

from dash import html, dcc, callback, MATCH, ctx, no_update, ALL, register_page
import plotly.express as px
from components.ui_components import upload_area
import logging
import numpy as np
import base64
import tempfile
from datetime import datetime
from plotly import graph_objects as go
from readlif.reader import LifFile
from components.parsing import parse_parameters
import dash_bootstrap_components as dbc
import plotly.express as px
import uuid
import dash_uploader as du
from dash.dependencies import Input, Output, State

from element_styles import UPLOAD_INDICATOR_STYLE, UPLOAD_STYLE,GENERIC_PAGE

register_page(__name__, path='/colocalizer')
parameters = parse_parameters('parameters.toml')
logger = logging.getLogger(__name__)
logger.warning(f'{__name__} loading')
available_cmaps = 'blackbody blues icefire bugn bupu cividis electric greens hot ylorbr gnbu greens amp thermal ice dense pubugn purd purp algae hot gray greys inferno magma pubu oranges reds purples rdpu tempo teal'.split()
available_cmaps.extend('Blackbody Bluered Blues Cividis Earth Electric Greens Greys Hot Jet Picnic Portland Rainbow RdBu Reds Viridis YlGnBu YlOrRd'.lower().split())
available_cmaps = sorted(list(set(available_cmaps)))#[u for u in usable if u.lower() in available_cmaps]


@callback(
    Output('image-data-store','data'),
    Output('microscopy-image-metainfo','children'),
    Output('microscopy-uploader-success', 'style'),
    Input('microscopy-uploader', 'contents'),
    State('microscopy-uploader', 'filename'),
    State('microscopy-uploader', 'last_modified'),
    State('microscopy-uploader-success', 'style'),
    prevent_initial_call = True
)
def handle_uploaded_data_table(file_contents, file_name, mod_date, current_upload_style) -> tuple:
    """Handle uploaded microscopy file and update UI elements.
    
    Args:
        file_contents: Base64 encoded file contents
        file_name: Name of uploaded file
        mod_date: Modification timestamp of file
        current_upload_style: Current style dictionary for upload indicator
        
    Returns:
        tuple: (parsed image data, UI elements for file info, updated upload style)
    """
    if file_contents is not None:
        current_upload_style['background-color'] = 'green'
        mod_date = datetime.fromtimestamp(mod_date).strftime('%Y-%m-%d %H:%M:%S')
        return (
            load_image(file_contents),
            [
                f'Name: {file_name}',
                html.Br(),
                f'Modified: {mod_date} (UTC)',
            ],
            current_upload_style
            )
    return no_update,no_update,no_update

def load_image(image_data:str):
    """Parse and load microscopy image data from base64 encoded string.
    
    Args:
        image_data: Base64 encoded .lif file contents
        
    Returns:
        dict: Parsed image data organized by:
            - Image name
            - Timepoint
            - Z-stack level
            - Channel
    """
    content_string: str
    _, content_string = image_data.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    with tempfile.NamedTemporaryFile() as imagefile:
        imagefile.write(decoded_content)
        imagefile.flush()
        lif_obj = LifFile(imagefile.name)
        images: dict = {}
        for i, img in enumerate(lif_obj.get_iter_image()):
            dims: dict = img.dims_n
            name: str = img.name
            z_stack: int = 1
            time: int = 1
            if 3 in dims:
                z_stack = dims[3]
            if 4 in dims:
                time = dims[4]
            image_assembly: dict = {}
            for t in range(time):
                stack = {
                    z: np.stack([np.array(img.get_frame(z,t,ch,0)) for ch in range(img.channels)])
                    for z in range(z_stack)
                }
                image_assembly[t] = stack
            images[name] = image_assembly
    return images

@callback(
    Output('names-radio', 'options'),
    Output('names-radio', 'value'),
    Input('image-metadata-only-store','data'),
    prevent_initial_call=True
)
def load_names_options(image_metadata: dict):
    """Generate radio options for available image series names.
    
    Args:
        image_metadata: Dictionary containing image metadata
        
    Returns:
        tuple: (radio options list, default selected value)
    """
    vals: list = sorted(list(image_metadata.keys()))
    return (
        [
            { 
                'label': o, 'value': o
            } for o in vals
        ],
        vals[0]
    )

@callback(
    Output('timepoint-radio', 'options'),
    Output('timepoint-radio', 'value'),
    State('image-metadata-only-store','data'),
    Input('names-radio','value'),
    prevent_initial_call=True
)
def load_timepoint_options(image_metadata: dict, img_name: str):
    """Generate radio options for available timepoints in selected image.
    
    Args:
        image_metadata: Dictionary containing image metadata
        img_name: Selected image series name
        
    Returns:
        tuple: (radio options list, default selected value)
    """
    if image_metadata is None: # TODO: check if this still happens;; There is no reason why this gets triggered with a none value in metadata, but it does. Even with prevent_initial_call.
        return no_update
    vals: list = sorted(list(image_metadata[img_name].keys()))
    return (
        [
            {
                'label': o, 'value': o
            } for o in vals
        ],
        vals[0]
    )

@callback(
    Output('zlevel-radio', 'options'),
    Output('zlevel-radio', 'value'),
    State('image-metadata-only-store','data'),
    State('names-radio','value'),
    Input('timepoint-radio','value'),
    prevent_initial_call = True
)
def load_zleveloptions(image_metadata: dict, img_name: str, timepoint: int):
    """Generate radio options for available Z-stack levels.
    
    Args:
        image_metadata: Dictionary containing image metadata
        img_name: Selected image series name
        timepoint: Selected timepoint
        
    Returns:
        tuple: (radio options list, default middle Z-level as selected value)
    """
    vals: list = image_metadata[img_name][timepoint]
    default = vals[int(len(vals)/2)]
    return [
        {
            'label': o, 'value': o
        } for o in vals
    ], default

@callback(
    Output({'type': 'channel-image', 'name': MATCH}, 'figure',allow_duplicate=True),
    Input({'type': 'channel-cmap-chooser', 'name': MATCH}, 'value'),
    Input({'type': 'channel-cmap-reverser', 'name': MATCH}, 'value'),
    State({'type': 'channel-image', 'name': MATCH}, 'figure'),
    prevent_initial_call=True
)
def update_cmap(channel_cmap: str, reverse: list, fig: go.Figure):
    """Update colormap settings for a channel visualization.
    
    Args:
        channel_cmap: Name of colormap to use
        reverse: Whether to reverse the colormap
        fig: Current figure object
        
    Returns:
        go.Figure: Updated figure with new colormap settings
    """
    fig=go.Figure(fig)
    fig.update_coloraxes(colorscale=channel_cmap,reversescale = reverse)
    return fig


@callback(
    Output('selected-image-data-store','data'),
    State('image-data-store','data'),
    Input('names-radio','value'),
    Input('timepoint-radio','value'),
    Input('zlevel-radio','value'),
    prevent_initial_call=True
)
def load_image_slice(image_data, name, t, z):
    """Load specific image slice based on selected parameters.
    
    Args:
        image_data: Full image dataset
        name: Selected image series name
        t: Selected timepoint
        z: Selected Z-level
        
    Returns:
        numpy.ndarray: Image data for selected slice
    """
    if (name is None) or (t is None) or (z is None) or (image_data is None): # TODO: check if this still happens;; This also gets called when app loads, even with prevent
        return no_update
    else:
        return image_data[name][t][z]

@callback(
    Output('channels-row', 'children'),
    Output('colocalization-row', 'children'),
    Input('selected-image-data-store','data')
)
def generate_figures(image_data:list):
    """Generate channel and merged view figures.
    
    Args:
        image_data: List of channel image data arrays
        
    Returns:
        tuple: (list of channel figures, merged view area components)
    """
    if image_data is None:
        return no_update
    channel_figs = []
    cmap = 'gray'

    for i, ch_np_stack in enumerate(image_data):
        fig = px.imshow(
            ch_np_stack,
            aspect='equal',
            title=f'Channel {i}',
            color_continuous_scale=cmap,
            height=300,
            width=300
        )
        fig.update_layout(showlegend=False, margin={'pad': 0, 't': 0, 'b': 0, 'l': 0, 'r': 0})
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        channel_figs.append(
            dbc.Col([
                dcc.Dropdown(
                    id={'type': 'channel-cmap-chooser', 'name': f'Channel-{i}'},
                    options = available_cmaps,
                    value=cmap),
                dbc.Switch(
                    label='Reverse cmap',
                    value=False,
                    id={'type': 'channel-cmap-reverser', 'name': f'Channel-{i}'}
                ),
                dcc.Graph(
                    id={'type': 'channel-image', 'name': f'Channel-{i}'},
                    figure = fig
                )]
            )
        )
    merge_area = [
        dbc.Col([
            dcc.Dropdown(
                id={'type': 'overview-cmap-tool', 'name': 'Chooser'},
                options = available_cmaps,
                value='electric'),
            dbc.Switch(
                label='Reverse cmap',
                value=False,
                id={'type': 'overview-cmap-tool', 'name': 'Reverse'}
            ),
            dbc.Label('Select two channels to inspect:'),
            dbc.Checklist(
                id='channel-selection',
                options=[
                    {'label': o, 'value': o}
                    for o in range(len(image_data))
                ],
                value=[0,1],
                switch=True
            ),
            dbc.RadioItems(
                id='method-radio',
                options = [
                    {'label': o, 'value': o}
                    for o in ['Multiply','Add']
                ],
                value='Multiply',
                labelCheckedClassName="text-success",
                inputCheckedClassName="border border-success bg-success",
            )
        ], width=2),
        dbc.Col(
            dcc.Graph(
                id={'type': 'channel-image', 'name': f'Overview'},
                figure = fig
            ),
            width=10
        )
    ]
    # Draw figures
    return channel_figs, merge_area

@callback(
    Output('image-metadata-only-store','data'),
    Input('image-data-store','data'),
)
def parse_metadata(image_data):
    """Extract metadata from image data structure.
    
    Args:
        image_data: Full image dataset
        
    Returns:
        dict: Metadata organized by image name, timepoint and Z-levels
    """
    if image_data is None:
        return no_update
    metadata = {}
    for imagename, img_data in image_data.items():
        metadata[imagename] = {}
        for timepoint, time_stack in img_data.items():
            metadata[imagename][timepoint] = sorted(list(time_stack.keys()))
    return metadata

@callback(
    Output('channel-selection', 'options'),
    Input('channel-selection','value'),
    State('channel-selection','options')
)
def update_channel_selection(value, options):
    """Update channel selection options based on current selection.
    
    Args:
        value: Currently selected channels
        options: All available channel options
        
    Returns:
        list: Updated options with appropriate disabled states
    """
    if len(value) >= 2:
        options = [
            {
                'label': option['label'],
                'value': option['value'],
                'disabled': option['value'] not in value
            }
            for option in options
        ]
    elif len(value) < 2:
        for d in options:
            d['disabled'] = False
    return options

@callback(
    Output({'type': 'channel-image', 'name': ALL}, 'relayoutData'),
    Output({'type': 'channel-image', 'name': ALL}, 'figure',allow_duplicate=True),
    Input({'type': 'channel-image', 'name': ALL}, 'relayoutData'),
    State({'type': 'channel-image', 'name': ALL}, 'figure'),
    prevent_initial_call=True
)
def LinkedZoom(relayout_data, figure_states):
    """Synchronize zoom/pan across all channel views.
    
    Args:
        relayout_data: Layout update data from user interaction
        figure_states: Current state of all figures
        
    Returns:
        tuple: (updated layout data, updated figure states)
    """
    unique_data = None
    for data in relayout_data:
        if relayout_data.count(data) == 1:
            unique_data = data
    if unique_data:
        for figure_state in figure_states:
            if unique_data.get('xaxis.autorange'):
                figure_state['layout']['xaxis']['autorange'] = True
                figure_state['layout']['yaxis']['autorange'] = True
            else:
                if 'xaxis.range[0]' in unique_data:
                    figure_state['layout']['xaxis']['range'] = [
                        unique_data['xaxis.range[0]'], unique_data['xaxis.range[1]']
                    ]
                figure_state['layout']['xaxis']['autorange'] = False
                if 'yaxis.range[0]' in unique_data:
                    figure_state['layout']['yaxis']['range'] = [
                        unique_data['yaxis.range[0]'], unique_data['yaxis.range[1]']
                    ]
                figure_state['layout']['yaxis']['autorange'] = False
        return [unique_data] * len(relayout_data), figure_states
    return relayout_data, figure_states
 
@callback(
    Output({'type': 'channel-image', 'name': 'Overview'}, 'figure'),
    Input('selected-image-data-store','data'),
    Input('channel-selection','value'),
    Input('method-radio','value'),
    Input({'type': 'overview-cmap-tool', 'name': 'Chooser'}, 'value'),
    Input({'type': 'overview-cmap-tool', 'name': 'Reverse'}, 'value'),
    State({'type': 'channel-image', 'name': 'Overview'}, 'figure'),
)
def load_merge(image_data: list, channel_selection: list, method:str, channel_cmap: str, reverse: list, fig: go.Figure):
    """Generate merged view of selected channels.
    
    Args:
        image_data: List of channel image data arrays
        channel_selection: List of selected channel indices
        method: Merge method ('Multiply' or 'Add')
        channel_cmap: Colormap for merged view
        reverse: Whether to reverse the colormap
        fig: Current figure object
        
    Returns:
        go.Figure: Updated figure with merged channels
    """
    trigger = ctx.triggered_id
    if isinstance(trigger, dict):
        if trigger['type'] == 'overview-cmap-tool':
            fig=go.Figure(fig)
            fig.update_coloraxes(colorscale=channel_cmap,reversescale = reverse)
            return fig
        else:
            return no_update
    if image_data is None:
        return no_update
    if len(channel_selection) != 2:
        return no_update
    if reverse:
        channel_cmap = f'{channel_cmap}_r'
    np_matrices = [image_data[i] for i in channel_selection]
    if 'Multiply' in method:
        merged = np.multiply(*np_matrices)
        zmax = np.array(np_matrices[0]).max()*np.array(np_matrices[1]).max()
    elif method == 'Add':
        merged = np.add(*np_matrices)
        zmax = np.array(np_matrices[0]).max()+np.array(np_matrices[1]).max()
    else:
        merged = np_matrices[0]
        zmax = np.array(np_matrices[0]).max()
    fig = px.imshow(
        merged,
        aspect='equal',
        title=f'Merged: {channel_selection}',
        color_continuous_scale=channel_cmap,
        height=900,
        width=900,
        zmax=zmax
    )
    fig.update_layout(showlegend=False, margin={'pad': 0, 't': 0, 'b': 0, 'l': 5, 'r': 20})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def microscopy_content_div():
    """Create the main content area for microscopy images.
    
    Returns:
        dbc.Col: Column containing rows for channel images and colocalization analysis
    """
    return dbc.Col(
        [
            dbc.Row(id='channels-row'),
            dbc.Row(id='useless-spacer',children = ' '),
            dbc.Row(id='colocalization-row')
        ],width=9
    )

def utils():
    """Create hidden utility components for storing application state.
    
    Returns:
        html.Div: Container with Store components for:
            - image-data-store: Raw image data
            - image-metadata-only-store: Image metadata
            - selected-image-data-store: Currently selected image data
    """
    return html.Div([
        dcc.Store(id = 'image-data-store'),#,data=load_image(os.path.join('..','..','..','202410 PG microscopy/SP8/EFN1-GLI3-STAV.lif'))),
        dcc.Store(id = 'image-metadata-only-store'),
        dcc.Store(id = 'selected-image-data-store')
    ],hidden=True)

def make_du_uploader(id_str: str, message: str):
    """Create a dash-uploader component with success indicator.
    
    Args:
        id_str: ID for the uploader component
        message: Display text for the upload area
        
    Returns:
        tuple: (html.Div containing uploader and indicator, session_id)
    """
    session_id = str(uuid.uuid1())
    asty = {k: v for k, v in UPLOAD_INDICATOR_STYLE.items()}
    asty['height'] = '100px'
    return html.Div(
        children = [
            html.Div(
                du.Upload(
                        id=id_str,
                        text=message,
                        max_file_size=20000,  # 50 Mb
                        chunk_size=4,  # 4 MB
                        upload_id=session_id,  # Unique session id,
                        default_style = UPLOAD_STYLE
                ),
                style={'display': 'inline-block', 'width': '75%',
                    'float': 'left', 'height': '25px'},
                ),
            html.Div(
                id = f'{id_str}-success',
                style=asty,
            )
        ],  
    ), session_id

def sidebar():
    """Create the sidebar containing file upload and image selection controls.
    
    Returns:
        dbc.Col: Column containing:
            - File upload area
            - Image metadata display
            - Series selection radio buttons
            - Timepoint selection radio buttons
            - Z-level selection radio buttons
            - Download button
    """
    return dbc.Col(
        [
            html.Div(
                upload_area('microscopy-uploader','LIF file',True),
                style={
                        'width': '100%',
                        'display': 'inline-block',
                        'float': 'left',
                        'height': '25px',
                        'padding-top': '10px'
                        #'padding': '10px 10px 10px 10px'
                    }
            ),
            html.Br(),
            html.Div(id='microscopy-image-metainfo', children='', style={'padding-top': '45px'}),
            dbc.Label('Select series:'),
            dbc.RadioItems(
                id='names-radio',
                labelCheckedClassName="text-success",
                inputCheckedClassName="border border-success bg-success",
            ),
            dbc.Label('Select timepoint:'),
            dbc.RadioItems(
                id='timepoint-radio',
                labelCheckedClassName="text-success",
                inputCheckedClassName="border border-success bg-success",
            ),
            dbc.Label('Select Z-level:'),
            dbc.RadioItems(
                id='zlevel-radio',
                labelCheckedClassName="text-success",
                inputCheckedClassName="border border-success bg-success",
            )
        ],
        width=3,
        style={'padding': '0px 0px 0px 15px'}
    )


layout = html.Div(
    [
        dbc.Row([
            sidebar(),
            microscopy_content_div(),
        ]),
        dbc.Row(utils())
        
    ],style=GENERIC_PAGE
)