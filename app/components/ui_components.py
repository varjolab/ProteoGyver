"""Components for the user interface.

This module provides reusable UI components for building the application interface,
including checklists, range inputs, and other common elements.

Attributes:
    HEADER_DICT (dict): Mapping of header levels to HTML header components
"""

from typing import Any, List, Dict, Tuple, Optional, Union
import dash_bootstrap_components as dbc
from dash import dcc, html
from element_styles import SIDEBAR_STYLE, UPLOAD_A_STYLE, UPLOAD_STYLE, UPLOAD_BUTTON_STYLE, CONTENT_STYLE, SIDEBAR_LIST_STYLES, UPLOAD_INDICATOR_STYLE
from components import tooltips, text_handling
from numpy import log2
import uuid
import dash_uploader as du
from components.figures.figure_legends import INTERACTOMICS_LEGENDS as interactomics_legends
from components.figures.figure_legends import saint_legend

HEADER_DICT: Dict[str, Dict[int, Any]] = {
    'component': {
        1: html.H1,
        2: html.H2,
        3: html.H3,
        4: html.H4,
        5: html.H5,
        6: html.H6
    },
}


def checklist(
        label: str,
        options: List[str],
        default_choice: List[str],
        disabled_options: Optional[List[str]] = None,
        id_prefix: Optional[str] = None,
        id_only: bool = False,
        prefix_list: Optional[List[Any]] = None,
        postfix_list: Optional[List[Any]] = None,
        clean_id: bool = True,
        style_override: Optional[Dict[str, Any]] = None
) -> List[Any]:
    """Creates a Bootstrap checklist component with customizable options.

    Args:
        label (str): Label text for the checklist
        options (list): List of options to display in the checklist
        default_choice (list): List of pre-selected options
        disabled_options (list, optional): List of options to disable. Defaults to None.
        id_prefix (str, optional): Prefix for the component ID. Defaults to None.
        id_only (bool, optional): If True, removes label from display. Defaults to False.
        prefix_list (list, optional): Elements to prepend to checklist. Defaults to None.
        postfix_list (list, optional): Elements to append to checklist. Defaults to None.
        clean_id (bool, optional): If True, sanitizes the ID string. Defaults to True.
        style_override (dict, optional): Custom CSS styles. Defaults to None.

    Returns:
        dbc.Checklist: Bootstrap checklist component with specified options and styling
    """
    if disabled_options is None:
        disabled: set = set()
    else:
        disabled: set = set(disabled_options)
    if clean_id:
        checklist_id: str
        checklist_id = text_handling.replace_special_characters(
            f'{id_prefix}-{label}',
            '-', stripresult=True, remove_duplicates=True)
    else:
        checklist_id = label
    if id_only:
        label = ''
    if prefix_list is None:
        prefix_list = []
    if postfix_list is None:
        postfix_list = []
    retlist: list = [
        label,
        dbc.Checklist(
            options=[
                {
                    'label': o, 'value': o, 'disabled': o in disabled
                } for o in options
            ],
            value=default_choice,
            id=checklist_id,
            switch=True,
            style=style_override
        )
    ]
    return prefix_list + retlist + postfix_list


def range_input(
    label: str, 
    min_val: float, 
    max_val: float, 
    id_str: str, 
    typestr: str = 'number', 
    style_float: str = 'center', 
    stepsize: float = 1
) -> html.Div:
    """Creates a range input component with min and max value inputs.

    Args:
        label (str): Label text for the range input
        min (float): Initial minimum value
        max (float): Initial maximum value
        id_str (str): Base ID for the component
        typestr (str, optional): Input type ('number', 'text', etc). Defaults to 'number'.
        style_float (str, optional): Float style for positioning. Defaults to 'center'.
        stepsize (float, optional): Step size for number inputs. Defaults to 1.

    Returns:
        html.Div: Div containing the range input component with label and min/max inputs
    """
    pad = {
        'padding': '0px 5px 0px 5px', 
        'margin': 'auto',
        'margin-left': 'auto', 
        'margin-right': 'auto'
    }
    return html.Div(
        children = [
            html.P(label, style=pad|{'width': '50%'}),
            dcc.Input(id=f'{id_str}-min', type=typestr,value=min_val,style=pad|{'width': '20%'},step=stepsize),
            html.P('-', style=pad|{'width': '10%'}),
            dcc.Input(id=f'{id_str}-max', type=typestr,value=max_val,style=pad|{'width': '20%'},step=stepsize),
        ],
        style={'display': 'inline-flex', 'width': '100%', 'height': '20px',
                'float': style_float, 'height': '20px'},
        id = id_str
    )

def make_du_uploader(id_str: str, message: str) -> Tuple[html.Div, str]:
    """Creates a dash-uploader component with success indicator.

    Args:
        id_str (str): ID for the upload component
        message (str): Display message for the upload area

    Returns:
        tuple[html.Div, str]: A tuple containing:
            - html.Div: The upload component with success indicator
            - str: A unique session ID for the upload

    Notes:
        Creates an upload area with a 50MB file size limit and 4MB chunks.
        Includes a success indicator div that appears after successful upload.
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

def upload_area(id_text: str, upload_id: str, indicator: bool = True) -> html.Div:
    """Creates a drag-and-drop upload area with optional success indicator.

    Args:
        id_text (str): ID for the upload component
        upload_id (str): Display text for the upload area
        indicator (bool, optional): Whether to show upload success indicator. Defaults to True.

    Returns:
        html.Div: A div containing the upload area and optional success indicator

    Notes:
        Creates a drag-and-drop area with a clickable 'select' link.
        Includes a loading spinner and optional success indicator.
        Multiple file upload is disabled.
    """
    ret: list = [
        html.Div(
            children=[
                dcc.Upload(
                    id=id_text,
                    children=html.Div([
                        'Drag and drop or ',
                        html.A('select', style=UPLOAD_A_STYLE),
                        f' {upload_id}',
                        dcc.Loading(html.P(id=f'{id_text}-spinner'))
                    ]
                    ),
                    style=UPLOAD_STYLE,
                    multiple=False
                )
            ],
            style={'display': 'inline-block', 'width': '75%',
                   'float': 'left', 'height': '65px'},
        ),
    ]
    if indicator:
        ret.append(
            html.Div(
                id=f'{id_text}-success',
                style=UPLOAD_INDICATOR_STYLE
            )
        )
    return html.Div(ret)

def main_sidebar(figure_templates: List[str], implemented_workflows: List[str]) -> html.Div:
    """Creates the main sidebar component with input controls and workflow options.

    Args:
        figure_templates (list): Available figure style templates
        implemented_workflows (list): List of available workflow types

    Returns:
        html.Div: Sidebar component containing:
            - Example files download button
            - File upload areas
            - Analysis options checklist
            - Workflow selection dropdown
            - Figure style selection
            - Analysis control buttons
            - Table of contents
    """
    return html.Div(
        children = [
            html.H2(children='▼ Input', id='input-header', style={'textAlign': 'left'}),
            dbc.Collapse([
                dbc.Button(
                    'Download example files',
                    style=UPLOAD_BUTTON_STYLE,
                    id='button-download-example-files',
                    className='btn-info',
                ),
                html.Label('Upload files:'),
                upload_area('upload-data-file', 'Data file'),
                upload_area('upload-sample_table-file', 'Sample table'),
                html.Div(
                    [
                        html.Label('Options:'),
                        dcc.Checklist(
                            id='sidebar-options',
                            options=['Remove common contaminants', 'Rename replicates', 'Use unique proteins only (remove protein groups)'], value=['Remove common contaminants'],
                        )
                    ],
                    style={'display': 'inline-block'}
                ),
                html.Label('Select workflow:'),
                dbc.Select(
                    options=[
                        {'label': item, 'value': item} for item in implemented_workflows
                    ],
                    id='workflow-dropdown',
                ),
                html.Label('Select figure style:'),
                dbc.Select(
                    value=figure_templates[0],
                    options=[
                        {'label': item, 'value': item} for item in figure_templates
                    ],
                    id='figure-theme-dropdown',
                ),
                dbc.Button(
                    'Begin analysis',
                    id='begin-analysis-button',
                    style=UPLOAD_BUTTON_STYLE,
                    className='btn-info',
                    disabled=True,
                ),
            ],id='input-collapse',is_open=True),
            html.Div(
                id='discard-samples-div',
                children=[
                    dbc.Button(
                        'Choose samples to discard',
                        id='discard-samples-button',
                        style=UPLOAD_BUTTON_STYLE,
                        className='btn-warning',
                        n_clicks=0
                    ),
                ],
                hidden=True
            ),
            dbc.Button(
                children = dcc.Loading(
                            html.Div(id='button-download-all-data-text', children='Download all data')
                ),
                style=UPLOAD_BUTTON_STYLE,
                id='button-download-all-data',
                className='btn-info',
                disabled=True,
            ),
            # top right bottom left
            html.Div(id='toc-div', style={'padding': '0px 10px 10px 30px', 'overflow': 'scroll'}),
            dcc.Download(id='download-example-files'),
            dcc.Download(id='download-proteomics-comparison-example'),
            dcc.Download(id='download-all-data')
        ],
        className='card text-white bg-primary mb-3',
        id={'type': 'input-div','id': 'sidebar-input'},
        style=SIDEBAR_STYLE
    )


def modals() -> html.Div:
    """Creates modal dialogs for the application.

    Returns:
        html.Div: Container with modal components including:
            - Sample discard modal with scrollable content
            - Header with modal title
            - Body with discard button and checklist container

    Notes:
        Modals are initially hidden (is_open=False) and can be triggered
        by other components.
    """
    return html.Div([
        dbc.Modal(
            id='discard-samples-modal',
            is_open=False,
            scrollable=True,
            size='xl',
            children=[
                    dbc.ModalHeader(dbc.ModalTitle(
                        'Select samples to discard')),
                    dbc.ModalBody(
                        children=[
                            dbc.Button('Discard samples',
                                       id='done-discarding-button', n_clicks=0),
                            html.Div(
                                id='discard-sample-checklist-container'
                            ),
                        ]
                    ),
                    ]
        )
    ])


def main_content_div() -> html.Div:
    """Creates the main content area for displaying analysis results.

    Returns:
        html.Div: Main content container with:
            - Workflow-specific input area
            - QC analysis area
            - Workflow-specific results area

    Notes:
        Content areas are initially empty and populated based on
        user interactions and analysis results.
    """
    return html.Div(
        id='main-content-div',
        children=[
            html.Div(id='workflow-specific-input-div'),
            html.Div(
                id={'type': 'analysis-div', 'id': 'qc-analysis-area'},
                children=[
                ]
            ),
            html.Div(id='workflow-specific-div')
        ]
    )


def workflow_area(
    workflow: str, 
    workflow_specific_parameters: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> html.Div:
    """Creates the appropriate workflow area based on the selected workflow type.

    Args:
        workflow (str): Selected workflow type ('Proteomics', 'Interactomics', or 'Phosphoproteomics')
        workflow_specific_parameters (dict): Parameters specific to each workflow type
        data_dictionary (dict): Data required for the workflow analysis

    Returns:
        html.Div: Workflow-specific component containing:
            - Input controls specific to the workflow
            - Analysis results area
            - Visualization components

    Notes:
        The returned component structure varies based on the selected workflow type.
        Each workflow type has its own layout and functionality.
    """
    ret: list
    if workflow == 'Proteomics':
        ret = proteomics_area(
            workflow_specific_parameters['proteomics'], data_dictionary)
    elif workflow == 'Interactomics':
        ret = interactomics_area(
            workflow_specific_parameters['interactomics'], data_dictionary)
    elif workflow == 'Phosphoproteomics':
        ret = phosphoproteomics_area(
            workflow_specific_parameters['phosphoproteomics'], data_dictionary)
    return ret # type: ignore


def proteomics_input_card(
    parameters: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> dbc.Card:
    """Creates a card component containing proteomics analysis input controls.

    Args:
        parameters (dict): Configuration parameters including:
            - na_filter_default_value: Default NA filtering threshold
            - imputation methods: Available imputation method options
            - default imputation method: Pre-selected imputation method
            - normalization methods: Available normalization method options
            - default normalization method: Pre-selected normalization method
        data_dictionary (dict): Data containing sample groups and normalization info

    Returns:
        dbc.Card: Card component containing:
            - NA filtering controls with tooltip
            - Imputation method selection
            - Normalization method selection
            - Control group selection or comparison file upload
            - Fold change and p-value threshold controls
            - Analysis execution button
    """
    control_dropdown_options = ['']
    control_dropdown_options.extend(sorted(list(data_dictionary['sample groups']['norm'].keys())))
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        html.Div([
                            dbc.Label('Filter out proteins not present in at least:', id='filtering-label'),
                            tooltips.na_tooltip()
                        ]),
                        dcc.Slider(0, 100, 
                                step = 10, 
                                marks={
                                    i: f'{i}%' for i in range(0, 101, 10)
                                    },
                                value=parameters['na_filter_default_value'],
                                id='proteomics-filter-minimum-percentage'),
                        dbc.Label('of:', id='filtering-label'),
                        dbc.RadioItems(
                            options=[
                                {"label": "One sample group", "value": 'sample-group'},
                                {"label": "Whole sample set", "value": 'sample-set'}
                            ],
                            value='sample-group',
                            id="proteomics-filter-type",
                        ),
                    ], style={'padding': '5px 5px 5px 5px'}), 
                width=5),
                dbc.Col([
                    dbc.Label('Imputation:'),
                    dbc.RadioItems(
                        options=[
                            {'label': i_opt, 'value': i_opt_val}
                            for i_opt, i_opt_val in parameters['imputation methods'].items()
                        ],
                        value=parameters['default imputation method'],
                        id='proteomics-imputation-radio-option'
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label('Normalization:'),
                    dbc.RadioItems(
                        options=[
                            {'label': n_opt, 'value': n_opt_val}
                            for n_opt, n_opt_val in parameters['normalization methods'].items()
                        ],
                        value=parameters['default normalization method'],
                        id='proteomics-normalization-radio-option'
                    ),
                ], width=2),
            ]),
            dbc.Row([
                dbc.Label('Select control group:'),]),
            dbc.Row([
                dbc.Col([
                        dbc.Select(
                            options=[
                                {'label': sample_group, 'value': sample_group} for sample_group in
                                control_dropdown_options
                            ],
                            required=True,
                            id='proteomics-control-dropdown',
                        ),
                        ], width=4),
                dbc.Col([
                    html.Div(
                        dbc.Label('Or'),
                        # 'padding': '50% 0px 0px 0px'} # top right bottom left
                        style={'text-align': 'center', }
                    )
                ], width=1),
                dbc.Col(
                    upload_area('proteomics-comparison-table-upload',
                                'Comparison file', indicator=True),
                    width=4,
                ),
                dbc.Col(
                    dbc.Button('Download example comparison file',
                               id='download-proteomics-comparison-example-button'),
                    width=2
                ),
                dbc.Col(
                    '',
                    width=1
                )
            ], style={"display": "flex", "align-items": "bottom"},),
            dbc.Row([
                dbc.Col([
                    dbc.Label('log2 fold change threshold for comparisons:'),
                    dcc.RadioItems([
                        {'label': f'{log2(x):.2f} ({x}-fold change)',
                         'value': log2(x)}
                        for x in (1.2, 1.5, 2, 3, 4, 5)
                    ], 1, id='proteomics-fc-value-threshold'),
                ], width=6),
                dbc.Col([
                    dbc.Label('Adjusted p-value threshold for comparisons:'),
                    dcc.RadioItems([0.001, 0.01, 0.05], 0.01,
                                   id='proteomics-p-value-threshold'),
                    dbc.Label('Test type for comparisons:'),
                    dcc.RadioItems(['independent','paired'], 'independent',
                                   id='proteomics-test-type'),
                    tooltips.test_type_tooltip(),
                ], width=6)
                
            ]),
            dbc.Row(
                [
                    dbc.Button('Run proteomics analysis',
                               id='proteomics-run-button'),
                ]
            )
        ])
    )

def proteomics_area(
    parameters: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> List[html.Div]:
    """Creates the main proteomics analysis area with input controls and results display.

    Args:
        parameters (dict): Proteomics-specific configuration parameters
        data_dictionary (dict): Data required for proteomics analysis

    Returns:
        list[html.Div]: List containing:
            - Input div with proteomics-specific controls
            - Results div with loading indicators for:
                - NA filtering plots
                - Normalization plots
                - Missing value plots
                - Imputation plots
                - CV plots
                - PCA plots
                - Clustermap plots
                - Volcano plots

    Notes:
        Each plot area includes a loading indicator that displays while
        the corresponding analysis is being performed.
    """
    return [
        html.Div(
            id={'type': 'input-div', 'id': 'proteomics-analysis-area'},
            children=[
                html.H1('Proteomics-specific input options'),
                proteomics_input_card(parameters, data_dictionary),
                html.Hr()
            ]
        ),
        html.Div(
            id={'type': 'analysis-div', 'id': 'proteomics-analysis-results-area'},
            children=[
                html.H1(id='proteomics-result-header', children='Proteomics'),
                dcc.Loading(
                    id='proteomics-loading-filtering',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-na-filtered-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-normalization',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-normalization-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-missing-in-other',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-missing-in-other-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-imputation',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-imputation-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-cv',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-cv-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-pca',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-pca-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-clustermap',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-clustermap-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-pertubation-volcano',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-pertubation-plot-div'}),
                    type='default'
                ),
                dcc.Loading(
                    id='proteomics-loading-volcano',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-volcano-plot-div'}),
                    type='default'
                ),
            ]
        )
    ]


def discard_samples_checklist(
    count_plot: html.Div, 
    list_of_samples: List[str]
) -> List[Any]:
    """Creates a checklist for selecting samples to discard from analysis.

    Args:
        count_plot (html.Div): Plot component showing sample counts
        list_of_samples (list): List of sample names that can be discarded

    Returns:
        list: List containing:
            - Count plot visualization
            - Checklist for sample selection with:
                - All samples as options
                - None pre-selected
                - Checklist ID prefixed with 'checklist'

    Notes:
        The returned components are typically displayed in a modal dialog
        for sample filtering.
    """
    return [
        count_plot,
        html.Div(
            checklist(
                label='Select samples to discard',
                id_only=True,
                options=list_of_samples,
                default_choice=[],
                id_prefix='checklist'
            )
        )
    ]


def interactomics_control_col(
    all_sample_groups: List[str], 
    chosen: List[str]
) -> dbc.Col:
    """Creates a column component for selecting uploaded control samples.

    Args:
        all_sample_groups (list): List of all available sample groups
        chosen (list): List of pre-selected sample groups

    Returns:
        dbc.Col: Column component containing:
            - "Select all" checkbox for uploaded controls
            - Checklist of available sample groups
            - Label for the control selection area
    """
    return dbc.Col([
        html.Div(
            checklist(
                'select all uploaded',
                ['Select all uploaded'],
                [],
                id_only=True,
                id_prefix='interactomics',
            )
        ),
        html.Div(
            checklist(
                'Choose uploaded controls:',
                all_sample_groups,
                chosen,
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[dbc.Label('Choose uploaded controls:')]
            )
        ),
        html.Br(),
        html.Div(
        )
    ])


def interactomics_inbuilt_control_col(controls_dict: Dict[str, List[str]]) -> dbc.Col:
    """Creates a column component for selecting built-in control sets.

    Args:
        controls_dict (dict): Dictionary containing:
            - available (list): Available control set options
            - default (list): Pre-selected control sets
            - disabled (list): Control sets that should be disabled

    Returns:
        dbc.Col: Column component containing:
            - "Select all" checkbox for built-in controls
            - Checklist of available control sets
            - Label for the control sets selection area
    """
    return dbc.Col([
        html.Div(
            checklist(
                'select all inbuilt controls',
                ['Select all inbuilt controls'],
                [],
                id_only=True,
                id_prefix='interactomics',
            )
        ),
        html.Div(
            checklist(
                'Choose additional control sets:',
                controls_dict['available'],
                controls_dict['default'],
                disabled_options=controls_dict['disabled'],
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[dbc.Label('Choose additional control sets:')]
            )
        ),
    ])

def interactomics_crapome_col(crapome_dict: Dict[str, List[str]]) -> dbc.Col:
    """Creates a column component for selecting CRAPome control sets.

    Args:
        crapome_dict (dict): Dictionary containing:
            - available (list): Available CRAPome set options
            - default (list): Pre-selected CRAPome sets
            - disabled (list): CRAPome sets that should be disabled

    Returns:
        dbc.Col: Column component containing:
            - "Select all" checkbox for CRAPome sets
            - Checklist of available CRAPome sets
            - Label for the CRAPome selection area

    Notes:
        CRAPome (Contaminant Repository for Affinity Purification) sets are 
        used to filter out common contaminants in interaction studies.
    """
    return dbc.Col([
        html.Div(
            checklist(
                'select all crapomes',
                ['Select all crapomes'],
                [],
                id_only=True,
                id_prefix='interactomics',
            )
        ),
        html.Div(
            checklist(
                'Choose Crapome sets:',
                crapome_dict['available'],
                crapome_dict['default'],
                disabled_options=crapome_dict['disabled'],
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[dbc.Label('Choose Crapome sets:')]
            )
        )
    ])


def interactomics_enrichment_col(enrichment_dict: Dict[str, List[str]]) -> dbc.Col:
    """Creates a column component for selecting enrichment analysis options.

    Args:
        enrichment_dict (dict): Dictionary containing:
            - available (list): Available enrichment analysis options
            - default (list): Pre-selected enrichment options
            - disabled (list): Enrichment options that should be disabled

    Returns:
        dbc.Col: Column component containing:
            - Checklist of available enrichment options
            - "Deselect all" button
            - Label for the enrichment selection area

    Notes:
        Unlike other selection columns, this one includes a deselect button
        instead of a "Select all" checkbox.
    """
    return dbc.Col([
        html.Div(
            checklist(
                'Choose enrichments:',
                enrichment_dict['available'],
                enrichment_dict['default'],
                disabled_options=enrichment_dict['disabled'],
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[
                    dbc.Button('Deselect all enrichments',id='interactomics-select-none-enrichments'),
                    html.Br(),
                    dbc.Label('Choose enrichments:')
                ]
            )
        )
    ])


def interactomics_input_card(
    parameters: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> html.Div:
    """Creates the main input card for interactomics analysis configuration.

    Args:
        parameters (dict): Configuration parameters containing:
            - controls (dict): Built-in control set options
            - crapome (dict): CRAPome filtering options
            - enrichment (dict): Enrichment analysis options
        data_dictionary (dict): Data containing:
            - sample groups (dict): Normalized sample groups
            - guessed control samples (list): Auto-detected control samples

    Returns:
        html.Div: Input card containing:
            - Control selection columns (uploaded, built-in, CRAPome)
            - Rescue filtering options
            - Nearest control filtering settings
            - Enrichment selection
            - SAINT analysis button

    Notes:
        Automatically sorts and combines guessed controls with other sample groups.
        Includes tooltips for rescue and nearest control filtering options.
    """
    all_sample_groups: List[str] = []
    sample_groups: Dict[str, Any] = data_dictionary['sample groups']['norm']
    guessed_controls: List[str] = data_dictionary['sample groups']['guessed control samples'][0]
    for k in sample_groups.keys():
        if k not in guessed_controls:
            all_sample_groups.append(k)
    all_sample_groups = sorted(guessed_controls) + sorted(all_sample_groups)
    return html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Row(
                            [
                                interactomics_control_col(
                                    all_sample_groups, guessed_controls),
                                interactomics_inbuilt_control_col(
                                    parameters['controls']),
                                interactomics_crapome_col(
                                    parameters['crapome']),
                            ]
                        ),
                        dbc.Row([
                            dbc.Col(
                                children = checklist(
                                        'Rescue filtered out',
                                        ['Rescue interactions that pass filter in any sample group'],
                                        [],
                                        id_only=True,
                                        id_prefix='interactomics',
                                        style_override={
                                            'margin': '5px', 'verticalAlign': 'center'
                                        },
                                        prefix_list=[
                                            tooltips.rescue_tooltip()
                                        ]
                                    ),
                                width=4
                            ),
                            dbc.Col([
                                html.Div(
                                    children = [
                                        dbc.Row([
                                            dbc.Col(
                                                checklist(
                                                    'Nearest control filtering',
                                                    ['Select'],
                                                    [],
                                                    id_only=True,
                                                    id_prefix='interactomics',
                                                    style_override={
                                                        'margin': '5px', 'verticalAlign': 'center'
                                                    },
                                                    prefix_list=[
                                                        tooltips.nearest_tooltip()
                                                    ]
                                                ), width=3
                                            ),
                                            dbc.Col([
                                                dbc.Input(
                                                    id='interactomics-num-controls', type='number', value=30,
                                                    min=0, max=200, step=1, style={'margin': '5px', 'verticalAlign': 'center'}
                                                ),
                                                tooltips.interactomics_select_top_controls_tooltip()
                                            ], width=3),
                                            dbc.Col(
                                                html.P('most similar inbuilt control runs',
                                                    style={'margin': '5px', 'verticalAlign': 'center'}),
                                                width=6
                                            )
                                        ])
                                    ],
                                    hidden=True,
                                    id='interactomics-nearest-controls-div'
                                )
                            ], width=6),
                            dbc.Col(width=1)
                        ])
                    ], width=9),
                    dbc.Col(
                        interactomics_enrichment_col(parameters['enrichment']),
                        width=3
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Button('Run SAINT analysis',
                               id='button-run-saint-analysis'),
                ]
            )
        ]
    )


def saint_filtering_container(
    defaults: Dict[str, Any], 
    rescue: bool,
    saint_found: bool
) -> html.Div:
    """Creates a container for SAINT analysis filtering controls and visualizations.

    Args:
        defaults (dict): Default configuration including:
            - config (dict): Graph configuration settings
        rescue (bool): Whether rescue filtering is enabled

    Returns:
        html.Div: Container with:
            - SAINT BFDR histogram
            - Filtered prey counts visualization
            - BFDR threshold slider
            - CRAPome filtering controls
            - SPC fold change threshold slider
            - Filtering completion button

    Notes:
        SAINT (Significance Analysis of INTeractome) filtering allows users to:
        - Visualize BFDR value distributions
        - Set filtering thresholds for interactions
        - Control CRAPome-based contaminant filtering
        - Adjust rescue thresholds for borderline interactions
    """
    return html.Div(
        id={'type': 'input-div', 'id': 'interactomics-saint-filtering-area'},
        children=[
            html.Div(
                id='interactomics-saint-has-error',
                children=[
                    html.Div(
                        'SAINT EXECUTABLE WAS NOT FOUND, SCORING DATA IS RANDOMIZED',
                        style={
                            'fontSize': '24px',
                            'fontWeight': 'bold',
                            'textDecoration': 'underline',
                            'color': 'black',
                            'backgroundColor': 'red',
                            'padding': '10px',
                        }
                    ),
                    html.Div('If this is the demo version, this is expected behavior. Otherwise, you need to rebuild the docker image with SAINTExpress available in the expected folder (see README), or you need to add SAINTexpress as executable to the container itself, and make sure it\'s in PATH.')
                ],
                hidden = saint_found, 
            ),
            html.H4(id='interactomics-saint-histo-header',
                    children='SAINT BFDR value distribution'),
            dcc.Graph(id='interactomics-saint-bfdr-histogram',
                      config=defaults['config']),
            interactomics_legends['saint-histo'],
            html.H4(id='interactomics-saint-filtered-counts-header',
                    children='Filtered Prey counts per bait'),
            dcc.Graph(id='interactomics-saint-graph',
                      config=defaults['config']),
            saint_legend(rescue),
            dbc.Label('Saint BFDR threshold:'),
            dcc.Slider(0, 0.1, 0.01, value=0.05,
                       id='interactomics-saint-bfdr-filter-threshold'),
            dbc.Label('Crapome filtering percentage:'),
            dcc.Slider(1, 100, 10, value=20,
                       id='interactomics-crapome-frequency-threshold'),
            dbc.Label('SPC fold change vs crapome threshold for rescue'),
            dcc.Slider(0, 10, 1, value=3,
                       id='interactomics-crapome-rescue-threshold'),
            html.Div(
                [dbc.Button('Done filtering', id='interactomics-button-done-filtering')])
        ]
    )


def post_saint_container() -> List[html.Div]:
    """Creates a container for post-SAINT analysis visualizations.

    Returns:
        list: Container with loading indicators for:
            - Known interactions plot
            - Common interactions plot
            - PCA analysis
            - Network visualization
            - Volcano plot
            - MS microscopy analysis
            - Enrichment analysis

    Notes:
        Each visualization has its own loading indicator that displays
        while the corresponding analysis is being performed.
    """
    return [
        html.Div(
            id={'type': 'workflow-area', 'id': 'interactomcis-count-plot-div'},
            children=[
                dcc.Loading(id='interactomics-known-loading'),
                dcc.Loading(id='interactomics-common-loading'),
                dcc.Loading(id='interactomics-pca-loading'),
                dcc.Loading(id='interactomics-network-loading'),
                dcc.Loading(id='interactomics-volcano-loading'),
                dcc.Loading(id='interactomics-msmic-loading'),
                dcc.Loading(id='interactomics-enrichment-loading'),
            ]
        ),
    ]


def interactomics_area(
    parameters: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> List[html.Div]:
    """Creates the main interactomics analysis area with input controls and results display.

    Args:
        parameters (dict): Interactomics-specific configuration parameters
        data_dictionary (dict): Data required for interactomics analysis

    Returns:
        list: List containing:
            - Input div with interactomics-specific controls
            - Results div with:
                - SAINT analysis container
                - SAINT processing indicator
                - Post-SAINT analysis area

    Notes:
        The area is organized into distinct sections for input controls
        and analysis results, with loading indicators for long-running
        operations.
    """
    return [
        html.Div(
            id={'type': 'input-div', 'id': 'interactomics-analysis-area'},
            children=[
                html.H1('Interactomics-specific input options'),
                interactomics_input_card(parameters, data_dictionary),
                html.Hr()
            ]
        ),
        html.Div(
            id={'type': 'analysis-div', 'id': 'interactomics-analysis-results-area'},
            children=[
                html.H1(id='interactomics-main-header',
                        children='Interactomics'),
                dcc.Loading(
                    id='interactomics-saint-container-loading',
                    children=html.Div(id={'type': 'workflow-plot', 'id': 'interactomics-saint-container'})),
                dcc.Loading(id='interactomics-saint-running-loading'),
                html.Div(id={'type': 'analysis-div',
                         'id': 'interactomics-analysis-post-saint-area'},)
            ]
        ),
    ]


def phosphoproteomics_area(
    parameters: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> list:
    """Creates the main phosphoproteomics analysis area.

    Args:
        parameters (dict): Phosphoproteomics-specific configuration parameters
        data_dictionary (dict): Data required for phosphoproteomics analysis

    Returns:
        html.Div: Container for phosphoproteomics analysis components

    Notes:
        Currently returns an empty div as placeholder for future implementation.
    """
    return [html.Div(id={'type': 'analysis-div', 'id': 'phosphoproteomics-analysis-area'})]


def qc_area() -> html.Div:
    """Creates the quality control analysis area with multiple QC visualizations.

    Returns:
        html.Div: Container with loading indicators for QC plots including:
            - Total Ion Current (TIC) plot
            - Protein count plot
            - Common protein plot
            - Coverage plot
            - Reproducibility plot
            - Missing values plot
            - Sum intensity plot
            - Mean intensity plot
            - Distribution plot
            - Commonality plot

    Notes:
        Each QC plot has its own loading indicator that displays while
        the corresponding analysis is being performed. The plots are
        arranged vertically in the order listed above.
    """
    return html.Div(
        id='qc-area',
        children=[
            html.H1(id='qc-main-header', children='Quality control'),
            dcc.Loading(
                id='qc-loading-tic',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'tic-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-count',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'count-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-common-protein',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'common-protein-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-coverage',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'coverage-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-reproducibility',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'reproducibility-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-missing',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'missing-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-sum',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'sum-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-mean',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'mean-plot-div'}),
                type='default'
            ),
            dcc.Loading(
                id='qc-loading-distribution',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'distribution-plot-div'}),
                type='default'
            ),
            html.Div(id={'type': 'qc-plot', 'id': 'commonality-plot-div'})
        ])


def navbar(navbar_pages: List[Tuple[str, str]]) -> dbc.NavbarSimple:
    """Creates the main navigation bar for the application.

    Args:
        navbar_pages (list): List of tuples containing (name, link) pairs
            for each navigation item

    Returns:
        dbc.NavbarSimple: Bootstrap navbar component with:
            - Navigation items generated from navbar_pages
            - 'Quick analysis' brand text
            - Primary color theme
            - Dark mode enabled

    Notes:
        Creates a simple horizontal navigation bar with consistent styling
        across the application.
    """
    navbar_items: List[dbc.NavItem] = [
        dbc.NavItem(dbc.NavLink(name, href=link)) for name, link in navbar_pages
    ]
    return dbc.NavbarSimple(
        id='main-navbar',
        children=navbar_items,
        brand='Quick analysis',
        color='primary',
        dark=True
    )


def table_of_contents(
    main_div_children: List[Dict[str, Any]], 
    itern: int = 0
) -> List[Any]:
    """Recursively generates a table of contents from HTML header elements.

    Args:
        main_div_children (list): List of HTML components to process
        itern (int, optional): Current recursion depth. Defaults to 0.

    Returns:
        list: List of HTML components representing the table of contents, including:
            - Title ('Table of contents') at the root level
            - Nested header elements with appropriate styling
            - Links to corresponding sections in the document

    Notes:
        - Processes header elements (H1-H6) and creates corresponding TOC entries
        - Maintains hierarchy through recursive processing of nested components
        - Applies different padding/styling based on header level
        - Creates clickable links using element IDs
        - Skips elements without IDs or non-header elements
        - Limits header levels to H1-H6 (higher levels are treated as H6)

    Example:
        >>> table_of_contents([
        ...     {'type': 'H1', 'props': {'id': 'section1', 'children': 'Section 1'}},
        ...     {'type': 'H2', 'props': {'id': 'subsection', 'children': 'Subsection'}}
        ... ])
        [
            html.H3('Table of contents'),
            html.Div(html.H1(html.A(href='#section1', children='Section 1'))),
            html.Div(html.H2(html.A(href='#subsection', children='Subsection')))
        ]
    """
    ret: List[Any] = []
    if itern == 0:
        ret.append(html.H3('Table of contents'))
    if main_div_children is None:
        return ret
    if isinstance(main_div_children, dict):
        ret.extend(table_of_contents(
            main_div_children['props']['children'], itern+1)) # type: ignore
    else:
        for element in main_div_children:
            try:
                kids: List[Any] | str | Dict[str, Any] = element['props']['children']
            except KeyError:
                continue
            except TypeError:
                continue
            ctype: str = element['type']
            if isinstance(kids, list):
                ret.extend(table_of_contents(kids, itern + 1))
            elif isinstance(kids, str):
                if ctype.startswith('H'):
                    level = int(ctype[1])
                    if level > 6:
                        level = 6
                    html_component: Any = HEADER_DICT['component'][level]
                    list_component: Any = html.Div
                    style: Dict[str, Any] = SIDEBAR_LIST_STYLES[level]
                    if level == 1:
                        style['padding-left'] = '0%'
                    try:
                        idstr: str = element['props']['id']
                    except KeyError:
                        continue
                    ret.append(
                        list_component(
                            html_component(
                                html.A(
                                    href=f'#{idstr}',
                                    children=kids,
                                ),
                                style=style
                            )
                        )
                    )
            elif isinstance(kids, dict):
                ret.extend(table_of_contents(
                    kids['props']['children'], itern+1))
    return ret
