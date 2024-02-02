"""Components for the user interface"""


import dash_bootstrap_components as dbc
from itertools import chain
from dash import dcc, html
from element_styles import SIDEBAR_STYLE, UPLOAD_A_STYLE, UPLOAD_STYLE, UPLOAD_BUTTON_STYLE, CONTENT_STYLE, SIDEBAR_LIST_STYLES, UPLOAD_INDICATOR_STYLE
from typing import Any
from components import tooltips, text_handling
from numpy import log2
from components.parsing import guess_control_samples
from components.figures.figure_legends import INTERACTOMICS_LEGENDS as interactomics_legends

HEADER_DICT: dict = {
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
        options: list,
        default_choice: list,
        disabled: list = None,
        id_prefix: str = None,
        id_only: bool = False,
        prefix_list: list = None,
        postfix_list: list = None,
        clean_id = True,
        style_override: dict = None
) -> dbc.Checklist:
    if disabled is None:
        disabled: set = set()
    else:
        disabled: set = set(disabled)
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
    retlist: html.Div = [
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


def range_input(label, min, max, id_str, typestr = 'number'):
    pad = {
        'padding': '0px 5px 0px 5px', 'margin': 'auto','margin-left': 'auto', 'margin-right': 'auto'
    }
    return html.Div(
        children = [
            html.P(label, style=pad|{'width': '50%'}),
            dcc.Input(id=f'{id_str}-min', type=typestr,value=min,style=pad|{'width': '20%'}),
            html.P('-', style=pad|{'width': '10%'}),
            dcc.Input(id=f'{id_str}-max', type=typestr,value=max,style=pad|{'width': '20%'}),
        ],
        style={'display': 'flex', 'width': '100%', 'height': '20px',
                'float': 'center', 'height': '20px'},
        id = id_str
    )

def upload_area(id_text, upload_id, indicator=True) -> html.Div:
    ret: list = [
        html.Div(
            children=[
                dcc.Upload(
                    id=id_text,
                    children=html.Div([
                        'Drag and drop or ',
                        html.A('select', style=UPLOAD_A_STYLE),
                        f' {upload_id}'
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

def main_sidebar(figure_templates: list, implemented_workflows: list) -> html.Div:
    return html.Div(
        [
            html.H2("Input", style={'textAlign': 'center'}),
            dbc.Button(
                'Download example sample table',
                style=UPLOAD_BUTTON_STYLE,
                id='button-download-sample_table-template',
                className='btn-info',
            ),
            dbc.Button(
                'Download example data file',
                style=UPLOAD_BUTTON_STYLE,
                id='button-download-datafile-example',
                className='btn-info',
            ),
            html.H4('Upload files:'),
            upload_area('upload-data-file', 'Data file'),
            upload_area('upload-sample_table-file', 'Sample table'),
            html.Br(),
            dcc.Checklist(
                id='sidebar-remove-common-contaminants',
                options=['Remove common contaminants'], value=['Remove common contaminants']
            ),
            dcc.Checklist(
                id='sidebar-rename-replicates',
                options=['Rename replicates'], value=[]
            ),
            html.H4('Select workflow:'),
            dbc.Select(
                options=[
                    {'label': item, 'value': item} for item in implemented_workflows
                ],
                id='workflow-dropdown',
            ),
            html.H4('Select figure style:'),
            dbc.Select(
                value=figure_templates[0],
                options=[
                    {'label': item, 'value': item} for item in figure_templates
                ],
                id='figure-theme-dropdown',
            ),
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
                'Begin analysis',
                id='begin-analysis-button',
                style=UPLOAD_BUTTON_STYLE,
                className='btn-info',
                disabled=True,
            ),
            dbc.Button(
                children = [
                    dbc.Spinner(
                        children = html.Div(id='button-download-all-data-spinner-output'),
                        size = 'sm',
                        show_initially = False,
                        delay_show = 10,
                        delay_hide = 10,
                    ), ' Download all data'
                ],
                style=UPLOAD_BUTTON_STYLE,
                id='button-download-all-data',
                className='btn-info',
                disabled=True,
            ),
            # top right bottom left
            html.Div(id='toc-div', style={'padding': '0px 10px 10px 30px'}),
            dcc.Download(id='download-sample_table-template'),
            dcc.Download(id='download-datafile-example'),
            dcc.Download(id='download-proteomics-comparison-example'),
            dcc.Download(id='download-all-data')
            

        ],
        className='card text-white bg-primary mb-3',
        style=SIDEBAR_STYLE
    )


def modals() -> html.Div:
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


def workflow_area(workflow: str, workflow_specific_parameters: dict, data_dictionary: dict) -> html.Div:
    ret: html.Div
    if workflow == 'Proteomics':
        ret = proteomics_area(
            workflow_specific_parameters['proteomics'], data_dictionary)
    elif workflow == 'Interactomics':
        ret = interactomics_area(
            workflow_specific_parameters['interactomics'], data_dictionary)
    elif workflow == 'Phosphoproteomics':
        ret = phosphoproteomics_area(
            workflow_specific_parameters['phosphoproteomics'], data_dictionary)
    return ret


def proteomics_input_card(parameters: dict, data_dictionary: dict) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Label('NA Filtering:', id='filtering-label'),
                        tooltips.na_tooltip()
                    ]),
                    dcc.Slider(0, 100, 10, value=parameters['na_filter_default_value'],
                               id='proteomics-filter-minimum-percentage'),
                ], width=5),
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
                                sorted(
                                    list(data_dictionary['sample groups']['norm'].keys()))
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
                        for x in (1.5, 2, 3, 4, 5)
                    ], 1, id='proteomics-fc-value-threshold'),
                ], width=6),
                dbc.Col([
                    dbc.Label('Adjusted p-value threshold for comparisons:'),
                    dcc.RadioItems([0.001, 0.01, 0.05], 0.01,
                                   id='proteomics-p-value-threshold'),
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

def windowmaker_input_options( offered_equations):
    premade_eqs = [dbc.Label('Load pre-defined line equation:')]
    for equation, eqname in offered_equations:
        premade_eqs.append(
            html.Div(
                dbc.Button(
                    equation, 
                    id={'type': 'PREDEFEQBUTTON', 'name':eqname},
                    style={'padding': '5px 5px 5px 5px'}, disabled=True
                ),style = {'display': 'block','padding': '5px 5px 5px 5px'}
            )
        )
    input_row_list = [
        dbc.Row([
            dbc.Col([
                range_input('MZ Range: ', 400, 1200, 'windowmaker-mz-input')
            ]),
            dbc.Col([
                range_input('IM Range: ', 0.8, 1.4, 'windowmaker-mob-input')
            ]),
            html.Hr(style={'margin-top': '25px'})
        ],style={'padding': '5px 5px 5px 5px'}),
        dbc.Row([
            dbc.Col(
                premade_eqs
            ),
            dbc.Col([
                dcc.Input(id='windowmaker-line-equation-input',type='text',placeholder='input line equation',style={'padding': '5px 5px 5px 5px'}),
                html.Div([dbc.Button('Add line', id='windowmaker-add-line-button', disabled=True)], style={'padding': '15px 5px 5px 5px'})
            ]),
            dbc.Col([
                dbc.Label('Enabled filters:'),
                html.Ul(id='enabled-filters-list')
            ]),
            dbc.Col(id='windowmaker-filter-col'),
            html.Hr(style={'margin-top': '25px'})
        ],style={'padding': '5px 5px 5px 5px'})
    ]
    input_row_list.append(
        dbc.Row([
            html.Br(),
            html.Div(id='windowmaker-equations-list-group',children=[], style={'width': '100%', 'display': 'block'}),
            dbc.Button('Calculate windows', id='windowmaker-calculate-windows-button', style={'padding': '5px 5px 5px 5px'}, disabled=True)
        ],style={'padding': '5px 5px 5px 5px'})
    )
    return input_row_list

def windowmaker_interface(wmstyle, offered_equations) -> html.Div:
    return html.Div([
        html.Div(id='infra',children=[
            dcc.Store(id='windowmaker-full-data-store'),
            dcc.Store(id='windowmaker-filtered-data-store'),
            dcc.Store(id='windowmaker-prev-clicks-data-store', data={}),
            dcc.Store(id='windowmaker-filter-columns',data={}),
            html.Div(id='placeholder'),
            dcc.Download(id='windowmaker-download-method')
        ]),
        dbc.Row(
            id='windowmaker-input-row',
            children=[
                dbc.Card([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.H3('Upload an MGF file or a spectral library'),
                                html.P('Library MUST be in a text format (tsv, txt, csv, xlsx)'),
                                dcc.Checklist(
                                    id='windowmaker-play-notification-sound-when-done',
                                    options=['Play notification sound when done'], value=['Play notification sound when done']
                                ),
                            ]), width=6
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    upload_area('windowmaker-mgf-file-upload',
                                            'MGF/library file', indicator=True),
                                    style={'padding': '15px 0px 0px 0px'}
                                ),
                                html.P(id='windowmaker-input-file-info-text', style={'text-align': 'left'})
                            ],
                            width=6
                        )
                    ])
                ], body=True)
            ]
        ),
        dbc.Row(
            id='windowmaker-mod-row',
            children=[
                dbc.Col(
                    children=[
                        dbc.Card(
                            dcc.Loading(
                                html.Div(id='windowmaker-pre-plot-area')
                            ),body=True
                        ),
                        dbc.Card(
                            dcc.Loading(
                                html.Div(id='windowmaker-ch-plot-area')
                            ),body=True
                        )
                    ],
                    width=6
                ),
                dbc.Col(
                    dbc.Card(windowmaker_input_options(offered_equations), body=True),
                    width=6
                )
            ]
        ),
        dbc.Row(
            id='windowmaker-output-row',
            children=[
                dbc.Card([
                    dcc.Loading(html.Div(id='windowmaker-post-plot-area'))
                ], body=True)
            ]
        )
    ],style=wmstyle)

def proteomics_area(parameters: dict, data_dictionary: dict) -> html.Div:

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
                    id='proteomics-loading-imputation',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-imputation-plot-div'}),
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
                    id='proteomics-loading-volcano',
                    children=html.Div(
                        id={'type': 'workflow-plot', 'id': 'proteomics-volcano-plot-div'}),
                    type='default'
                ),
            ]
        )
    ]


def discard_samples_checklist(count_plot, list_of_samples) -> html.Div:
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


def interactomics_control_col(all_sample_groups, chosen) -> dbc.Col:
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


def interactomics_inbuilt_control_col(controls_dict) -> dbc.Col:
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
                disabled=controls_dict['disabled'],
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[dbc.Label('Choose additional control sets:')]
            )
        ),
    ])

def get_windowmaker_filcol_id(filname): 
    return 'exclude-'+text_handling.replace_special_characters(filname,'-', stripresult=True, remove_duplicates=True)

def generate_filter_group(data, filcols):
    dropdown = dcc.Dropdown(filcols, id='windowmaker-filter-col-dropdown',value=filcols[0])
    checklists = []
    checklist_target_cols = []
    for fcol in filcols:
        if fcol == 'Modifications':
            values = sorted(
                list(
                    set(
                        chain.from_iterable([
                            mods.split(';') for mods in data[fcol]
                        ])
                    )
                )
            )
        else:
            values = sorted(list(data[fcol].unique()))
        values = [str(x) for x in values]
        f = text_handling.replace_special_characters(fcol,'-', stripresult=True, remove_duplicates=True)
        checklist_target_cols.append(fcol)
        checklists.append(
            html.Div(
                id={'type': 'windowmaker-filter-div', 'name': f'exclude-{f}'},
                children = checklist(
                    {'type': 'windowmaker-filter-checklist', 'name': f},
                    values,
                    [],
                    id_only=True,
                    clean_id=False,
                    prefix_list = [dbc.Label(f'Exclude values:')]
                ),
                hidden=True
            )
        )
    return dropdown, checklists, checklist_target_cols

def interactomics_crapome_col(crapome_dict) -> dbc.Col:
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
                disabled=crapome_dict['disabled'],
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[dbc.Label('Choose Crapome sets:')]
            )
        )
    ])


def interactomics_enrichment_col(enrichment_dict) -> dbc.Col:
    return dbc.Col([
        html.Div(
            checklist(
                'Choose enrichments:',
                enrichment_dict['available'],
                enrichment_dict['default'],
                disabled=enrichment_dict['disabled'],
                id_prefix='interactomics',
                id_only=True,
                prefix_list=[dbc.Label('Choose enrichments:')]
            )
        )
    ])


def interactomics_input_card(parameters: dict, data_dictionary: dict) -> html.Div:
    all_sample_groups: list = []
    sample_groups: dict = data_dictionary['sample groups']['norm']
    chosen: list = guess_control_samples(list(sample_groups.keys()))
    for k in sample_groups.keys():
        if k not in chosen:
            all_sample_groups.append(k)
    all_sample_groups = sorted(chosen) + sorted(all_sample_groups)
    return html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Row(
                            [
                                interactomics_control_col(
                                    all_sample_groups, chosen),
                                interactomics_inbuilt_control_col(
                                    parameters['controls']),
                                interactomics_crapome_col(
                                    parameters['crapome']),
                            ]
                        ),
                        dbc.Row([
                            dbc.Col(width=1),
                            dbc.Col(
                                checklist(
                                    'Rescue filtered out',
                                    ['Rescue interactions that pass filter in any sample group'],
                                    ['Rescue interactions that pass filter in any sample group'],
                                    id_only=True,
                                    id_prefix='interactomics',
                                    style_override={
                                        'margin': '5px', 'verticalAlign': 'center'
                                    }
                                ), width=4
                            ),
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col(
                                        checklist(
                                            'Nearest control filtering',
                                            ['Select'],
                                            [''],
                                            id_only=True,
                                            id_prefix='interactomics',
                                            style_override={
                                                'margin': '5px', 'verticalAlign': 'center'
                                            },
                                        ), width=2
                                    ),
                                    dbc.Col([
                                        dbc.Input(
                                            id='interactomics-num-controls', type='number', value=30,
                                            min=0, max=200, step=1, style={'margin': '5px', 'verticalAlign': 'center'}
                                        ),
                                        tooltips.interactomics_select_top_controls_tooltip()
                                    ], width=2),
                                    dbc.Col(
                                        html.P('most similar inbuilt control runs',
                                               style={'margin': '5px', 'verticalAlign': 'center'}),
                                        width=7
                                    )
                                ])
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


def saint_filtering_container(defaults) -> list:
    return html.Div(
        id={'type': 'input-div', 'id': 'interactomics-saint-filtering-area'},
        children=[
            html.H4(id='interactomics-saint-histo-header',
                    children='SAINT BFDR value distribution'),
            dcc.Graph(id='interactomics-saint-bfdr-histogram',
                      config=defaults['config']),
            interactomics_legends['saint-histo'],
            html.H4(id='interactomics-saint-filtered-counts-header',
                    children='Filtered Prey counts per bait'),
            dcc.Graph(id='interactomics-saint-graph',
                      config=defaults['config']),
            interactomics_legends['filtered-saint-counts'],
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


def post_saint_cointainer() -> list:
    return [
        html.Div(
            id={'type': 'workflow-area', 'id': 'interactomcis-count-plot-div'},
            children=[
                dcc.Loading(id='interactomics-known-loading'),
                dcc.Loading(id='interactomics-pca-loading'),
                dcc.Loading(id='interactomics-network-loading'),
                dcc.Loading(id='interactomics-volcano-loading'),
                dcc.Loading(id='interactomics-msmic-loading'),
                dcc.Loading(id='interactomics-enrichment-loading'),
            ]
        ),
    ]


def interactomics_area(parameters: dict, data_dictionary: dict) -> html.Div:
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


def phosphoproteomics_area(parameters: dict, data_dictionary: dict) -> html.Div:
    return html.Div(id={'type': 'analysis-div', 'id': 'phosphoproteomics-analysis-area'}),


def qc_area() -> html.Div:
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
            dcc.Loading(
                id='qc-loading-commonality',
                children=html.Div(
                    id={'type': 'qc-plot', 'id': 'commonality-plot-div'}),
                type='default'
            ),
        ])


def navbar(navbar_pages) -> dbc.NavbarSimple:
    navbar_items: list = [
        dbc.NavItem(dbc.NavLink(name, href=link)) for name, link in navbar_pages
    ]
    return dbc.NavbarSimple(
        id='main-navbar',
        children=navbar_items,
        brand='Quick analysis',
        color='primary',
        dark=True
    )


def table_of_contents(main_div_children: list, itern=0) -> list:
    ret: list = []
    if main_div_children is None:
        return ret
    if isinstance(main_div_children, dict):
        ret.extend(table_of_contents(
            main_div_children['props']['children'], itern+1))
    else:
        for element in main_div_children:
            try:
                kids: list | str = element['props']['children']
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
                    # list_component: Any = html.Li
                    list_component: Any = html.Div
                    style: dict = SIDEBAR_LIST_STYLES[level]
                    if level == 1:
                        # list_component = html.Div
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
