"""Components for the user interface"""


import dash_bootstrap_components as dbc
from dash import dcc, html
from element_styles import SIDEBAR_STYLE, UPLOAD_A_STYLE, UPLOAD_STYLE, UPLOAD_BUTTON_STYLE, CONTENT_STYLE, SIDEBAR_LIST_STYLES,UPLOAD_INDICATOR_STYLE
from typing import Any
from components import tooltips, text_handling

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
        id_only:bool=False,
        prefix_list:list = None,
        postfix_list:list = None
        ) -> dbc.Checklist:
    if disabled is None:
        disabled: set = set()
    else:
        disabled: set = set(disabled)
    checklist_id: str
    checklist_id = text_handling.replace_special_characters(
        f'{id_prefix}-{label}',
        '-',stripresult=True,remove_duplicates=True)
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
            switch=True
        )
    ]
    return prefix_list + retlist + postfix_list

def upload_area(id_text, upload_id) -> html.Div:
    return html.Div(
        [
            html.Div(
                children = [
                    dcc.Upload(
                        id=id_text,
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('select',style=UPLOAD_A_STYLE),
                            f' {upload_id}'
                        ]
                        ),
                        style=UPLOAD_STYLE,
                        multiple=False
                    )
                ],
                style={'display': 'inline-block','width':'75%','float':'left','height':'65px'},
            ),
            html.Div(
                id=f'{id_text}-success',
                style=UPLOAD_INDICATOR_STYLE
            ) ##fixthis
        ]
    )

def main_sidebar(figure_templates: list, implemented_workflows: list) -> html.Div:
    return html.Div(
        [
            html.H2("Input", style={'textAlign': 'center'}),
            dbc.Button(
                'Download sample table template',
                style=UPLOAD_BUTTON_STYLE,
                id='button-download-sample_table-template',
                className='btn-info',
            ),
            dbc.Button(
                'Download Data file example',
                style=UPLOAD_BUTTON_STYLE,
                id='button-download-datafile-example',
                className='btn-info',
            ),
            html.H4('Upload files:'),
            upload_area('upload-data-file', 'Data file'),
            upload_area('upload-sample_table-file', 'Sample table'),
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
                children = [
                    dbc.Button(
                        'Choose samples to discard',
                        id='discard-samples-button',
                        style=UPLOAD_BUTTON_STYLE,
                        className='btn-warning',
                        n_clicks = 0
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
                'Download all data',
                id='download-all-data-button',
                style=UPLOAD_BUTTON_STYLE,
                className='btn-info',
                disabled=True,
            ),
            html.Div(id='toc-div',style={'padding': '0px 10px 10px 30px'}),# top right bottom left
            dcc.Download(id='download-sample_table-template'),
            dcc.Download(id='download-datafile-example'),
            dcc.Download(id='download-all-data'),

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
                    dbc.ModalHeader(dbc.ModalTitle('Select samples to discard')),
                    dbc.ModalBody(
                        children=[
                            dbc.Button('Discard samples', id = 'done-discarding-button', n_clicks = 0),
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
                id={'type': 'analysis-div','id':'qc-analysis-area'},
                children=[
                ]
            ),
            html.Div(id = 'workflow-specific-div')
        ],
        style=CONTENT_STYLE
    )

def workflow_area(workflow: str, workflow_specific_parameters: dict) -> html.Div:
    ret: html.Div
    if workflow == 'Proteomics':
        ret = proteomics_area(workflow_specific_parameters['proteomics'])
    elif workflow == 'Interactomics':
        ret = interactomics_area(workflow_specific_parameters['interactomics'])
    elif workflow == 'Phosphoproteomics':
        ret = phosphoproteomics_area(workflow_specific_parameters['phosphoproteomics'])
    return ret

def proteomics_area(parameters) -> html.Div:
    return [
        html.Div(
            id={'type': 'input-div','id':'proteomics-analysis-area'},
            children=[
                html.H1('Proteomics specific input options'),
                html.Div([
                    html.Div([
                        dbc.Label('NA Filtering:', id='filtering-label'),
                        tooltips.na_tooltip()
                    ]),
                    dcc.Slider(0, 100, 10, value=parameters['na_filter_default_value'],
                                id='proteomics-filter-minimum-percentage'),
                    dbc.Select(
                        options=[
                            {'label': 'wait', 'value': 'wait'}
                        ],
                        required=True,
                        id='proteomics-control-dropdown',
                    ),
                ]
            ),
            dbc.Label('Imputation:'),
            dbc.RadioItems(
                options=[
                    {'label': i_opt, 'value': i_opt_val}
                        for i_opt, i_opt_val in parameters['imputation methods'].items()
                ],
                value=parameters['default imputation method'],
                id='proteomics-imputation-radio-option'
            ),
            dbc.Label('Normalization:'),
            dbc.RadioItems(
                options=[
                    {'label': n_opt, 'value': n_opt_val}
                        for n_opt, n_opt_val in parameters['normalization methods'].items()
                ],
                value=parameters['default normalization method'],
                id='proteomics-normalization-radio-option'
            ),
            html.Hr()
        ]), 
        html.Div(
            id={'type': 'analysis-div','id':'proteomics-analysis-area'},
        )
    ]

def discard_samples_checklist(count_plot, list_of_samples) -> html.Div:
    return [
        count_plot,
        html.Div(
            checklist(
                label = 'Select samples to discard',
                id_only=True,
                options = list_of_samples,
                default_choice = [],
                id_prefix = 'checklist'
            )
        )
    ]
def interactomics_area(parameters) -> html.Div:
    return html.Div(id={'type': 'analysis-div','id':'phosphoproteomics-analysis-area'}),
def phosphoproteomics_area(parameters) -> html.Div:
    return html.Div(id={'type': 'analysis-div','id':'interactomics-analysis-area'}),

def qc_area() -> html.Div:
    return html.Div(
        id = 'qc-area',
        children=[
        html.H1(id='qc-main-header', children = 'Quality control'),
        dcc.Loading(
            id='qc-loading-count',
            children=html.Div(id={'type': 'qc-plot', 'id': 'count-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-coverage',
            children=html.Div(id={'type': 'qc-plot', 'id': 'coverage-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-reproducibility',
            children=html.Div(id={'type': 'qc-plot', 'id': 'reproducibility-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-missing',
            children=html.Div(id={'type': 'qc-plot', 'id': 'missing-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-sum',
            children=html.Div(id={'type': 'qc-plot', 'id': 'sum-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-mean',
            children=html.Div(id={'type': 'qc-plot', 'id': 'mean-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-distribution',
            children=html.Div(id={'type': 'qc-plot', 'id': 'distribution-plot-div'}),
            type='default'
        ),
        dcc.Loading(
            id='qc-loading-commonality',
            children=html.Div(id={'type': 'qc-plot', 'id': 'commonality-plot-div'}),
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

def table_of_contents(main_div_children: list, itern = 0) -> list:
    ret: list = []
    if main_div_children is None:
        return ret
    if isinstance(main_div_children, dict):
        ret.extend(table_of_contents(main_div_children['props']['children'], itern+1))
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
                    list_component: Any = html.Li
                    style: dict = SIDEBAR_LIST_STYLES[level]
                    if level == 1:
                        list_component = html.Div
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
                                style = style
                            )
                        )
                    )
            elif isinstance(kids, dict):
                ret.extend(table_of_contents(kids['props']['children'], itern+1))
    return ret
