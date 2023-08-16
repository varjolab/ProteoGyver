"""Components for the user interface"""


import dash_bootstrap_components as dbc
from dash import dcc, html
from element_styles import SIDEBAR_STYLE, UPLOAD_A_STYLE, UPLOAD_STYLE, UPLOAD_BUTTON_STYLE, CONTENT_STYLE, SIDEBAR_LIST_STYLES,UPLOAD_INDICATOR_STYLE
import text_functions
from typing import Any

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
        simple_text_clean: bool = False,
        id_only:bool=False,
        prefix_list:list = None,
        postfix_list:list = None
        ) -> dbc.Checklist:
    if disabled is None:
        disabled: set = set()
    else:
        disabled: set = set(disabled)
    checklist_id: str
    if simple_text_clean:
        checklist_id = f'{id_prefix}-{label.strip(":").strip().replace(" ","-").lower()}'
    else:
        checklist_id = f'{id_prefix}-{text_functions.clean_text(label.lower())}'
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
        style=SIDEBAR_STYLE,

    )

def main_content_div() -> html.Div:
    return html.Div(
        id='main-content-div',
        children=[
            html.Div(
                id={'type': 'analysis-div','id':'qc-analysis-area'},
                children=[
                ]
            ),
            html.Div(id = 'workflow-specific-div')
            ],
        style=CONTENT_STYLE
    )

def workflow_area(workflow) -> html.Div:
    ret: html.Div
    if workflow == 'Proteomics':
        ret = proteomics_area()
    elif workflow == 'Interactomics':
        ret = interactomics_area()
    elif workflow == 'Phosphoproteomics':
        ret = phosphoproteomics_area()
    return ret


def proteomics_area() -> html.Div:
    return html.Div(id={'type': 'analysis-div','id':'proteomics-analysis-area'}),
def interactomics_area() -> html.Div:
    return html.Div(id={'type': 'analysis-div','id':'phosphoproteomics-analysis-area'}),
def phosphoproteomics_area() -> html.Div:
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

def table_of_contents(main_div_children: list, itern = 0) -> list:
    ret: list = []
    if isinstance(main_div_children, dict):
        ret.extend(table_of_contents(main_div_children['props']['children'], itern+1))
    else:
        for element in main_div_children:
            try:
                kids: list | str = element['props']['children']
            except KeyError:
                continue
            ctype: str = element['type']
            if isinstance(kids, list):
                ret.extend(table_of_contents(kids), itern + 1)
            elif isinstance(kids, str):
                if ctype.startswith('H'):
                    level = int(ctype[1])
                    if level > 6:
                        level = 6
                    html_component: Any = HEADER_DICT['component'][level]
                    idstr: str = element['props']['id']
                    ret.append(
                        html.Li(
                            html_component(
                                html.A(
                                    href=f'#{idstr}', 
                                    children=kids,
                                ),
                                style = SIDEBAR_LIST_STYLES[level]
                            )
                        )
                    )
            elif isinstance(kids, dict):
                ret.extend(table_of_contents(kids['props']['children'], itern+1))
    return ret

def old_table_of_contents(contents) -> list:
    headers: list = [html.H1('Contents:', style=SIDEBAR_LIST_STYLES[1]), html.Ul(children=[])]
    for element in contents:
        for child_element in element['props']['children']:
            ctype: str = child_element['type']
            if ctype.startswith('H'):
                if len(ctype) == 2:
                    level = int(ctype[1])
                    child_id: str = child_element['props']['id']
                    child_heading: str = child_element['props']['children']
                    if level > 6:
                        level = 6
                    html_component: Any = HEADER_DICT['component'][level]
                    headers[1].children.append(
                        html.Li(
                            html_component(
                                html.A(
                                    href=f'#{child_id}', 
                                    children=child_heading,
                                ),
                                style = SIDEBAR_LIST_STYLES[level]
                            )
                        )
                    )
    return headers