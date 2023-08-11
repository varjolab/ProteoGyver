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

def upload_area(id_text, upload_name) -> html.Div:
    return html.Div(
        [
            html.Div(
                children = [
                    dcc.Upload(
                        id=id_text,
                        children=html.Div([
                            'Drag and drop or ',
                            html.A('select',style=UPLOAD_A_STYLE),
                            f' {upload_name}'
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
            html.H2("Sidebar", style={'textAlign': 'center'}),
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
            dbc.Button(
                'Begin!',
                id='input-complete-button',
                style=UPLOAD_BUTTON_STYLE,
                className='btn-info',
            ),
            dbc.Button(
                'Download all data',
                id='download-all-data-button',
                style=UPLOAD_BUTTON_STYLE,
                className='btn-info',
                disabled=True,
            ),
            html.Div(id='toc-div',children=[]),
            dcc.Download(id='download-sample_table-template'),
            dcc.Download(id='download-datafile-example'),
            dcc.Download(id='download-all-data'),
            dbc.Button(
                'call1',
                id='call1-button'
            ),

        ],
        className='card text-white bg-primary mb-3',
        style=SIDEBAR_STYLE,

    )

def main_content_div() -> html.Div:
    return html.Div(
        id="first-div",
        children=[
            'discard samples checklist here',
            html.Div([
                html.Div(id='upload-complete-indicator'),
                dcc.Loading(
                    id='qc-loading',
                    children=[
                        html.Div(id='qc-plot-container',
                                children=[
                                ])
                    ],
                    type='default'
                ),
                html.Hr(),
                html.P(
                    "First row stuff", className="lead"
                )
            ]),
            html.Div(id='tic-graph'),

            # second row
            html.Div([
                html.H2("Second Row"),
                html.Hr(),
                html.P(
                    "Second row stuff", className="lead"
                )
            ]),
            html.Div(id='contents-div', children=[])

        ],
        style=CONTENT_STYLE
    )

def table_of_contents(contents) -> list:
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