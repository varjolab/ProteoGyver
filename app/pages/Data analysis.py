"""Dash app for data upload"""

import base64
import io
import json
import dash
import dash_bootstrap_components as dbc
import data_functions
import figure_generation
import pandas as pd
import plotly.io as pio
from dash import callback, dcc, html, DiskcacheManager
from dash.dependencies import Input, Output, State
from DbEngine import DbEngine
from styles import Styles
from utilitykit import dftools, plotting
from dash.exceptions import PreventUpdate

# db will house all data, keep track of next row ID, and validate any new data
db: DbEngine = DbEngine()
#app: Dash = Dash(__name__)
#server: app.server = app.server
dash.register_page(__name__, path='/')
styles: Styles = Styles()
figure_templates = [
    'plotly',
    'plotly_white',
    'plotly_dark',
    'ggplot2',
    'seaborn',
    'simple_white'
]
# dash.register_page(__name__)


def read_df_from_content(content, filename) -> pd.DataFrame:
    _, content_string = content.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    f_end: str = filename.rsplit('.', maxsplit=1)[-1]
    data = None
    if f_end == 'csv':
        data: pd.DataFrame = pd.read_csv(
            io.StringIO(decoded_content.decode('utf-8')))
    elif f_end in ['tsv', 'tab', 'txt']:
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), sep='\t')
    elif f_end in ['xlsx', 'xls']:
        data: pd.DataFrame = pd.read_excel(io.StringIO(decoded_content))
    return data


def add_replicate_colors(data_df, column_to_replicate) -> None:
    need_cols: int = list(
        {
            column_to_replicate[sname] for sname in
            data_df.index.unique()
            if sname in column_to_replicate
        }
    )
    colors: list = plotting.get_cut_colors(number_of_colors=len(need_cols))
    colors = plotting.cut_colors_to_hex(colors)
    colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
    color_column: list = []
    for sn in data_df.index.values:
        color_column.append(colors[column_to_replicate[sn]])
    data_df.loc[:, 'Color'] = color_column


@callback(
    Output('workflow-choice', 'data'),
    Input('workflow-dropdown', 'value'),
    Input('upload-data-file', 'contents'),
    Input('upload-sample_table-file', 'contents'),
    State('session-uid', 'children')
)
def set_workflow(workflow_setting_value, _, __, session_uid) -> str:
    with open(db.get_cache_file(session_uid + '_workflow-choice.txt'), 'w', encoding='utf-8') as fil:
        fil.write(workflow_setting_value)
    return str(workflow_setting_value)


@callback([
    Output('output-data-upload', 'data'),
    Output('output-data-upload-problems', 'children'),
    Output('figure-template-choice', 'data'),
    Output('placeholder', 'children')
],
    Input('figure-template-dropdown', 'value'),
    Input('upload-data-file', 'contents'),
    State('upload-data-file', 'filename'),
    Input('upload-sample_table-file', 'contents'),
    State('upload-sample_table-file', 'filename'),
    State('session-uid', 'children')
)
def process_input_tables(
    figure_template_dropdown_value,
    data_file_contents,
    data_file_name,
    sample_table_file_contents,
    sample_table_file_name,
    session_uid
) -> tuple:
    if figure_template_dropdown_value:
        with open(db.get_cache_file(session_uid + '_figure-template-choice.txt'), 'w', encoding='utf-8') as fil:
            fil.write(figure_template_dropdown_value)
        pio.templates.default = figure_template_dropdown_value
    return_message: list = []
    return_dict: dict = {}
    if data_file_contents is None:
        return_message.append('Missing data table')
    if sample_table_file_contents is None:
        return_message.append('Missing sample_table')
    if len(return_message) == 0:
        table, column_map, sample_groups, rev_sample_groups = data_functions.parse_data(
            data_file_contents,
            data_file_name,
            sample_table_file_contents,
            sample_table_file_name,
            log2_transform=True
        )
        return_dict['table'] = table.to_json(orient='split')
        return_dict['column map'] = column_map
        return_dict['sample groups'] = sample_groups
        return_dict['rev sample groups'] = rev_sample_groups
        with open(db.get_cache_file(session_uid + '_data_dict.json'), 'w', encoding='utf-8') as fil:
            json.dump(return_dict, fil)

        return_message.append('Upload successful')
    return return_dict, ' ; '.join(return_message), figure_template_dropdown_value, ''

@callback(
    Output('qc-plot-container', 'children'),
    Output('rep-colors', 'data'),
    Input('placeholder', 'children'),
    State('output-data-upload', 'data')
)
def quality_control_charts(_: str, data_dictionary: dict) -> list:
    figures: list = []
    rep_colors = {}
    if 'table' in data_dictionary:
        data_table: pd.DataFrame = pd.read_json(
            data_dictionary['table'], orient='split')
        count_data: pd.DataFrame = data_functions.get_count_data(data_table)
        add_replicate_colors(count_data, data_dictionary['rev sample groups'])
        rep_colors: dict = {}
        for sname, sample_color_row in count_data[['Color']].iterrows():
            rep_colors[sname] = sample_color_row['Color']
        data_dictionary['Replicate colors'] = rep_colors
        figures.append(figure_generation.protein_count_figure(count_data))

        na_data: pd.DataFrame = data_functions.get_na_data(data_table)
        na_data['Color'] = [rep_colors[sample_name]
                            for sample_name in na_data.index.values]
        figures.append(figure_generation.missing_figure(na_data))

        sumdata: pd.DataFrame = data_functions.get_sum_data(data_table)
        sumdata['Color'] = [rep_colors[sample_name]
                            for sample_name in sumdata.index.values]
        figures.append(figure_generation.sum_value_figure(sumdata))

        avgdata: pd.DataFrame = data_functions.get_avg_data(data_table)
        avgdata['Color'] = [rep_colors[sample_name]
                            for sample_name in avgdata.index.values]
        figures.append(figure_generation.avg_value_figure(avgdata))

        figures.append(figure_generation.distribution_figure(
            data_table, rep_colors, data_dictionary['rev sample groups'], log2_transform=False))
        # Long-term:
        # - protein counts compared to previous similar samples
        # - sum value compared to previous similar samples
        # - Person-to-person comparisons: protein counts, intensity/psm totals
    return (figures, rep_colors)


@callback(
    Output('download-sample_table-template', 'data'),
    Input('button-download-sample_table-template', 'n_clicks'),
    prevent_initial_call=True,
)
def sample_table_download(n_clicks):
    return dcc.send_file(db.request_file('assets', 'example-sample_table'))

@callback(
    Output('download-datafile-example', 'data'),
    Input('button-download-datafile-example', 'n_clicks'),
    prevent_initial_call=True,
)
def download_data_table(_):
    return dcc.send_file(db.request_file('assets', 'example-data_file'))

def filter_missing(data_table: pd.DataFrame, sample_groups: dict, threshold: float = 0.6) -> pd.DataFrame:
    keeps: list = []
    for _, row in data_table.iterrows():
        keep: bool = False
        for _, sample_columns in sample_groups.items():
            keep = keep | (row[sample_columns].notna().sum()
                           >= (threshold*len(sample_columns)))
            if keep:
                break
        keeps.append(keep)
    return data_table[keeps]

def count_per_sample(data_table: pd.DataFrame, rev_sample_groups: dict) -> pd.Series:
    index: list = list(rev_sample_groups.keys())
    retser: pd.Series = pd.Series(
        index=index,
        data=[data_table[i].notna().sum() for i in index]
    ).to_json()
    return retser


def impute(data_table, method='QRILC') -> pd.DataFrame:
    ret: pd.DataFrame = data_table
    if method == 'minProb':
        ret = dftools.impute_minprob_df(data_table)
    elif method == 'minValue':
        ret = dftools.impute_minval(data_table)
    elif method == 'QRILC':
        ret = dftools.impute_qrilc(data_table, tempdir = db.temp_dir)
    return ret


def normalize(data_table: pd.DataFrame, normalization_method: str) -> pd.DataFrame:
    if normalization_method == 'Median':
        data_table: pd.DataFrame = data_functions.median_normalize(data_table)
    elif normalization_method == 'Quantile':
        data_table = data_functions.quantile_normalize(data_table)
    return data_table


@callback(
    Output('data-processing-figures', 'children'),
    Output('processed-proteomics-data','data'),
    Input('filter-minimum-percentage', 'value'),
    Input('imputation-radio-option', 'value'),
    Input('normalization-radio-option', 'value'),
    State('output-data-upload', 'data'),
    
)
def make_data_processing_figures(filter_threshold, imputation_method, normalization_method, data_dictionary) -> list:
    if isinstance(data_dictionary, dict):
        if 'table' in data_dictionary:
            # define the data:
            data_table: pd.DataFrame = pd.read_json(
                data_dictionary['table'], orient='split')
            sample_groups: dict = data_dictionary['sample groups']
            rev_sample_groups: dict = data_dictionary['rev sample groups']

            # Filter by missing value proportion
            original_counts: pd.Series = count_per_sample(
                data_table, rev_sample_groups)
            data_table = filter_missing(
                data_table, sample_groups, threshold=filter_threshold)
            filtered_counts: pd.Series = count_per_sample(
                data_table, rev_sample_groups)
            data_dictionary['filter data'] = [original_counts, filtered_counts]

            data_dictionary['normalization data'] = [data_table.to_json(orient='split')]
            # Normalization, if needed:
            if normalization_method:
                data_table = normalize(data_table, normalization_method)
                data_dictionary['normalization data'].append(data_table.to_json(orient='split'))
            data_table = impute(
                data_table, method=imputation_method)
            data_dictionary['final data table'] = data_table.to_json(orient='split')
            
            return [
                [
                    dcc.Loading(type='circle',id='proteomics-filtering-figure'),
                    dcc.Loading(type='circle',id='proteomics-normalization-figure'),
                    dcc.Loading(type='circle',id='proteomics-imputation-figure'),
                    dcc.Loading(type='circle',id='proteomics-distribution-figure'),
                    dcc.Loading(type='circle',id='proteomics-cv-figure'),
                    dcc.Loading(type='circle',id='proteomics-pca-figure'),
                    dcc.Loading(type='circle',id='proteomics-tsne-figure'),
                    dcc.Loading(type='circle',id='proteomics-correlation-clustermap-figure'),
                    dcc.Loading(type='circle',id='proteomics-full-clustermap-figure'),
                    dcc.Loading(type='circle',id='proteomics-volcano-plots'),
                ],
                data_dictionary
                ]
    return []
@callback(
    Output('proteomics-filtering-figure', 'children'),
    Input('data-processing-figures', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_filter_figure(_,data_dictionary) -> list:
    original_counts, filtered_counts = data_dictionary['filter data']
    return [
        figure_generation.before_after_plot(
            pd.read_json(original_counts, typ='series'), 
            pd.read_json(filtered_counts, typ='series'), 
            data_dictionary['rev sample groups'],
            title='NA Filtering'
            )
        ]

@callback(
    Output('proteomics-normalization-figure', 'children'),
    Input('proteomics-filtering-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    
    
)
def proteomics_normalization_figure(_,data_dictionary,) -> list:
    if len(data_dictionary['normalization data']) < 2:
        return ''
    pre_norm, data_table= data_dictionary['normalization data']
    pre_norm:pd.DataFrame = pd.read_json(pre_norm, orient='split')
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.comparative_violin_plot(
        [pre_norm, data_table],
        names=['Before normalization', 'After normalization'],
        id_name='normalization-plot', title='Normalization'
    )]

@callback(
    Output('proteomics-imputation-figure', 'children'),
    Input('proteomics-normalization-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    
    
)
def proteomics_imputation_figure(_, data_dictionary) -> list:
    pre_imp = data_dictionary['normalization data'][0]
    data_table = data_dictionary['final data table']
    pre_imp:pd.DataFrame = pd.read_json(pre_imp, orient='split')
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.imputation_histogram(pre_imp,data_table)]

@callback(
    Output('proteomics-distribution-figure', 'children'),
    Input('proteomics-imputation-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    State('rep-colors', 'data'),
    
    
)
def proteomics_distribution_figure(_, data_dictionary,rep_colors) -> list:
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.distribution_figure(
                data_table, 
                rep_colors, 
                data_dictionary['rev sample groups'], 
                log2_transform=False
                )]

@callback(
    Output('proteomics-cv-figure', 'children'),
    Input('proteomics-distribution-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    
)
def proteomics_cv_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.coefficient_of_variation_plot(data_table,title='%CV')]

@callback(
    Output('proteomics-pca-figure', 'children'),
    Input('proteomics-cv-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    
    
)
def proteomics_pca_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.pca_plot(data_table,data_dictionary['rev sample groups'])]

@callback(
    Output('proteomics-tsne-figure', 'children'),
    Input('proteomics-pca-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    
    
)
def proteomics_tsne_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.t_sne_plot(data_table,data_dictionary['rev sample groups'])]

@callback(
    Output('proteomics-correlation-clustermap-figure', 'children'),
    Input('proteomics-tsne-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    
    
)
def proteomics_correlation_clustermap_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [
            html.Div('Sample correlation'),
            figure_generation.correlation_clustermap(data_table)
        ]

@callback(
    Output('proteomics-full-clustermap-figure', 'children'),
    Input('proteomics-correlation-clustermap-figure', 'children'),
    State('processed-proteomics-data', 'data'),   
)
def proteomics_full_clustermap_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [
            html.Div('Sample clustering'),
            figure_generation.full_clustermap(data_table)
        ]

@callback(
    Output('proteomics-volcano-plots', 'children'),
    Input('proteomics-full-clustermap-figure', 'children'),
    Input('control-dropdown','value'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_volcano_plots(_, control_group, data_dictionary) -> list:
    if control_group is None: return []
    data_table = data_dictionary['final data table']
    data_table:pd.DataFrame = pd.read_json(data_table, orient='split')
    return [
                html.Div(id='volcano-plot-label', children = 'Volcano plots'),
                html.Div(id='volcano-plot-container', children = figure_generation.volcano_plots(
                    data_table, 
                    data_dictionary['sample groups'], 
                    control_group
                    )
                )
            ]

@callback(
    Output('analysis-tabs', 'children'),
    Input('qc-plot-container','children'),
    State('workflow-choice', 'data'),
    State('analysis-tabs', 'children'),
    State('output-data-upload', 'data'),
    prevent_initial_call=True,
)
def create_workflow_specific_tabs(_,workflow_choice_data, current_tabs, data_dictionary) -> list:
    return_tabs: list = current_tabs
    if workflow_choice_data == 'proteomics':
        return_tabs = [current_tabs[0]]
        if 'table' in data_dictionary:
            return_tabs.append(
                generate_proteomics_tab()
            )
    return return_tabs

@callback(
    Output('control-dropdown','options'),
    Input('processed-proteomics-data', 'data')
)
def set_control_dropdown_values(data_dictionary) -> dict:
    sample_groups: list = ['temp']
    if data_dictionary is not None:
            if 'sample groups' in data_dictionary:
                sample_groups = sorted(list(data_dictionary['sample groups'].keys()))
    return [{'label': sample_group, 'value': sample_group} for sample_group in sample_groups]


def generate_proteomics_tab() -> dbc.Tab:
    proteomics_summary_tab: dbc.Card = dbc.Card(
        dbc.CardBody(
            children=[
                dcc.Store(id='processed-proteomics-data'),
                html.Div(
                    [
                        dbc.Label('Filter:'),
                        dcc.Slider(0, 1, 0.1, value=0.7,
                                   id='filter-minimum-percentage'),
                        dbc.Select(
                            options=[
                                {'label': 'placeholder', 'value': 'placeholder'}
                            ],
                            required=True,
                            id='control-dropdown',
                        ),
                    ]
                ),
                dbc.Label('Imputation:'),
                dbc.RadioItems(
                    options=[
                        {'label': 'QRILC', 'value': 'QRILC'},
                        {'label': 'minProb', 'value': 'minProb'},
                        {'label': 'minValue', 'value': 'minValue'},
                        {'label': 'No imputation', 'value': None},
                    ],
                    value='QRILC',
                    id='imputation-radio-option'
                ),
                dbc.Label('Normalization:'),
                dbc.RadioItems(
                    options=[
                        {'label': 'Median', 'value': 'Median'},
                        {'label': 'Quantile', 'value': 'Quantile'},
                        {'label': 'No normalization', 'value': None}
                    ],
                    value='Median',
                    id='normalization-radio-option'
                ),
                html.Div(
                    id='data-processing-figures',
                )
            ],
            id='proteomics-summary-tab-contents'
        ),
        className='mt-3'
    )
    return dbc.Tab(proteomics_summary_tab, label='Proteomics', id='proteomics-tab')


upload_row_1: list = [
    dbc.Col(
        html.Div('Workflow:'),
    ),
    dbc.Col(
        html.Div('Figure Theme:'),
    ),
    dbc.Col(
        dbc.Button('Download sample_table template',
                   id='button-download-sample_table-template'),
    )
]
upload_row_2: list = [
    dbc.Col(
        dbc.Select(
            value=db.default_workflow,
            options=[
                {'label': item, 'value': item} for item in db.implemented_workflows
            ],
            id='workflow-dropdown',
        ),
    ),
    dbc.Col(
        dbc.Select(
            value=figure_templates[2],
            options=[
                {'label': item, 'value': item} for item in figure_templates
            ],
            id='figure-template-dropdown',
        ),
    ),
    dbc.Col(
        dbc.Button(
            'Download Datafile example',
            id='button-download-datafile-example'
        ),
    )
]
upload_row_3: list = [
    dbc.Col(
        dcc.Upload(
            id='upload-data-file',
            children=html.Div([
                'Drag and drop or ',
                html.A(
                    'select',
                    style=styles.upload_a_style
                ),
                ' Data file'
            ]
            ),
            style=styles.upload_style,
            multiple=False
        )
    ),
    dbc.Col(
        dcc.Upload(
            id='upload-sample_table-file',
            children=html.Div([
                'Drag and drop or ',
                html.A(
                    'select',
                    style=styles.upload_a_style
                ),
                ' Sample table file'
            ]),
            style=styles.upload_style,
            multiple=False
        )
    ),
]
upload_tab: dbc.Card = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    dbc.Row(
                        upload_row_1
                    ),
                    dbc.Row(
                        upload_row_2
                    ),
                    dbc.Row(
                        upload_row_3
                    ),
                    dbc.Row(
                        dbc.Col(
                            html.Div(id='output-data-upload-problems')
                        )
                    ),
                ]
            ),
            dcc.Download(id='download-sample_table-template'),
            dcc.Download(id='download-datafile-example'),
            html.Div(id='placeholder'),
            dcc.Store(id='output-data-upload'),
            dcc.Store(id='figure-template-choice'),
            dcc.Store(id='workflow-choice'),
            dcc.Store(id='rep-colors'),
            dcc.Loading(
                id='qc-loading',
                children = [
                    html.Div(id='qc-plot-container',
                    children = [
                        html.Br(),
                        html.Br(),
                        html.Br()
                    ])
                ],
                type='default'
            ),
        ],
        className="mt-3"
    ),
)


tabs = dbc.Tabs(
    id='analysis-tabs',
    children=[
        dbc.Tab(upload_tab, label='Data upload and quick QC'),
    ]
)
layout = html.Div([
    dbc.Button('Save session as project',
               id='session-save-button', color='success'),
    tabs
])
