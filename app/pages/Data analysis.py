"""Dash app for data upload"""

import base64
import io
import json
import os
from typing import Any
import dash
import dash_bootstrap_components as dbc
import data_functions
import tooltips
import pandas as pd
import plotly.io as pio
from dash import callback, dcc, html
from dash.dependencies import Input, Output, State
from DbEngine import DbEngine
from FigureGeneration import FigureGeneration
from styles import Styles
from utilitykit import plotting
import shutil

import text_functions


# db will house all data, keep track of next row ID, and validate any new data
db: DbEngine = DbEngine()
figure_generation: FigureGeneration = FigureGeneration()

#app: Dash = Dash(__name__)
#server: app.server = app.server
dash.register_page(__name__, path='/')
styles: Styles = Styles()
figure_templates: list = [
    'plotly',
    'plotly_white',
    'plotly_dark',
    'ggplot2',
    'seaborn',
    'simple_white'
]
# dash.register_page(__name__)


def read_df_from_content(content, filename) -> pd.DataFrame:
    _: str
    content_string: str
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
    for sample_name in data_df.index.values:
        color_column.append(colors[column_to_replicate[sample_name]])
    data_df.loc[:, 'Color'] = color_column


@callback(
    Output('workflow-choice', 'data'),
    Input('workflow-dropdown', 'value'),
    Input('upload-data-file', 'contents'),
    Input('upload-sample_table-file', 'contents'),
    State('session-uid', 'children')
)
def set_workflow(workflow_setting_value, _, __, session_uid) -> str:
    if workflow_setting_value is None:
        return dash.no_update
    with open(db.get_cache_file(session_uid,'workflow-choice.txt'), 'w', encoding='utf-8') as fil:
        fil.write(workflow_setting_value)
    return str(workflow_setting_value)


@callback([
    Output('output-data-upload', 'data'),
    Output('output-data-upload-problems', 'children'),
    Output('figure-template-choice', 'data'),
    Output('placeholder', 'children')
],
    Input('figure-theme-dropdown', 'value'),
    Input('upload-data-file', 'contents'),
    State('upload-data-file', 'filename'),
    Input('upload-sample_table-file', 'contents'),
    State('upload-sample_table-file', 'filename'),
    State('session-uid', 'children'),
)
def process_input_tables(
    figure_template_dropdown_value,
    data_file_contents,
    data_file_name,
    sample_table_file_contents,
    sample_table_file_name,
    session_uid,
) -> tuple:
    if figure_template_dropdown_value:
        with open(db.get_cache_file(session_uid, 'figure-template-choice.txt'), 'w', encoding='utf-8') as fil:
            fil.write(figure_template_dropdown_value)
        pio.templates.default = figure_template_dropdown_value
    return_message: list = []
    return_dict: dict = {}
    if data_file_contents is None:
        return_message.append('Missing data table')
    if sample_table_file_contents is None:
        return_message.append('Missing sample_table')
    if len(return_message) == 0:
        intensity_table: pd.DataFrame
        sample_groups: dict
        rev_sample_groups: dict
        spc_table: pd.DataFrame
        data_type: tuple
        raw_intensity_table: pd.DataFrame
        protein_lengths: dict
        discarded_columns: list
        intensity_table, sample_groups, rev_sample_groups, spc_table, data_type, raw_intensity_table, protein_lengths, discarded_columns = data_functions.parse_data(
            data_file_contents,
            data_file_name,
            sample_table_file_contents,
            sample_table_file_name,
            max_theoretical_spc=db.max_theoretical_spc
        )
        return_dict['sample groups'] = sample_groups
        return_dict['discarded columns'] = discarded_columns
        return_dict['rev sample groups'] = rev_sample_groups
        return_dict['raw intensity table'] = raw_intensity_table.to_json(
            orient='split')
        return_dict['spc table'] = spc_table.to_json(orient='split')
        return_dict['intensity table'] = intensity_table.to_json(
            orient='split')
        return_dict['data type'] = data_type
        return_dict['protein lengths'] = protein_lengths
        return_dict['guessed control samples'] = data_functions.guess_controls(
            sample_groups)
        if len(intensity_table.columns) < 2:
            return_dict['table'] = return_dict['spc table']
            return_dict['values'] = 'SPC'
        else:
            return_dict['table'] = return_dict['intensity table']
            return_dict['values'] = 'intensity'
        with open(db.get_cache_file(session_uid, 'data_dict.json'), 'w', encoding='utf-8') as fil:
            json.dump(return_dict, fil)
        with open(db.get_cache_file(session_uid, 'sample groups.json'),'w', encoding='utf-8') as fil:
            json.dump(sample_groups, fil)
        with open(db.get_cache_file(session_uid, 'rev sample groups.json'),'w', encoding='utf-8') as fil:
            json.dump(rev_sample_groups, fil)
        with open(db.get_cache_file(session_uid, 'protein lengths.json'),'w', encoding='utf-8') as fil:
            json.dump(protein_lengths, fil)
        with open(db.get_cache_file(session_uid, 'info.txt'),'w', encoding='utf-8') as fil:
            fil.write(f'Data type: {return_dict["values"]}')
            discarded_str:str = '\t'.join(discarded_columns)
            fil.write(f'Discarded columns:\n{discarded_str}')
            fil.write(f'Data type: {data_type}')
        spc_table.to_csv(db.get_cache_file(session_uid, 'SPC table.tsv'),sep='\t')
        intensity_table.to_csv(db.get_cache_file(session_uid, 'Intensity table.tsv'),sep='\t')
        raw_intensity_table.to_csv(db.get_cache_file(session_uid, 'Raw intensity table.tsv'),sep='\t')
        return_message = f'Succesful Upload! Data file: {data_file_name}  Sample table file: {sample_table_file_name}'
    else:
        return_message = ' ; '.join(return_message)
    return return_dict, return_message, figure_template_dropdown_value, ''


@callback(
    Output('void','children'),
    Input('figures-to-save','data'),
    State('session-uid', 'children')
)
def save_figures(save_data, session_uid) -> None:
    figure_generation.save_figures(
        save_data,
        os.path.join(db.get_cache_dir(session_uid), 'Figures')
    )


@callback(
    Output('qc-plot-container', 'children'),
    Output('rep-colors', 'data'),
    Output('figures-to-save','data'),
    Input('placeholder', 'children'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children')
)
def quality_control_charts(_, data_dictionary,session_uid) -> list:
    figures: list = []
    to_save: list = []
    figure_names_and_legends: list = []
    rep_colors = {}
    if data_dictionary is not None:
        if 'table' in data_dictionary:
            data_table: pd.DataFrame = pd.read_json(
                data_dictionary['table'], orient='split')
            raw_intensity_table: pd.DataFrame = pd.read_json(
                data_dictionary['raw intensity table'], orient='split'
            )
            count_data: pd.DataFrame = data_functions.get_count_data(
                data_table)
            add_replicate_colors(
                count_data, data_dictionary['rev sample groups'])
            rep_colors: dict = {}
            for sname, sample_color_row in count_data[['Color']].iterrows():
                rep_colors[sname] = sample_color_row['Color']
            data_dictionary['Replicate colors'] = rep_colors
            figures.append(figure_generation.protein_count_figure(count_data))
            figure_names_and_legends.append(['Protein count in samples',''])
            figures.append(figure_generation.contaminant_figure(data_table, db.contaminant_list))
            figure_names_and_legends.append(['Contaminants in samples',''])
            figures.append(figure_generation.protein_coverage(data_table))
            figure_names_and_legends.append(['Protein coverage',''])

            figures.append(figure_generation.reproducibility_figure(
                                data_table,
                                data_dictionary['sample groups']
                            )
                        )
            figure_names_and_legends.append(['Reproducibility',''])
            na_data: pd.DataFrame = data_functions.get_na_data(data_table)
            na_data['Color'] = [rep_colors[sample_name]
                                for sample_name in na_data.index.values]
            figures.append(figure_generation.missing_figure(na_data))
            figure_names_and_legends.append(['Missing values in samples',''])
            figures.append(figure_generation.missing_clustermap(data_table))
            figure_names_and_legends.append(['Missing value clustermap',''])
            sumdata: pd.DataFrame = data_functions.get_sum_data(
                data_table)
            sumdata['Color'] = [rep_colors[sample_name]
                                for sample_name in sumdata.index.values]
            figures.append(figure_generation.sum_value_figure(sumdata))
            figure_names_and_legends.append(['Sum of values in samples',''])

            avgdata: pd.DataFrame = data_functions.get_avg_data(
                data_table)
            avgdata['Color'] = [rep_colors[sample_name]
                                for sample_name in avgdata.index.values]
            figures.append(figure_generation.avg_value_figure(avgdata))
            figure_names_and_legends.append(['Sample averages',''])
            dist_title: str = 'Value distribution'
            if data_dictionary['values'] == 'intensity':
                figures.append(
                    figure_generation.distribution_figure(
                        raw_intensity_table,
                        rep_colors,
                        data_dictionary['rev sample groups'],
                        title='Raw value distribution'
                    )
                )
                figure_names_and_legends.append(['Raw value distribution',''])
                dist_title = 'Log2 transformed value distribution'
            figures.append(
                figure_generation.distribution_figure(
                    data_table,
                    rep_colors,
                    data_dictionary['rev sample groups'],
                    title=dist_title
                )
            )
            figure_names_and_legends.append(['Processed value distribution', 'This plot describes the value distribution in each of the samples after possible log transformation (used for intensity data)'])

            figure_dir:str = os.path.join(db.get_cache_dir(session_uid), 'Figures')
            figure_data_dir: str = os.path.join(figure_dir, 'data')
            if not os.path.isdir(figure_data_dir):
                os.makedirs(figure_data_dir)
            figures.append(
                figure_generation.supervenn(
                    data_table,
                    data_dictionary['rev sample groups'],
                    save_figure = os.path.join(figure_dir, 'Supervenn'),
                    save_format = 'pdf'
                )
            )
            count_data.to_csv(os.path.join(figure_data_dir, 'Count data.tsv'),sep='\t')
            sumdata.to_csv(os.path.join(figure_data_dir, 'Sum data.tsv'),sep='\t')
            na_data.to_csv(os.path.join(figure_data_dir, 'NA data.tsv'),sep='\t')
            avgdata.to_csv(os.path.join(figure_data_dir, 'AVG data.tsv'),sep='\t')
            with open(os.path.join(figure_data_dir, 'rep colors.json'),'w',encoding='utf-8') as fil:
                json.dump(rep_colors,fil)
            
            with open(os.path.join(figure_data_dir, 'rep colors.json'),'w',encoding='utf-8') as fil:
                json.dump(rep_colors,fil)

            # Long-term:
            # - protein counts compared to previous similar samples
            # - sum value compared to previous similar samples
            # - Person-to-person comparisons: protein counts, intensity/psm totals
            to_save = [
                f[0] for f in figures if f[0] is not None
                ]
            figures = [f[1] for f in figures]
        return (figures, rep_colors, [to_save, figure_names_and_legends])
    else:
        return (dash.no_update, dash.no_update, dash.no_update)


@callback(
    Output('download-sample_table-template', 'data'),
    Input('button-download-sample_table-template', 'n_clicks'),
    prevent_initial_call=True,
)
def sample_table_example_download(_) -> dict:
    return dcc.send_file(db.request_file('example data', 'example-sample_table'))


@callback(
    Output('download-datafile-example', 'data'),
    Input('button-download-datafile-example', 'n_clicks'),
    prevent_initial_call=True,
)
def download_data_table_example(_) -> dict:
    return dcc.send_file(db.request_file('example data', 'example-data_file'))


@callback(
    Output('download-all-data', 'data'),
    Input('button-export-all-data', 'n_clicks'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children'),
    prevent_initial_call=True
)
def download_data_table(_, data_dictionary,session_uid) -> dict:
    export_csvs: list = [
        'spc table',
        'intensity table',
        'raw intensity table'
    ]
    export_jsons: list = [
        'sample groups',
        'rev sample groups'
    ]
    export_fileinfo: list = [
        f'Values: {data_dictionary["values"]}',
        f'Data type: {data_dictionary["data type"]}',
        'Guessed controls:'
    ]
    export_fileinfo.extend(data_dictionary['guessed control samples'][0])

    export_dir: str = db.get_cache_dir(session_uid)
    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)
    for key in export_csvs:
        pd.read_json(data_dictionary[key], orient='split').to_csv(
            os.path.join(export_dir, key + '.tsv'), sep='\t')
    for jsonkey in export_jsons:
        with open(os.path.join(export_dir, key + '.json'), 'w',encoding='utf-8') as fil:
            json.dump(data_dictionary[jsonkey], fil)
    with open(os.path.join(export_dir, 'Info.txt'), 'w',encoding='utf-8') as fil:
        fil.write('\n'.join(export_fileinfo))
    shutil.make_archive(export_dir.rstrip(os.sep), 'zip', export_dir)
    return dcc.send_file(export_dir.rstrip(os.sep) + '.zip')


@callback(
    Output('data-processing-figures', 'children'),
    Output('processed-proteomics-data', 'data'),
    Input('filter-minimum-percentage', 'value'),
    Input('imputation-radio-option', 'value'),
    Input('normalization-radio-option', 'value'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children')
)
def make_proteomics_data_processing_figures(filter_threshold, imputation_method, normalization_method, data_dictionary, session_uid) -> list:
    if data_dictionary is not None:
        if 'table' in data_dictionary:
            if data_dictionary['values'] == 'SPC':
                return ['No intensity data in input, cannot generate figures.', data_dictionary]
            # define the data:
            data_table: pd.DataFrame = pd.read_json(
                data_dictionary['table'], orient='split')
            sample_groups: dict = data_dictionary['sample groups']
            rev_sample_groups: dict = data_dictionary['rev sample groups']

            # Filter by missing value proportion
            original_counts: pd.Series = data_functions.count_per_sample(
                data_table, rev_sample_groups)
            original_counts.to_csv(db.get_cache_file(session_uid, 'Original counts.tsv'),sep='\t')
            data_table = data_functions.filter_missing(
                data_table, sample_groups, threshold=filter_threshold)
            data_table = data_table.loc[~data_table.index.isin(db.contaminant_list)]
            data_table.to_csv(db.get_cache_file(session_uid, 'NA and contaminant filtereddata table.tsv'),sep='\t')
            filtered_counts: pd.Series = data_functions.count_per_sample(
                data_table, rev_sample_groups)
            filtered_counts.to_csv(db.get_cache_file(session_uid, 'Filtered counts.tsv'),sep='\t')
            data_dictionary['filter data'] = [original_counts.to_json(), filtered_counts.to_json()]

            data_dictionary['normalization data'] = [
                data_table.to_json(orient='split')]
            # Normalization, if needed:
            if normalization_method:
                data_table = data_functions.normalize(
                    data_table, normalization_method)
                data_table.to_csv(db.get_cache_file(session_uid, 'NA Filtered and normalized data table.tsv'),sep='\t')
                data_dictionary['normalization data'].append(
                    data_table.to_json(orient='split'))
            data_table = data_functions.impute(
                data_table, method=imputation_method, tempdir=db.temp_dir)
            data_table.to_csv(db.get_cache_file(session_uid, 'NA filtered, normalized and imputed data table.tsv'),sep='\t')
            data_dictionary['final data table'] = data_table.to_json(
                orient='split')
            if data_dictionary['protein lengths'] is None:
                data_dictionary['protein lengths'] = db.get_protein_lengths(
                    data_table.index)
            
            with open(db.get_cache_file(session_uid, 'protein lengths.json'),'w',encoding='utf-8') as fil:
                json.dump(data_dictionary['protein lengths'] ,fil)

            return [
                [
                    dcc.Loading(type='circle',
                                id='proteomics-filtering-figure'),
                    dcc.Loading(type='circle',
                                id='proteomics-normalization-figure'),
                    dcc.Loading(type='circle',
                                id='proteomics-imputation-figure'),
                    dcc.Loading(type='circle',
                                id='proteomics-distribution-figure'),
                    dcc.Loading(type='circle', id='proteomics-cv-figure'),
                    dcc.Loading(type='circle', id='proteomics-pca-figure'),
                    dcc.Loading(type='circle',
                                id='proteomics-correlation-clustermap-figure'),
                    dcc.Loading(type='circle',
                                id='proteomics-full-clustermap-figure'),
                    dcc.Loading(type='circle', id='proteomics-volcano-plots'),
                ],
                data_dictionary
            ]
    return dash.no_update


@callback(
    Output('proteomics-filtering-figure', 'children'),
    Input('data-processing-figures', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_filter_figure(figure_loadings, data_dictionary) -> list:
    if len(figure_loadings) < 3:
        return dash.no_update
    original_counts: pd.Series
    filtered_counts: pd.Series
    original_counts, filtered_counts = data_dictionary['filter data']
    return [
        figure_generation.before_after_plot(
            pd.read_json(original_counts, typ='series'),
            pd.read_json(filtered_counts, typ='series'),
            title='NA Filtering'
        )
    ]


@callback(
    Output('proteomics-normalization-figure', 'children'),
    Input('proteomics-filtering-figure', 'children'),
    State('processed-proteomics-data', 'data'),


)
def proteomics_normalization_figure(_, data_dictionary,) -> list:
    if len(data_dictionary['normalization data']) < 2:
        return ''
    pre_norm, data_table = data_dictionary['normalization data']
    pre_norm: pd.DataFrame = pd.read_json(pre_norm, orient='split')
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
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
    pre_imp: pd.DataFrame = pd.read_json(pre_imp, orient='split')
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.imputation_histogram(pre_imp, data_table)]


@callback(
    Output('proteomics-distribution-figure', 'children'),
    Input('proteomics-imputation-figure', 'children'),
    State('processed-proteomics-data', 'data'),
    State('rep-colors', 'data'),
)
def proteomics_distribution_figure(_, data_dictionary, rep_colors) -> list:
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.distribution_figure(
        data_table,
        rep_colors,
        data_dictionary['rev sample groups']
    )]


@callback(
    Output('proteomics-cv-figure', 'children'),
    Input('proteomics-distribution-figure', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_cv_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.coefficient_of_variation_plot(data_table, title='%CV')]


@callback(
    Output('proteomics-pca-figure', 'children'),
    Input('proteomics-cv-figure', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_pca_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.pca_plot(data_table, data_dictionary['rev sample groups'])]


@callback(
    Output('proteomics-correlation-clustermap-figure', 'children'),
    Input('proteomics-pca-figure', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_correlation_clustermap_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
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
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [
        html.Div('Sample clustering'),
        figure_generation.full_clustermap(data_table)
    ]


@callback(
    Output('proteomics-volcano-plots', 'children'),
    Input('proteomics-full-clustermap-figure', 'children'),
    Input('control-dropdown', 'value'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_volcano_plots(_, control_group, data_dictionary) -> list:
    if control_group is None:
        return dash.no_update
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [
        html.Div(id='volcano-plot-label', children='Volcano plots'),
        html.Div(id='volcano-plot-container', children=figure_generation.volcano_plots(
                    data_table,
                    data_dictionary['sample groups'],
                    control_group
        )
        )
    ]


@callback(
    Output('analysis-tabs', 'children'),
    Input('qc-plot-container', 'children'),
    Input('workflow-choice', 'data'),
    State('analysis-tabs', 'children'),
    State('output-data-upload', 'data'),
)
def create_workflow_specific_tabs(_, workflow_choice_data, current_tabs, data_dictionary) -> list:
    return_tabs: list = current_tabs
    if workflow_choice_data == 'proteomics':
        return_tabs = [current_tabs[0]]
        if data_dictionary is not None:
            if 'table' in data_dictionary:
                return_tabs.append(
                    generate_proteomics_tab()
                )
    elif workflow_choice_data == 'interactomics':
        return_tabs = [current_tabs[0]]
        if data_dictionary is not None:
            if 'table' in data_dictionary:
                return_tabs.append(
                    generate_interactomics_tab(
                        data_dictionary['sample groups'], data_dictionary['guessed control samples'])
                )
    if return_tabs == current_tabs:
        return dash.no_update
    else:
        return return_tabs


@callback(
    Output('control-dropdown', 'options'),
    Input('processed-proteomics-data', 'data')
)
def set_volcano_plot_control_dropdown_values(data_dictionary) -> dict:
    sample_groups: list = ['temp']
    if data_dictionary is not None:
        if 'sample groups' in data_dictionary:
            sample_groups = sorted(
                list(data_dictionary['sample groups'].keys()))
    return [{'label': sample_group, 'value': sample_group} for sample_group in sample_groups]


def generate_proteomics_tab() -> dbc.Tab:
    proteomics_tab: dbc.Card = dbc.Card(
        dbc.CardBody(
            children=[
                dcc.Store(id='processed-proteomics-data'),
                html.Div(
                    [

                        html.Div([
                            dbc.Label('NA Filtering:', id='filtering-label'),
                            tooltips.na_tooltip()
                        ]),
                        dcc.Slider(0, 100, 10, value=70,
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
                        {'label': i_opt, 'value': i_opt_val}
                            for i_opt, i_opt_val in db.imputation_options.items()
                    ],
                    value=db.default_imputation_method,
                    id='imputation-radio-option'
                ),
                dbc.Label('Normalization:'),
                dbc.RadioItems(
                    options=[
                        {'label': n_opt, 'value': n_opt_val}
                            for n_opt, n_opt_val in db.normalization_options.items()
                    ],
                    value=db.default_normalization_method,
                    id='normalization-radio-option'
                ),
                html.Hr(),
                html.Div(
                    id='data-processing-figures',
                )
            ],
            id='proteomics-summary-tab-contents'
        ),
        className='mt-3'
    )
    return dbc.Tab(proteomics_tab, label='Proteomics', id='proteomics-tab')


def checklist(label: str, options: list, default_choice: list, disabled: list = None, id_prefix: str = None, simple_text_clean: bool = False) -> dbc.Checklist:
    if disabled is None:
        disabled: set = set()
    else:
        disabled: set = set(disabled)
    checklist_id: str
    if simple_text_clean:
        checklist_id = f'{id_prefix}-{label.strip(":").strip().replace(" ","-").lower()}'
    else:
        checklist_id = f'{id_prefix}-{text_functions.clean_text(label.lower())}'
    return [
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


@callback(
    Output('interactomics-saint-container', 'children'),
    Output('raw-interactomics-data', 'data'),
    Output('crapome-column-groups', 'data'),
    Input('button-run-saint-analysis', 'n_clicks'),
    State('interactomics-choose-additional-control-sets', 'value'),
    State('interactomics-choose-crapome-sets', 'value'),
    State('interactomics-choose-uploaded-controls', 'value'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children'),
    prevent_initial_call=True
)
def generate_saint_container(_, inbuilt_controls, crapome_controls, control_sample_groups, data_dictionary, session_uid) -> html.Div:
    spc_table: pd.DataFrame = pd.read_json(
        data_dictionary['spc table'], orient='split')
    if spc_table.columns[0] == 'No data':
        return html.Div(['No spectral count data in input, cannot run SAINT.'])
    inbuilt_control_table: pd.DataFrame = db.controls(inbuilt_controls)
    crapome_table: pd.DataFrame
    crapome_column_groups: list
    crapome_table, crapome_column_groups = db.crapome(
        crapome_controls, list(spc_table.index))


# check control samples from container -> add to control groups list
# get chosen GFP sets as control_table
# get project output directory from somewhere at some point

    saint_output: pd.DataFrame
    discarded_proteins: list
    saint_output, discarded_proteins = data_functions.run_saint(
        spc_table,
        data_dictionary['rev sample groups'],
        data_dictionary['protein lengths'],
        db.scripts('SAINTexpress'),
        control_table=inbuilt_control_table,
        control_groups=control_sample_groups,
    )
    saint_output.loc['Is contaminant'] = saint_output['Prey'].isin(db.contaminant_list)
    saint_output = pd.merge(
        left=saint_output,
        right=crapome_table,
        how='left',
        left_on='Prey',
        right_index=True)

    discarded_proteins = sorted(
        list((set(discarded_proteins) & set(spc_table.index))))

    new_crapome_column_groups: list = []
    drop_cols: list = []
    for cr_avgspec, cr_freq in crapome_column_groups:
        cr_group: str = cr_avgspec.replace(' AvgSpec', '')
        cr_fccol: str = f'{cr_group} FC'
        saint_output[cr_fccol] = saint_output['AvgSpec'] / \
            saint_output[cr_avgspec].fillna(0)
        drop_cols.append(cr_avgspec)
        new_crapome_column_groups.append([cr_fccol, cr_freq])
    crapome_column_groups = new_crapome_column_groups
    saint_output = saint_output.drop(columns=drop_cols)

    container_contents: list = []
    if len(discarded_proteins) > 0:
        discarded_specsum: pd.Series = spc_table.loc[discarded_proteins].sum()
        container_contents.append(
            html.Div([
                f'Proteins without length discarded from SAINT analysis: {", ".join(discarded_proteins)}',
                html.Br(),
                f'With combined spectral count sum of {discarded_specsum.sum()}',
                html.Br(),
                f'Found in samples: {", ".join(sorted(list(discarded_specsum.index.values)))}'
            ])
        )

    before_contaminants: int = saint_output.shape[0]
    removed_contaminants: list = list(saint_output[saint_output['Is contaminant']]['Prey'].values)
    saint_output = saint_output[~saint_output['Is contaminant']]
    after_contaminants: int = saint_output.shape[0]
    if after_contaminants == before_contaminants:
        saint_output = saint_output.drop(columns=['Is contaminant'])
    else:
        container_contents.append(
            html.Div([
                f'Removed {before_contaminants-after_contaminants} common contaminants from the output dataset.',
                html.Br(),
                f'Removed contaminants: {", ".join(removed_contaminants)}'
            ])
        )

    container_contents.append(
        html.Div([
            figure_generation.histogram(
                saint_output, x_column='BFDR', title='Saint BFDR distribution'),
            dcc.Graph(id='interactomics-saint-graph'),
            dbc.Label('Saint BFDR threshold:'),
            dcc.Slider(0, 0.1, 0.01, value=0.05,
                       id='saint-bfdr-filter-threshold'),
            dbc.Label('Crapome filtering:'),
            dbc.Label('Crapome frequency'),
            dcc.Slider(1, 100, 10, value=20,
                       id='crapome-frequency-threshold'),
            dbc.Label('SPC fold change vs crapome threshold for rescue'),
            dcc.Slider(0, 10, 1, value=3,
                       id='crapome-rescue-threshold'),
        ])
    )
    

    with open(db.get_cache_file(session_uid, 'SAINT info.txt'),'w',encoding='utf-8') as fil:
        discarded_str: str = '\t'.join(discarded_proteins)
        fil.write(f'Discarded proteins:\n{discarded_str}')
    saint_output.to_csv(db.get_cache_file(session_uid, 'Saint output.tsv'),sep='\t')
    inbuilt_control_table.to_csv(db.get_cache_file(session_uid, 'Inbuilt controls.tsv'),sep='\t')
    crapome_table.to_csv(db.get_cache_file(session_uid, 'Crapome table.tsv'),sep='\t')
    return container_contents, saint_output.to_json(orient='split'), crapome_column_groups


@callback(
    [
        Output('interactomics-saint-graph', 'figure'),
        Output('processed-interactomics-data', 'data')
    ],
    Input('saint-bfdr-filter-threshold', 'value'),
    Input('crapome-frequency-threshold', 'value'),
    Input('crapome-rescue-threshold', 'value'),
    State('raw-interactomics-data', 'data'),
    State('crapome-column-groups', 'data'),
    State('session-uid', 'children')
)
def filter_saint_table_and_update_graph(bfdr_threshold, crapome_freq, crapome_fc, raw_data, crapome_column_groups, session_uid) -> tuple:
    filtered_saint: pd.DataFrame = pd.read_json(raw_data, orient='split')
    filtered_saint = filtered_saint[filtered_saint['BFDR'] < bfdr_threshold]

    for crapome_fc_column, crapome_freq_column in crapome_column_groups:
        filtered_saint = filtered_saint[
            (filtered_saint[crapome_freq_column] < crapome_freq) |
            (filtered_saint[crapome_fc_column] > crapome_fc)
        ]

    bar_plot_df: pd.DataFrame = pd.DataFrame(
        filtered_saint.value_counts(subset=['Bait']), columns=['Prey count']
    ).reset_index()
    figure: Any = figure_generation.bar_plot(
        bar_plot_df,
        'Protein counts after filtering',
        y_name='Prey count',
        color_col='Bait'
    )
    filtered_saint.to_csv(db.get_cache_file(session_uid, 'Saint output filtered.tsv'),sep='\t')
    return (figure, filtered_saint.to_json(orient='split'))


def generate_interactomics_tab(sample_groups: dict, guessed_controls: tuple) -> dbc.Tab:
    all_sample_groups: list = []
    chosen: list = guessed_controls[0]
    for k in sample_groups.keys():
        if k not in chosen:
            all_sample_groups.append(k)
    all_sample_groups = sorted(chosen) + sorted(all_sample_groups)
    interactomics_tab: dbc.Card = dbc.Card(
        dbc.CardBody(
            children=[
                dcc.Store(id='processed-interactomics-data'),
                dcc.Store(id='raw-interactomics-data'),
                dcc.Store(id='crapome-column-groups'),
                html.Div(
                    id='interactomics-options',
                    children=[
                        html.Div(
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            checklist(
                                                'Choose additional control sets:',
                                                db.controlsets,
                                                db.default_controlsets,
                                                disabled=db.disabled_controlsets,
                                                id_prefix='interactomics',
                                                simple_text_clean=True
                                            )
                                        ),
                                        dbc.Col(
                                            checklist(
                                                'Choose Crapome sets:',
                                                db.crapomesets,
                                                db.default_crapomesets,
                                                disabled=db.disabled_crapomesets,
                                                id_prefix='interactomics',
                                                simple_text_clean=True
                                            )
                                        ),
                                        dbc.Col(
                                            checklist(
                                                'Choose uploaded controls:',
                                                all_sample_groups,
                                                chosen,
                                                id_prefix='interactomics',
                                                simple_text_clean=True
                                            )
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
                        ),
                        html.Div(id='interactomics-saint-container'),
                        html.Div(id='interactomics-crapome-container'),
                        html.Div(id='interactomics-totals-continer'),
                        html.Div(id='interactomics-overall-container'),
                        html.Div(id='interactomics-pca-container'),
                        html.Div(id='interactomics-network-container'),
                    ]
                )
            ]
        )
    )
    return dbc.Tab(interactomics_tab, label='Interactomics', id='interactomics-tab')


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
            # value=db.default_workflow,
            options=[
                {'label': item, 'value': item} for item in db.implemented_workflows
            ],
            id='workflow-dropdown',
        ),
    ),
    dbc.Col(
        dbc.Select(
            value=figure_templates[0],
            options=[
                {'label': item, 'value': item} for item in figure_templates
            ],
            id='figure-theme-dropdown',
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
            dbc.Button(
                'Export all data',
                id='button-export-all-data'
            ),
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
            dcc.Download(id='download-all-data'),
            html.Div(id='placeholder'),
            html.Hr(),
            dcc.Store(id='output-data-upload'),
            dcc.Store(id='figure-template-choice'),
            dcc.Store(id='workflow-choice'),
            dcc.Store(id='rep-colors'),
            dcc.Loading(
                id='qc-loading',
                children=[
                    html.Div(id='qc-plot-container',
                             children=[
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
    # dbc.Button('Save session as project',
    #            id='session-save-button', color='success'),
    html.Div(id='void',hidden=True),
    dcc.Store(id='figures-to-save'),
    tabs
])
