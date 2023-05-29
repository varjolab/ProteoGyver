"""Dash app for data upload"""

__version__:str = '0.1a3'

import base64
import io
import json
import os
from typing import Any
import dash
from matplotlib.pyplot import isinteractive
import dash_bootstrap_components as dbc
from DataFunctions import DataFunctions
import plotly.graph_objects as go
import tooltips
import pandas as pd
import plotly.io as pio
from dash import callback, dcc, html, ALL, MATCH
from dash.dependencies import Input, Output, State
from DbEngine import DbEngine
from FigureGeneration import FigureGeneration
from styles import Styles
import shutil
import dash_cytoscape as cyto
from dash import dash_table
import text_functions
import plotly.io as pio


db: DbEngine = DbEngine()
figure_generation: FigureGeneration = FigureGeneration()
data_functions: DataFunctions = DataFunctions(os.path.join(*db.parameters['data functions']))

dash.register_page(__name__, path='/')
styles: Styles = Styles()
figure_templates: list = [
    'plotly_white',
    'simple_white'
]

figure_template_colors: dict = {}
for t in figure_templates:
    figure_template_colors[t] = pio.templates[t].layout['colorway']


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
    Output('upload-complete-indicator', 'children')
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
        return_dict = data_functions.parse_data(
            data_file_contents,
            data_file_name,
            sample_table_file_contents,
            sample_table_file_name,
            max_theoretical_spc=db.max_theoretical_spc
        )
        #TODO: Move writing these to a subprocess?
        with open(db.get_cache_file(session_uid, 'data_dict.json'), 'w', encoding='utf-8') as fil:
            json.dump(return_dict, fil, indent = 4)
        with open(db.get_cache_file(session_uid, 'sample groups.json'),'w', encoding='utf-8') as fil:
            json.dump(return_dict['sample groups']['norm'], fil, indent = 4)
        with open(db.get_cache_file(session_uid, 'rev sample groups.json'),'w', encoding='utf-8') as fil:
            json.dump(return_dict['sample groups']['rev'], fil, indent = 4)
        with open(db.get_cache_file(session_uid, 'protein lengths.json'),'w', encoding='utf-8') as fil:
            json.dump(return_dict['other']['protein lengths'], fil, indent = 4)
        with open(db.get_cache_file(session_uid, 'info.txt'),'w', encoding='utf-8') as fil:
            fil.write(f'Data type: {return_dict["info"]["values"]}\n')
            fil.write(f'Data type: {return_dict["info"]["data type"]}\n')
            fil.write('Discarded columns:\n')
            for chunk in list_to_chunks(list(return_dict['info']['discarded columns']), 5):
                chunkstr: str = '\t'.join(chunk)
                fil.write(f'{chunkstr}\n')
            fil.write('Used columns:\n')
            for chunk in list_to_chunks(list(return_dict['info']['used columns']), 5):
                chunkstr: str = '\t'.join(chunk)
                fil.write(f'{chunkstr}\n')
        name_dict: dict = {}
        for protein_id in return_dict['other']['all proteins']:
             name_dict[protein_id] =  db.get_name(protein_id)
        return_dict['other']['protein names'] = name_dict
        if return_dict['other']['protein lengths'] is None:
            return_dict['other']['protein lengths'] = db.get_protein_lengths(return_dict['other']['all proteins'])
        for table_name, table_json in return_dict['data tables'].items():
            pd.read_json(table_json,orient='split').to_csv(db.get_cache_file(session_uid, f'{table_name} table.tsv'),sep='\t',encoding = 'utf-8')
        return_message = f'Succesful Upload! Data file: {data_file_name}  Sample table file: {sample_table_file_name}'
    else:
        return_message = ' ; '.join(return_message)
    return return_dict, return_message, figure_template_dropdown_value, ''

@callback(
    Output('interval-component','disabled'),
    Output('tic-traces','data'),
    Output('tic-info','data'),
    Output('auc-traces','data'),
    Input('output-data-upload','data'),
    Input('figure-template-choice', 'data')
)
def generate_tic_graph_data(data_dictionary, figure_template) -> tuple:
    start:bool = True
    if data_dictionary is None:
        start = False
    elif not 'data tables' in data_dictionary:
        start = False
    if not start:
        return [
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update
        ]
    info_df: pd.DataFrame
    tics_found: dict
    info_df, tics_found = db.get_tic_dfs_from_expdesign(
                        pd.read_json(data_dictionary['data tables']['experimental design'], orient='split')
                    )
    num_of_traces_visible: int = 7
    
    max_auc: float = info_df['AUC'].max()
    max_auc += (max_auc*0.15)
    max_tic_int: float = info_df['tic_max_intensity'].max()
    max_tic_int += (max_tic_int*0.15)
    tic_info: dict = {
        'tic_graph_max_x': info_df['max_time'].max(),
        'tic_graph_max_y': max_tic_int,
        'auc_graph_max_x': info_df.shape[0],
        'auc_graph_max_y': max_auc,
        'num_of_traces_visible': num_of_traces_visible
    }

    tic_traces: list = []
    auc_traces: list = []

    color_to_use: str = figure_template_colors[figure_template][0]

    for index, row in info_df.iterrows():
        run_id: str = row['run_id']
        run_label: str = row['run_name']
        ticdf: pd.DataFrame = tics_found[run_id]
        tic_traces.append({})
        for color_i in range(num_of_traces_visible):
            tic_traces[-1][color_i] = go.Scatter(
                x=ticdf['Time'],
                y=ticdf['SumIntensity'],
                mode = 'lines',
                opacity=(1/num_of_traces_visible)*(num_of_traces_visible - color_i),
                line = {'color' : color_to_use, 'width': 1},
                name = run_label,
            )
        auc_up_to_index: pd.DataFrame = info_df[:index]
        auc_traces.append(
            go.Scatter(
                x = list(range(index)),
                y = auc_up_to_index['AUC'],
                mode = 'lines+markers',
                opacity = 1,
                line={'width': 1, 'color': color_to_use},
                showlegend=False
            )
        )
    return False, tic_traces, tic_info, auc_traces


def list_to_chunks(original_list, chunk_size):
    # looping till length l
    for i in range(0, len(original_list), chunk_size):
        yield original_list[i:i + chunk_size]

@callback(
    Output('void','children'),
    Input('figures-to-save','data'),
    State('session-uid', 'children')
)
def save_figures(save_data, session_uid) -> None:
    figure_generation.save_figures(
        os.path.join(db.get_cache_dir(session_uid), 'Figures'),
        save_data
    )

@callback(
    Output('tic-graph', 'children'),
    Output('current-tic-idx','children'),
    Input('interval-component', 'n_intervals'),
    State('tic-traces','data'),
    State('tic-info','data'),
    State('auc-traces','data'),
    State('current-tic-idx','children'),
    State('figure-template-choice', 'data'),
    prevent_initial_call=True
)
def update_tic_graph(_,tic_traces: list, tic_info:dict, auc_traces: list, current_tic_idx: int, figure_template_name:str) -> tuple:
    if tic_info is None:
        return '',current_tic_idx
    if len(auc_traces) == 0:
        return 'No TIC information available.',current_tic_idx
    num_of_traces_visible: int = tic_info['num_of_traces_visible']

    return_tic_index: int = current_tic_idx + 1
    if return_tic_index >= len(tic_traces):
        return_tic_index = 0

    tic_figure: go.Figure = go.Figure()
    these_tics: list
    if current_tic_idx <= num_of_traces_visible:
        these_tics = tic_traces[:current_tic_idx]
    else:
        these_tics: list = tic_traces[current_tic_idx-num_of_traces_visible:current_tic_idx]
    #these_tics: list = tic_traces[:num_of_traces_visible]
    for i, trace_dict in enumerate(these_tics[::-1]):
        tic_figure.add_traces(trace_dict[str(i)])

    tic_figure.update_layout(
        title = 'TIC',
        height=400,
        xaxis_range=[0,tic_info['tic_graph_max_x']],
        yaxis_range=[0,tic_info['tic_graph_max_y']],
        template=figure_template_name,
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        #paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=5, r=5, t=35, b=5)
    )


    auc_figure: go.Figure = go.Figure()
    auc_figure.add_traces(auc_traces[current_tic_idx])
    auc_figure.update_layout(
        title = 'AUC',
        height=150,
        xaxis_range=[0,tic_info['auc_graph_max_x']],
        yaxis_range=[0,tic_info['auc_graph_max_y']],
        template=figure_template_name,
        #plot_bgcolor= 'rgba(0, 0, 0, 0)',
        #paper_bgcolor= 'rgba(0, 0, 0, 0)',
        margin=dict(l=5, r=5, t=35, b=5)
    )

    tic_graph = dcc.Graph(id='tic-graph-with-figure',figure=tic_figure)
    auc_graph = dcc.Graph(id='auc-graph-with-figure',figure=auc_figure)
    tic_div = html.Div([
        dbc.Row([tic_graph]),
        dbc.Row([auc_graph]),
    ])
    return tic_div, return_tic_index

@callback(
    Output('qc-plot-container', 'children'),
    Output('rep-colors', 'data'),
    Output('figures-to-save','data'),
    Input('upload-complete-indicator', 'children'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children')
)
def quality_control_charts(_, data_dictionary,session_uid) -> list:
    figures: list = []
    to_save: list = []
    figure_names_and_legends: list = []
    rep_colors = {}
    if data_dictionary is not None:
        if 'data tables' in data_dictionary:
            data_table: pd.DataFrame = pd.read_json(
                data_dictionary['data tables']['main table'], orient='split')
            count_data: pd.DataFrame = data_functions.get_count_data(
                data_table)
            figure_generation.add_replicate_colors(
                count_data, data_dictionary['sample groups']['rev'])
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
                                data_dictionary['sample groups']['norm']
                            )
                        )
            figure_names_and_legends.append(['Reproducibility',''])
            na_data: pd.DataFrame = data_functions.get_na_data(data_table)
            na_data['Color'] = [rep_colors[sample_name]
                                for sample_name in na_data.index.values]
            figures.append(figure_generation.missing_figure(na_data))
            figure_names_and_legends.append(['Missing values in samples',''])
            # figures.append(figure_generation.missing_clustermap(data_table))
            # figure_names_and_legends.append(['Missing value clustermap',''])
            sumdata: pd.DataFrame = data_functions.get_sum_data(
                data_table)
            sumdata['Color'] = [rep_colors[sample_name]
                                for sample_name in sumdata.index.values]
            figures.append(figure_generation.sum_value_figure(sumdata, valname = data_dictionary['info']['values'].capitalize()))
            figure_names_and_legends.append(['Sum of values in samples',''])

            avgdata: pd.DataFrame = data_functions.get_avg_data(
                data_table)
            avgdata['Color'] = [rep_colors[sample_name]
                                for sample_name in avgdata.index.values]
            figures.append(figure_generation.avg_value_figure(avgdata, valname = data_dictionary['info']['values'].capitalize()))
            figure_names_and_legends.append(['Sample averages',''])
            dist_title: str = 'Value distribution'
            if data_dictionary['info']['values'] == 'intensity':
                # figures.append(
                #     figure_generation.distribution_figure(
                #         raw_intensity_table,
                #         rep_colors,
                #         data_dictionary['sample groups']['rev'],
                #         title='Raw value distribution'
                #     )
                # )
                # figure_names_and_legends.append(['Raw value distribution',''])
                dist_title = 'Log2 transformed value distribution'
            figures.append(
                figure_generation.distribution_figure(
                    data_table,
                    rep_colors,
                    data_dictionary['sample groups']['rev'],
                    title=dist_title
                )
            )
            figure_names_and_legends.append(['Processed value distribution', 'This plot describes the value distribution in each of the samples after possible log transformation (used for intensity data)'])

            figure_dir:str = os.path.join(db.get_cache_dir(session_uid), 'Figures')
            figure_data_dir: str = os.path.join(figure_dir, 'data')
            if not os.path.isdir(figure_data_dir):
                os.makedirs(figure_data_dir)
            figures.append(
                figure_generation.sample_commonality_plot(
                    data_table,
                    data_dictionary['sample groups']['rev'],
                    save_figure = os.path.join(figure_dir, 'Supervenn'),
                    save_format = 'pdf'
                )
            )
            figure_names_and_legends.append(['Shared proteins', 'This plot describes number of shared proteins between different sample groups.'])
            count_data.to_csv(os.path.join(figure_data_dir, 'Count data.tsv'),sep='\t',encoding = 'utf-8')
            sumdata.to_csv(os.path.join(figure_data_dir, 'Sum data.tsv'),sep='\t',encoding = 'utf-8')
            na_data.to_csv(os.path.join(figure_data_dir, 'NA data.tsv'),sep='\t',encoding = 'utf-8')
            avgdata.to_csv(os.path.join(figure_data_dir, 'AVG data.tsv'),sep='\t',encoding = 'utf-8')
            with open(os.path.join(figure_data_dir, 'rep colors.json'),'w',encoding='utf-8') as fil:
                json.dump(rep_colors,fil, indent = 4)
            
            with open(os.path.join(figure_data_dir, 'rep colors.json'),'w',encoding='utf-8') as fil:
                json.dump(rep_colors,fil, indent = 4)

            # Long-term:
            # - protein counts compared to previous similar samples
            # - sum value compared to previous similar samples
            # - Person-to-person comparisons: protein counts, intensity/psm totals
            to_save = [f[0] for f in figures if f[0] is not None]
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
    return dcc.send_file(db.request_file('example files', 'example-sample_table'))


@callback(
    Output('download-datafile-example', 'data'),
    Input('button-download-datafile-example', 'n_clicks'),
    prevent_initial_call=True,
)
def download_data_table_example(_) -> dict:
    return dcc.send_file(db.request_file('example files', 'example-data_file'))


@callback(
    Output('download-all-data', 'data'),
    Input('button-export-all-data', 'n_clicks'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children'),
    prevent_initial_call=True
)
def download_all_data(_, data_dictionary,session_uid) -> dict:
    export_dir: str = os.path.join(db.get_cache_dir(session_uid),'export')
    figure_generation.save_figures(os.path.join(db.get_cache_dir(session_uid),'Figures'), None)
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    copydirs: list = [
        'Enrichments',
        'Figures'
    ]
    for dirname in copydirs:
        dir_to_copy: str = os.path.join(db.get_cache_dir(session_uid), dirname)
        if os.path.isdir(dir_to_copy):
            shutil.copytree(
                os.path.join(dir_to_copy),
                os.path.join(export_dir, dirname)
            )

    export_fileinfo: list = [f'Session UID:{session_uid}']
    for key, value in data_dictionary['info'].items():
        export_fileinfo.append(key)
        if isinstance(value,list) or isinstance(value,set):
            export_fileinfo.append('\n'.join(value))
        else:
            export_fileinfo.append(f'{value}')
        export_fileinfo.append('')
    for key, table in data_dictionary['data tables'].items():
        pd.read_json(table, orient='split').to_excel(os.path.join(export_dir, f'{key}.xlsx'))
    with open(os.path.join(export_dir, 'Info.txt'), 'w',encoding='utf-8') as fil:
        fil.write('\n'.join(export_fileinfo))
    with open(os.path.join(export_dir, 'full data for debugging.json'),'w',encoding='utf-8') as fil:
        json.dump(data_dictionary,fil, indent = 4)
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
        if 'data tables' in data_dictionary:
            if data_dictionary['info']['values'] == 'SPC':
                return ['No intensity data in input, cannot generate figures.', data_dictionary]
            # define the data:
            data_table: pd.DataFrame = pd.read_json(
                data_dictionary['data tables']['main table'], orient='split')
            sample_groups: dict = data_dictionary['sample groups']['norm']
            rev_sample_groups: dict = data_dictionary['sample groups']['rev']

            # Filter by missing value proportion
            original_counts: pd.Series = data_functions.count_per_sample(
                data_table, rev_sample_groups)
            original_counts.to_csv(db.get_cache_file(session_uid, 'Original counts.tsv'),sep='\t',encoding = 'utf-8')
            data_table = data_functions.filter_missing(
                data_table, sample_groups, threshold=filter_threshold)
            data_table = data_table.loc[~data_table.index.isin(db.contaminant_list)]
            data_table.to_csv(db.get_cache_file(session_uid, 'NA and contaminant filtereddata table.tsv'),sep='\t',encoding = 'utf-8')
            filtered_counts: pd.Series = data_functions.count_per_sample(
                data_table, rev_sample_groups)
            filtered_counts.to_csv(db.get_cache_file(session_uid, 'Filtered counts.tsv'),sep='\t',encoding = 'utf-8')
            data_dictionary['filter data'] = [original_counts.to_json(), filtered_counts.to_json()]

            data_dictionary['normalization data'] = [
                data_table.to_json(orient='split')]
            # Normalization, if needed:
            if normalization_method:
                data_table = data_functions.normalize(
                    data_table, normalization_method)
                data_table.to_csv(db.get_cache_file(session_uid, 'NA Filtered and normalized data table.tsv'),sep='\t',encoding = 'utf-8')
                data_dictionary['normalization data'].append(
                    data_table.to_json(orient='split'))
            data_table = data_functions.impute(
                data_table, method=imputation_method, tempdir=db.temp_dir)
            data_table.to_csv(db.get_cache_file(session_uid, 'NA filtered, normalized and imputed data table.tsv'),sep='\t',encoding = 'utf-8')
            data_dictionary['final data table'] = data_table.to_json(
                orient='split')
            
            with open(db.get_cache_file(session_uid, 'protein lengths.json'),'w',encoding='utf-8') as fil:
                json.dump(data_dictionary['other']['protein lengths'] ,fil, indent = 4)

            return (
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
            )
    return (dash.no_update, dash.no_update)


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
    graph: dcc.Graph = figure_generation.before_after_plot(
            pd.read_json(original_counts, typ='series'),
            pd.read_json(filtered_counts, typ='series'),
            title='NA Filtering',
            name_legend = db.figure_data['Proteomics NA filtering figure']
        )[1]
    return [graph]

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
    )[1]]


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
    return [figure_generation.imputation_histogram(pre_imp, data_table)[1]]


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
        data_dictionary['sample groups']['rev']
    )[1]]


@callback(
    Output('proteomics-cv-figure', 'children'),
    Input('proteomics-distribution-figure', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_cv_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.coefficient_of_variation_plot(data_table, title='%CV')[1]]


@callback(
    Output('proteomics-pca-figure', 'children'),
    Input('proteomics-cv-figure', 'children'),
    State('processed-proteomics-data', 'data'),
)
def proteomics_pca_figure(_, data_dictionary) -> list:
    data_table = data_dictionary['final data table']
    data_table: pd.DataFrame = pd.read_json(data_table, orient='split')
    return [figure_generation.pca_plot(data_table, data_dictionary['sample groups']['rev'])[1]]


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
        figure_generation.correlation_clustermap(data_table)[1]
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
        figure_generation.full_clustermap(data_table)[1]
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

    significants: pd.DataFrame
    figures: list
    volcano_graphs: list
    significants, figures, volcano_graphs = figure_generation.volcano_plots(
                    data_table,
                    db.names_for_protein_list(data_table.index),
                    data_dictionary['sample groups']['norm'],
                    control_group
        )
    data_dictionary['data tables'][f'Comparisons vs {control_group}'] = significants.to_json(orient='split')
    return [
        html.Div(id='volcano-plot-label', children='Volcano plots'),
        html.Div(id='volcano-plot-container', children=volcano_graphs
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
            if 'data tables' in data_dictionary:
                return_tabs.append(
                    generate_proteomics_tab()
                )
    elif workflow_choice_data == 'interactomics':
        return_tabs = [current_tabs[0]]
        if data_dictionary is not None:
            if 'data tables' in data_dictionary:
                return_tabs.append(
                    generate_interactomics_tab(
                        data_dictionary['sample groups']['norm'], data_dictionary['sample groups']['guessed control samples'] )
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
                list(data_dictionary['sample groups']['norm'].keys()))
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
                                {'label': 'wait', 'value': 'wait'}
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


def checklist(label: str, options: list, default_choice: list, disabled: list = None, id_prefix: str = None, simple_text_clean: bool = False, id_only:bool=False) -> dbc.Checklist:
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

def map_intensity(saint_output, intensity_table, sample_groups) -> list:
    intensity_column: list = []
    import numpy as np        
    for _, row in saint_output.iterrows():
        try:
            intensity_column.append(intensity_table[sample_groups[row['Bait']]].loc[row['Prey']].mean())
        except KeyError:
            intensity_column.append(np.nan)
    return intensity_column


@callback(
    Output('interactomics-saint-container', 'children'),
    Output('raw-interactomics-data', 'data'),
    Output('crapome-column-groups', 'data'),
    Input('button-run-saint-analysis', 'n_clicks'),
    State('interactomics-choose-additional-control-sets', 'value'),
    State('interactomics-choose-crapome-sets', 'value'),
    State('interactomics-choose-uploaded-controls', 'value'),
    State('interactomics-additional-steps','value'),
    State('output-data-upload', 'data'),
    State('session-uid', 'children'),
    prevent_initial_call=True
)
def generate_saint_container(_, inbuilt_controls, crapome_controls, control_sample_groups, additional_options, data_dictionary, session_uid) -> html.Div:
    spc_table: pd.DataFrame = pd.read_json(
        data_dictionary['data tables']['spc'], orient='split')
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
        data_dictionary['sample groups']['rev'],
        data_dictionary['other']['protein lengths'],
        data_dictionary['other']['protein names'],
        db.scripts('SAINTexpress'),
        control_table=inbuilt_control_table,
        control_groups=control_sample_groups,
    )
    saint_output.loc[:,'Is contaminant'] = saint_output['Prey'].isin(db.contaminant_list)
    saint_output = pd.merge(
        left=saint_output,
        right=crapome_table,
        how='left',
        left_on='Prey',
        right_index=True)
    saint_output['Bait uniprot'] = [data_dictionary['other']['bait uniprots'][b] for b in saint_output['Bait'].values]

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
    if 'Remove contaminants' in additional_options:
        before_contaminants: int = saint_output.shape[0]
        removed_contaminants: list = sorted(list(set(saint_output[saint_output['Is contaminant']]['PreyGene'].values)))
        saint_output = saint_output[~saint_output['Is contaminant']]
        after_contaminants: int = saint_output.shape[0]
        if after_contaminants == before_contaminants:
            saint_output = saint_output.drop(columns=['Is contaminant'])
        else:
            container_contents.append(
                html.Div([
                    f'Removed {before_contaminants-after_contaminants} "interactions" with common contaminants from the output dataset.',
                    html.Br(),
                    f'Removed contaminants: {", ".join(removed_contaminants)}'
                ])
            )

    intensity_table: pd.DataFrame = pd.read_json(
        data_dictionary['data tables']['intensity'], orient='split')
    if len(intensity_table.columns)>2:
        saint_output['Intensity'] = map_intensity(saint_output, intensity_table, data_dictionary['sample groups']['norm'])
    container_contents.append(
        html.Div([
            figure_generation.histogram(
                saint_output, x_column='BFDR', title='Saint BFDR distribution',height = 400, nbins = 100)[1],
            dcc.Graph(id='interactomics-saint-graph'),
            dbc.Label('Saint BFDR threshold:'),
            dcc.Slider(0, 0.1, 0.01, value=0.05,
                       id='saint-bfdr-filter-threshold'),
            dbc.Label('Crapome filtering percentage:'),
            dcc.Slider(1, 100, 10, value=20,
                       id='crapome-frequency-threshold'),
            dbc.Label('SPC fold change vs crapome threshold for rescue'),
            dcc.Slider(0, 10, 1, value=3,
                       id='crapome-rescue-threshold'),
            html.Div([dbc.Button('Done filtering',id='button-done-filtering')])
        ])
    )

    with open(db.get_cache_file(session_uid, 'SAINT info.txt'),'w',encoding='utf-8') as fil:
        discarded_str: str = '\t'.join(discarded_proteins)
        fil.write(f'Discarded proteins:\n{discarded_str}')
    saint_output.to_csv(db.get_cache_file(session_uid, 'Saint output.tsv'),sep='\t',index = False,encoding = 'utf-8')
    inbuilt_control_table.to_csv(db.get_cache_file(session_uid, 'Inbuilt controls.tsv'),sep='\t',index = False,encoding = 'utf-8')
    crapome_table.to_csv(db.get_cache_file(session_uid, 'Crapome table.tsv'),sep='\t',index = False,encoding = 'utf-8')
    return container_contents, saint_output.to_json(orient='split'), crapome_column_groups

def filter_saint(saint_output, saint_bfdr: float, crapome_freq: float, crapome_rescue: int)-> pd.DataFrame:
    crapome_columns: list = []
    for column in saint_output.columns:
        if ' Frequency' in column:
            crapome_columns.append((column, column.replace(' Frequency',' AvgSpc FC')))
    keep_col: list = []
    for _, row in saint_output.iterrows():
        keep: bool = True
        if row['BFDR']>= saint_bfdr:
            keep = False
        else:
            for freq_col, fc_col in crapome_columns:
                if row[freq_col] >= crapome_freq:
                    if row[fc_col] <= crapome_rescue:
                        keep = False
                        break
        keep_col.append(keep)
    saint_output: pd.DataFrame = saint_output[keep_col]
    if 'Bait uniprot' in saint_output.columns:
        saint_output = saint_output[saint_output['Prey'] != saint_output['Bait uniprot']]
    return saint_output.reset_index().drop(columns=['index'])

@callback (
    Output('interactomics-post-saint-analysis-graphs','children'),
    Output('interactomics-fully-filtered-saint-data','data'),
#    Input({'type': 'dynamic-button', 'index': ALL}, 'n_clicks'),
    Input('button-done-filtering', 'n_clicks'),
    State('saint-bfdr-filter-threshold', 'value'),
    State('crapome-frequency-threshold', 'value'),
    State('crapome-rescue-threshold', 'value'),
    State('raw-interactomics-data', 'data'),
    State('session-uid', 'children'),
    State('interactomics-choose-enrichments', 'value'),
    prevent_initial_call=True,
)
def post_saint_analysis(n_clicks,saint_bfdr,crapome_freq,crapome_rescue, saint_data,session_uid, enrichments_to_do) -> list:
    if n_clicks is None:
        return dash.no_update
    saint_output: pd.DataFrame = filter_saint(
        pd.read_json(saint_data, orient='split').reset_index().drop(columns=['index']),
        saint_bfdr,
        crapome_freq,
        crapome_rescue
    )

    # Filter based on saint BFDR and Crapome sets:
    # Each crapome set is considered by itself
    # TODO: Add combination crapome sets for the VL crapomes

    with open(db.get_cache_file(session_uid, 'Filtering info.txt'),'w',encoding='utf-8') as fil:
        fil.write(f'Saint BFDR filter: {saint_bfdr}\n')
        fil.write(f'Crapome frequency threshold filter: {crapome_freq}\n')
        fil.write(f'Fold change over crapome for rescue from crapome filter: {crapome_rescue}\n')

    # Map bait uniprot to saint table from data_dictionary, where it was put during experimental design table parsing.
    # Commented out because bait uniprot has already been mapped directly after SAINT analysis
    #saint_output['Bait uniprot'] = saint_output.apply(lambda x: data_dictionary['other']['bait uniprots'][x['Bait']],axis=1)
    data_functions.map_known(saint_output,db.get_known(saint_output['Bait uniprot'].unique()))
    saint_output.to_csv(db.get_cache_file(session_uid, 'Filtered Saint output.tsv'),sep='\t',index = False,encoding = 'utf-8')

    figures: list = []
    pdf_data: list  =[]
    known: pd.Series = saint_output[saint_output['Known interaction']]['Bait'].value_counts()
    unknown: pd.Series = saint_output[~saint_output['Known interaction']]['Bait'].value_counts()
    sernames: list = ['Known','Not known']
    index: list = sorted(list(saint_output['Bait'].unique()))
    for i in index:
        for j, ser in enumerate([known,unknown]):
            try:
                pdf_data.append([i, sernames[j], ser[i]])
            except KeyError:
                continue
    figure: dcc.Graph = figure_generation.bar_graph(
        'known-bar-graph',
        pd.DataFrame(data=pdf_data,columns=['Bait','Known or not','Preys']),
        title = 'Known interactors in preys',
        color_col = 'Known or not',
        x_name = 'Bait',
        y_name = 'Preys'
    )
    figures.append(figure)
    enrichment_names: list
    enrichment_results: pd.DataFrame
    enrichment_names, enrichment_results = data_functions.enrich_all_per_bait(saint_output, enrichments_to_do)#'pantherdb', 'defaults')
    save_dir:str = os.path.join(db.get_cache_dir(session_uid, make_subdirs = ['Enrichments']), 'Enrichments')
    with open(os.path.join(save_dir, f'Enrichment_information.txt'),'w',encoding='utf-8') as fil:
        for i, (rescol, sigcol, namecol, result) in enumerate(enrichment_results):
            result_filename:str = os.path.join(save_dir, enrichment_names[i])
            result.to_csv(os.path.join(save_dir, f'{result_filename}_enrichmentResult.tsv'),sep='\t')
            fil.write(f'{enrichment_names[i]}\n')
            fil.write(f'Result name column: {namecol}\n')
            fil.write(f'Result significance column: {sigcol}\n')
            fil.write(f'Result result column: {rescol}\n')
            fil.write(f'Result datafile: {result_filename}_enrichmentResult.tsv\n')
            fil.write('----------\n')
    
    figures.append(
        dbc.Tabs(
            id='interactomics-enrichment-tabs',
            children=make_enrichment_tabs(enrichment_names, enrichment_results)
        )
    )
    pca_figure: go.Figure
    pca_graph: dcc.Graph
    pca_figure, pca_graph = figure_generation.pca_plot(
        saint_output.pivot_table(index='Prey', columns='Bait',values='AvgSpec').fillna(0),
        {bait: bait for bait in saint_output['Bait'].unique()},
        plot_name = 'interactomics',
        plot_title = 'AvgSpec-based PCA'
    )
    figures.append(pca_graph)
    
    network_elements: list = create_network(saint_output)
    
    figures.append(
        html.Div([
            cyto.Cytoscape(
                id='cytoscape-layout-9',
                elements=network_elements,
                style={'width': '100%', 'height': '800px'},
                layout={
                    'name': 'cose',
                    #'EdgeLength': 50,
                    #'componentSpacing': 100,
                    #'nodeRepulsion': 400000,
                    'edgeElasticity': 100000,
                    'nodeSep': 80,
                    'spacingFactor': 0.8,
                    'animate': True
                }
            )
        ],style={"width": "75%", "border":"2px black solid"})
    )

    if 'Intensity' in saint_output.columns:
        figures.append(
            html.Div(
                id='volcano-plot-label',
                children=[
                    'Volcano plots',
                    html.Br(),
                    'Choose control sample:',
                    dbc.Select(
                        options=[
                            {'label': bait, 'value': bait}
                            for bait in sorted(list(saint_output['Bait'].unique()))
                        ],
                        required=True,
                        id='interactomics-control-dropdown',
                    ),
                    html.Div(id='interactomics-volcano-plot-container')
                ]
            )
        )
    return figures, saint_output.to_json(orient='split')


@callback(
    Output('interactomics-volcano-plot-container', 'children'),
    Input('interactomics-control-dropdown', 'value'),
    State('interactomics-fully-filtered-saint-data', 'data'),
    State('output-data-upload', 'data'),
)
def interactomics_volcano_plots(control_group, saint_output, data_dictionary) -> list:
    if control_group is None:
        return dash.no_update
    saint_output: pd.DataFrame = pd.read_json(saint_output, orient='split').reset_index().drop(columns=['index'])
    sample_groups: dict = data_dictionary['sample groups']['norm']
    intensity_table: pd.DataFrame = pd.read_json(
        data_dictionary['data tables']['intensity'], orient='split')
    non_hci_preys: list = [i for i in intensity_table.index if i not in saint_output['Prey'].values]
    intensity_table = intensity_table.drop(
        index=non_hci_preys
    )


    significants: pd.DataFrame
    figures: list
    volcano_graphs: list
    significants, figures, volcano_graphs = figure_generation.volcano_plots(
                intensity_table,
                sample_groups,
                control_group
            )
    significants.to_csv('SIGS.tsv',sep='\t')
    intensity_table.to_csv('INTS.tsv',sep='\t')
    data_dictionary['data tables'][f'Comparisons vs {control_group}'] = significants.to_json(orient='split')
    data_dictionary['data tables'][f'HCI intensity'] = intensity_table.to_json(orient='split')

    return [
        html.Div(id='volcano-plot-label', children='Volcano plots'),
        html.Div(id='volcano-plot-container', children=volcano_graphs
        )
    ]

def create_network( saint_data) -> list:
    nodes: list = [
        {'data': {'id': row['Prey'], 'label': row['PreyGene']}}
        for _,row in saint_data[['Prey','PreyGene']].drop_duplicates().iterrows()
    ]
    nodes.extend(
        [
            {'data': {'id': bait, 'label': bait}}
            for bait in saint_data['Bait'].unique()
        ]
    )
    edges: list = [
        {'data': {'source': row['Bait'], 'target': row['Prey']},}
        for _, row in saint_data.iterrows()
    ]

    return (nodes + edges)





def make_enrichment_tabs(names, results) -> list:
    tablist: list = []
    for i, (rescol, sigcol, namecol, result) in enumerate(results):
        keep_these: set = set(result[result[rescol]>=2][namecol].values)
        keep_these = keep_these & set(result[result[sigcol]<0.01][namecol].values)
        filtered_result: pd.DataFrame = result[result[namecol].isin(keep_these)]
        matrix: pd.DataFrame = pd.pivot_table(
            filtered_result,
            index=namecol,
            columns='Bait',
            values=rescol
            )
        graph:dcc.Graph
        if filtered_result.shape[0] == 0:
            graph = 'Nothing enriched.'
        else:
            graph = figure_generation.heatmap(
                matrix,
                plot_name = f'interactomics-enrichment-{names[i]}',
                value_name = rescol.replace('_',' '))

        table_label: str = f'{names[i]} data table'
        table: dash_table.DataTable = dash_table.DataTable(
                data = filtered_result.to_dict('records'),
                columns = [{"name": i, "id": i} for i in filtered_result.columns],
                page_size = 15,
                style_table={
                    'maxHeight': 600
                },
                style_data={
                    'width': '100px', 'minWidth': '25px', 'maxWidth': '250px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                filter_action = 'native',
                id = f'interactomics-enrichment-{table_label.replace(" ","-")}',
            )
        enrichment_tab: dbc.Card = dbc.Card(
            dbc.CardBody(
                [
                    html.P(f'{names[i]} heatmap'),
                    graph,
                    html.P(f'{names[i]} data table'),
                    table
                ]
            ),
            className="mt-3",
        )

        tablist.append(
            dbc.Tab(
                enrichment_tab,label=names[i]
            )
        )
    return tablist

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
    figure: go.Figure
    figure = figure_generation.bar_plot(
        bar_plot_df,
        'Protein counts after filtering',
        y_name='Prey count',
        x_name='Bait',
        color_col='Bait',
        height = 400,
        hide_legend = True
    )
    return (figure, filtered_saint.to_json(orient='split'))

@callback(
    Output('interactomics-choose-uploaded-controls','value'),
    [Input('interactomics-select-all-uploaded', 'value')],
    [State('interactomics-choose-uploaded-controls', 'options')],
    prevent_initial_call=True
)
def select_all_none_controls(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none

@callback(
    Output('interactomics-choose-additional-control-sets','value'),
    [Input('interactomics-select-all-inbuilt-controls', 'value')],
    [State('interactomics-choose-additional-control-sets', 'options')],
    prevent_initial_call=True
)
def select_all_none_inbuilt_controls(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none

@callback(
    Output('interactomics-choose-crapome-sets','value'),
    [Input('interactomics-select-all-crapomes', 'value')],
    [State('interactomics-choose-crapome-sets', 'options')],
    prevent_initial_call=True
)
def select_all_none_crapomes(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none

@callback(
    Output('interactomics-choose-enrichments','value'),
    [Input('interactomics-select-all-enrichments', 'value')],
    [State('interactomics-choose-enrichments', 'options')],
    prevent_initial_call=True
)
def select_all_none_enrichments(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none

def interactomics_control_col(all_sample_groups, chosen) -> dbc.Col:
    return dbc.Col([
        html.Div(
            checklist(
                'select all uploaded',
                ['Select all uploaded'],
                [],
                id_only = True,
                id_prefix = 'interactomics',
                simple_text_clean = True
            )
        ),
        html.Div(
            checklist(
                'Choose uploaded controls:',
                all_sample_groups,
                chosen,
                id_prefix='interactomics',
                simple_text_clean=True
            )
        ),
        html.Br(),
        html.Div(
            checklist(
                'Additional steps:',
                ['Remove contaminants'],
                ['Remove contaminants'],
                id_prefix='interactomics',
                simple_text_clean=True
            )
        )
    ])

def interactomics_inbuilt_control_col() -> dbc.Col:
    return dbc.Col([
        html.Div(
            checklist(
                'select all inbuilt controls',
                ['Select all inbuilt controls'],
                [],
                id_only = True,
                id_prefix = 'interactomics',
                simple_text_clean = True
            )
        ),
        html.Div(
            checklist(
                'Choose additional control sets:',
                db.controlsets,
                db.default_controlsets,
                disabled=db.disabled_controlsets,
                id_prefix='interactomics',
                simple_text_clean=True
            )
        )
    ])

def interactomics_crapome_col() -> dbc.Col:
    return dbc.Col([
        html.Div(
            checklist(
                'select all crapomes',
                ['Select all crapomes'],
                [],
                id_only = True,
                id_prefix = 'interactomics',
                simple_text_clean = True
            )
        ),
        html.Div(
            checklist(
                'Choose Crapome sets:',
                db.crapomesets,
                db.default_crapomesets,
                disabled=db.disabled_crapomesets,
                id_prefix='interactomics',
                simple_text_clean=True
            )
        )
    ])

def interactomics_enrichment_col() -> dbc.Col:
    return dbc.Col([
        html.Div(
            checklist(
                'select all enrichments',
                ['Select all enrichments'],
                [],
                id_only = True,
                id_prefix = 'interactomics',
                simple_text_clean = True
            )
        ),
        html.Div(
            checklist(
                'Choose enrichments:',
                data_functions.available_enrichments,
                data_functions.default_enrichments,
                id_prefix='interactomics',
                simple_text_clean=True
            )
        )
    ])

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
                dcc.Store(id='interactomics-fully-filtered-saint-data'),
                dcc.Store(id='crapome-column-groups'),
                html.Div(
                    id='interactomics-options',
                    children=[
                        html.Div(
                            children=[
                                dbc.Row(
                                    [
                                        interactomics_control_col(all_sample_groups, chosen),
                                        interactomics_inbuilt_control_col(),
                                        interactomics_crapome_col(),
                                        interactomics_enrichment_col()
                                    ]),
                                dbc.Row(
                                    [
                                        dbc.Button('Run SAINT analysis',
                                                   id='button-run-saint-analysis'),
                                    ])
                                ])
                            ]
                ),
                dcc.Loading(id='interactomics-saint-container',),
                dcc.Loading(id='interactomics-post-saint-analysis-graphs'),
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
            html.Div(id='upload-complete-indicator'),
            html.Hr(),
            dcc.Store(id='output-data-upload'),
            dcc.Store(id='figure-template-choice'),
            dcc.Store(id='workflow-choice'),
            dcc.Store(id='rep-colors'),
            dcc.Store(id='tic-traces'),
            dcc.Store(id='tic-info'),
            dcc.Store(id='auc-traces'),
            dcc.Interval(
                id='interval-component',
                interval=5*1000, # in milliseconds
                n_intervals=0,
                disabled=True
            ),
            html.Div(id='current-tic-idx', children = 0,hidden=True),
            html.Div(id='tic-graph'),
            dcc.Loading(
                id='qc-loading',
                children=[
                    html.Div(id='qc-plot-container',
                             children=[
                             ])
                ],
                type='default'
            ),
        ],
        className="mt-3"
    ),
)

tabs: html.Div = html.Div([
    dbc.Tabs(
        id='analysis-tabs',
        children=[
            dbc.Tab(upload_tab, label='Data upload and quick QC'),
        ]
    ),
    dbc.Button(
        'Export all data',
        id='button-export-all-data'
    )
    ])
layout: html.Div = html.Div([
    # dbc.Button('Save session as project',
    #            id='session-save-button', color='success'),
    html.Div(id='void',hidden=True),
    dcc.Store(id='figures-to-save'),
    tabs
])
