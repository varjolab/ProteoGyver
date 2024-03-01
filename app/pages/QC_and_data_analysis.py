""" Restructured frontend for proteogyver app"""
import os
import shutil
from uuid import uuid4
from datetime import datetime
from dash import html, callback, no_update, ALL, dcc, register_page
from dash.dependencies import Input, Output, State
from components import ui_components as ui
from components import infra
from components import parsing, qc_analysis, proteomics, interactomics, db_functions
from components.figures.color_tools import get_assigned_colors
from components.figures import tic_graph
import plotly.io as pio
import logging
from element_styles import CONTENT_STYLE


register_page(__name__, path='/')
logger = logging.getLogger(__name__)
logger.warning(f'{__name__} loading')

parameters = parsing.parse_parameters('parameters.json')
db_file: str = os.path.join(*parameters['Data paths']['Database file'])
contaminant_list: list = db_functions.get_contaminants(db_file)

layout = html.Div([
        ui.main_sidebar(
            parameters['Possible values']['Figure templates'],
            parameters['Possible values']['Implemented workflows']),
        ui.modals(),
        ui.main_content_div(),
        infra.invisible_utilities()
    ],
    style=CONTENT_STYLE
)

@callback(
    #  Output('workflow-stores', 'children'),
    # Output({'type': 'data-store', 'name': ALL}, 'clear_data'),
    Output('start-analysis-notifier', 'children'),
    Input('begin-analysis-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_data_stores(begin_clicks):
    '''Clears all data stores before analysis begins'''
# TODO: implement data clear operation.
    logger.warning(
        f'Data cleared. Start clicks: {begin_clicks}: {datetime.now()}')
    # return (tuple(True for _ in range(NUM_DATA_STORES)), '')
    # return (working_data_stores(), '')
    return ''


@callback(
    Output('upload-data-file-success', 'style'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-data-table-info-data-store'}, 'data'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-data-table-data-store'}, 'data'),
    Input('upload-data-file', 'contents'),
    State('upload-data-file', 'filename'),
    State('upload-data-file', 'last_modified'),
    State('upload-data-file-success', 'style'),
    prevent_initial_call=True
)
def handle_uploaded_data_table(file_contents, file_name, mod_date, current_upload_style) -> tuple:
    """Parses uploaded data table and sends data to data stores"""
    if file_contents is not None:
        return parsing.parse_data_file(
            file_contents, file_name, mod_date, current_upload_style, parameters['file loading']
        )


@callback(
    Output('upload-sample_table-file-success', 'style'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-sample-table-info-data-store'}, 'data'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-sample-table-data-store'}, 'data'),
    Input('upload-sample_table-file', 'contents'),
    State('upload-sample_table-file', 'filename'),
    State('upload-sample_table-file', 'last_modified'),
    State('upload-sample_table-file-success', 'style'),
    prevent_initial_call=True
)
def handle_uploaded_sample_table(file_contents, file_name, mod_date, current_upload_style) -> tuple:
    """Parses uploaded data table and sends data to data stores"""
    if file_contents is not None:
        return parsing.parse_sample_table(file_contents, file_name, mod_date, current_upload_style)


@callback(
    Output({'type': 'data-store', 'name': 'upload-data-store'},
           'data', allow_duplicate=True),
    Output('button-download-all-data', 'disabled'),
    Input('start-analysis-notifier', 'children'),
    State({'type': 'uploaded-data-store',
          'name': 'uploaded-data-table-data-store'}, 'data'),
    State({'type': 'uploaded-data-store',
          'name': 'uploaded-data-table-info-data-store'}, 'data'),
    State({'type': 'uploaded-data-store',
          'name': 'uploaded-sample-table-data-store'}, 'data'),
    State({'type': 'uploaded-data-store',
          'name': 'uploaded-sample-table-info-data-store'}, 'data'),
    State('figure-theme-dropdown', 'value'),
    State('sidebar-remove-common-contaminants', 'value'),
    State('sidebar-rename-replicates', 'value'),
    prevent_initial_call=True
)
def validate_data(_, data_tables, data_info, expdes_table, expdes_info, figure_template, remove_contaminants, replace_names) -> tuple:
    """Sets the figure template, and \
        sends data to preliminary analysis and returns the resulting dictionary.
    """
    logger.warning(f'Validating data: {datetime.now()}')
    cont: list = []
    if len(remove_contaminants) > 0:
        cont = contaminant_list
    repnames: bool = False
    if len(replace_names) > 0:
        repnames = True
    pio.templates.default = figure_template
    return (parsing.format_data(
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}--{uuid4()}',
        data_tables, data_info, expdes_table, expdes_info, cont, repnames), False)

@callback(
    Output({'type': 'data-store', 'name': 'upload-data-store'},
           'data', allow_duplicate=True),
    Input({'type': 'data-store', 'name': 'discard-samples-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def remove_samples(discard_samples_list, data_dictionary) -> dict:
    """Sends data to preliminary analysis and returns the resulting dictionary.
    """
    return parsing.delete_samples(discard_samples_list, data_dictionary)


@callback(
    Output({'type': 'analysis-div', 'id': 'qc-analysis-area'}, 'children'),
    Output('discard-samples-div', 'hidden'),
    Input({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def create_qc_area(_) -> tuple:
    """Creates the qc area div and unhides sample discard button"""
    return (ui.qc_area(), False)


@callback(
    Output({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    Output({'type': 'data-store',
           'name': 'replicate-colors-with-contaminants-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def assign_replicate_colors(data_dictionary) -> dict:
    return get_assigned_colors(data_dictionary['sample groups']['norm'])


@callback(
    Output('begin-analysis-button', 'disabled'),
    Input({'type': 'uploaded-data-store', 'name': 'uploaded-data-table-info-data-store'},
          'data'),
    Input({'type': 'uploaded-data-store',
          'name': 'uploaded-sample-table-info-data-store'}, 'data'),
    Input('workflow-dropdown', 'value'), Input('figure-theme-dropdown', 'value'),
    Input('upload-data-file-success',
          'style'), Input('upload-data-file-success', 'style'),
    prevent_initial_call=True
)
def check_inputs(*args) -> bool:
    """Checks that all inputs are present and we can begin."""
    return parsing.validate_basic_inputs(*args)


@callback(
    Output('discard-sample-checklist-container', 'children'),
    Input('discard-samples-button',
          'n_clicks'), State({'type': 'qc-plot', 'id': 'count-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def open_discard_samples_modal(_, count_plot: list, data_dictionary: dict) -> tuple[bool, list]:
    return ui.discard_samples_checklist(
        count_plot,
        sorted(list(data_dictionary['sample groups']['rev'].keys()))
    )


@callback(
    Output('discard-samples-modal', 'is_open'),
    Input('discard-samples-button', 'n_clicks'),
    Input('done-discarding-button', 'n_clicks'),
    State('discard-samples-modal', 'is_open'),
    prevent_initial_call=True
)
def toggle_discard_modal(n1, n2, is_open) -> bool:
    if (n1 > 0) or (n2 > 0):
        return not is_open
    return is_open


@callback(
    Output({'type': 'data-store', 'name': 'discard-samples-data-store'}, 'data'),
    Input('done-discarding-button', 'n_clicks'),
    State('checklist-select-samples-to-discard', 'value'),
    prevent_initial_call=True
)
def add_samples_to_discarded(n_clicks, chosen_samples: list) -> list:
    if n_clicks is None:
        return no_update
    if n_clicks < 1:
        return no_update
    return chosen_samples


@callback(
    Output({'type': 'qc-plot', 'id': 'tic-plot-div'},
           'children'), 
    Output({'type': 'data-store', 'name': 'tic-data-store'}, 'data'),
    Input('qc-area', 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'},
          'data'), State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def parse_tic_data(_, data_dictionary: dict, replicate_colors: dict) -> tuple:
    """Calls qc_analysis.count_plot function to generate a count plot from the samples."""
    return qc_analysis.parse_tic_data(
        data_dictionary['data tables']['experimental design'],
        replicate_colors,
        db_file,
        parameters['Figure defaults']['full-height']
    )

@callback(
    Output('qc-tic-plot','figure'),
    State({'type': 'data-store', 'name': 'tic-data-store'}, 'data'),
    Input('qc-tic-dropdown','value')
)
def plot_tic(tic_data, graph_type):
    return tic_graph.tic_figure(parameters['Figure defaults']['full-height'], tic_data, graph_type)
    

@callback(
    Output({'type': 'qc-plot', 'id': 'count-plot-div'},
           'children'), 
    Output({'type': 'data-store', 'name': 'count-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'tic-plot-div'},
          'children'), 
    State({'type': 'data-store', 'name': 'upload-data-store'},
          'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-with-contaminants-data-store'}, 'data'),
    prevent_initial_call=True
)
def count_plot(_, data_dictionary: dict, replicate_colors: dict) -> tuple:
    """Calls qc_analysis.count_plot function to generate a count plot from the samples."""
    return qc_analysis.count_plot(
        data_dictionary['data tables']['with-contaminants'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        contaminant_list,
        parameters['Figure defaults']['full-height'],
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'common-protein-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'common-protein-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'count-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def common_proteins_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.common_proteins(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        db_file,
        parameters['Figure defaults']['full-height'],
        additional_groups = {
            'Other contaminants': contaminant_list
        }
    )


@callback(
    Output({'type': 'qc-plot', 'id': 'coverage-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'coverage-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'common-protein-plot-div'},
          'children'), State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def coverage_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.coverage_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        parameters['Figure defaults']['half-height']
    )


@callback(
    Output({'type': 'qc-plot', 'id': 'reproducibility-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'reproducibility-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'coverage-plot-div'},
          'children'), State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def reproducibility_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.reproducibility_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        data_dictionary['sample groups']['norm'],
        data_dictionary['data tables']['table to use'],
        parameters['Figure defaults']['full-height']
    )


@callback(
    Output({'type': 'qc-plot', 'id': 'missing-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'missing-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'reproducibility-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), State(
        {'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def missing_plot(_, data_dictionary: dict, replicate_colors: dict) -> tuple:
    return qc_analysis.missing_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
    )


@callback(
    Output({'type': 'qc-plot', 'id': 'sum-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'sum-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'missing-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), State(
        {'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def sum_plot(_, data_dictionary: dict, replicate_colors: dict) -> tuple:
    return qc_analysis.sum_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
    )


@callback(
    Output({'type': 'qc-plot', 'id': 'mean-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'mean-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'sum-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), State(
        {'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def mean_plot(_, data_dictionary: dict, replicate_colors: dict) -> tuple:
    return qc_analysis.mean_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'distribution-plot-div'},
           'children'), Output({'type': 'data-store', 'name': 'distribution-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'mean-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), State(
        {'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def distribution_plot(_, data_dictionary: dict, replicate_colors: dict) -> tuple:
    return qc_analysis.distribution_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        replicate_colors,
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
        parsing.get_distribution_title(
            data_dictionary['data tables']['table to use'])
    )
@callback(
    Output({'type': 'data-store',
           'name': 'qc-commonality-plot-visible-groups-data-store'}, 'data'),
    Input('qc-commonality-plot-update-plot-button','n_clicks'),
    State('qc-commonality-select-visible-sample-groups', 'value'),
)
def pass_selected_groups_to_data_store(_, selection):
    return {'groups': selection}

@callback(
    Output({'type': 'qc-plot', 'id': 'commonality-plot-div'}, 'children'),
    Input({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
)
def generate_commonality_container(data_dictionary):
    print('THIS SHOULD ONLY BE DONE ONCE')
    sample_groups = sorted(list(data_dictionary['sample groups']['norm'].keys()))
    return qc_analysis.generate_commonality_container(sample_groups)

@callback(
    Output('qc-commonality-graph-div','children'),
    Output({'type': 'data-store', 'name': 'commonality-data-store'}, 'data'),
    Output({'type': 'data-store', 'name': 'commonality-figure-pdf-data-store'}, 'data'),
    State({'type': 'qc-plot', 'id': 'commonality-plot-div'}, 'children'),# TODO: remove if unneeded
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('sidebar-force-supervenn', 'value'),
    Input({'type': 'data-store',
           'name': 'qc-commonality-plot-visible-groups-data-store'}, 'data'),
    prevent_initial_call=True
)
def commonality_plot(_, data_dictionary: dict, force_supervenn: list, show_only_groups: dict) -> tuple:
    force_svenn: bool = False
    if len(force_supervenn) > 0:
        force_svenn = True
    show_groups = None
    if show_only_groups is not None:
        show_groups = show_only_groups['groups']
    else:
        show_groups = 'all'
    return qc_analysis.commonality_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
        force_svenn, only_groups=show_groups
    )

@callback(
    Output('qc-done-notifier', 'children'),
    Input({'type': 'qc-plot', 'id': 'commonality-plot-div'}, 'children'),
    prevent_initial_call=True
)
def qc_done(_) -> str:
    return ''


@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-na-filtered-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-na-filtered-data-store'}, 'data'),
    # Input('proteomics-loading-filtering', 'children'),
    Input('proteomics-run-button', 'n_clicks'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('proteomics-filter-minimum-percentage', 'value'),
    prevent_initial_call=True
)
def proteomics_filtering_plot(nclicks, uploaded_data: dict, filtering_percentage: int) -> tuple:
    if nclicks is None:
        return (no_update, no_update)
    if nclicks < 1:
        return (no_update, no_update)
    return proteomics.na_filter(uploaded_data, filtering_percentage, parameters['Figure defaults']['full-height'])

@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-normalization-plot-div'}, 'children'),
    Output({'type': 'data-store',
           'name': 'proteomics-normalization-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-na-filtered-data-store'}, 'data'),
    Input('proteomics-normalization-radio-option', 'value'),
    prevent_initial_call=True
)
def proteomics_normalization_plot(filtered_data: dict, normalization_option: str) -> html.Div:
    if filtered_data is None:
        return no_update
    return proteomics.normalization(filtered_data, normalization_option, parameters['Figure defaults']['full-height'], parameters['Config']['R error file'])


@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-missing-in-other-plot-div'}, 'children'),
    Input({'type': 'data-store', 'name': 'proteomics-normalization-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_missing_in_other_samples(normalized_data: dict) -> html.Div:
    return proteomics.missing_values_in_other_samples(normalized_data, parameters['Figure defaults']['half-height'])

@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-imputation-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-normalization-data-store'}, 'data'),
    Input('proteomics-imputation-radio-option', 'value'),
    prevent_initial_call=True
)
def proteomics_imputation_plot(normalized_data: dict, imputation_option: str) -> html.Div:
    if normalized_data is None:
        return no_update
    return proteomics.imputation(normalized_data, imputation_option, parameters['Figure defaults']['full-height'], parameters['Config']['R error file'])

@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-pertubation-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-pertubation-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('proteomics-control-dropdown', 'value'),
    State({'type': 'data-store',
           'name': 'proteomics-comparison-table-data-store'}, 'data'),
    State('proteomics-comparison-table-upload-success', 'style'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_pertubation(imputed_data: dict, data_dictionary: dict, control_group, comparison_data, comparison_upload_success_style, replicate_colors):
    return (html.Div(), '')
    if control_group is None:
        if (comparison_data is None):
            logger.warning(f'Proteomics volcano: no comparison data: {datetime.now()}')
            return no_update
        if (len(comparison_data) == 0):
            logger.warning(f'Proteomics volcano: Comparison data len 0: {datetime.now()}')
            return no_update
        if comparison_upload_success_style['background-color'] in ('red', 'grey'):
            logger.warning(f'Proteomics volcano: comparison data failed validation: {datetime.now()}')
            return no_update
    if imputed_data is None:
        return no_update
    sgroups: dict = data_dictionary['sample groups']['norm']
    comparisons: list = parsing.parse_comparisons(
        control_group, comparison_data, sgroups)
    
    return proteomics.pertubation(
        imputed_data,
        sgroups,
        [c[1] for c in comparisons],
        replicate_colors,
        parameters['Figure defaults']['half-height'],
        parameters['Figure defaults']['full-height']
    )

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-pca-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-pca-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_pca_plot(imputed_data: dict, upload_dict: dict, replicate_colors: dict) -> html.Div:
    return proteomics.pca(imputed_data, upload_dict['sample groups']['rev'], parameters['Figure defaults']['full-height'], replicate_colors)


@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-clustermap-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-clustermap-data-store'}, 'data'),
    Output({'type': 'done-notifier','name': 'proteomics-clustering-done-notifier'}, 'children', allow_duplicate=True),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_clustermap(imputed_data: dict) -> html.Div:
    return proteomics.clustermap(imputed_data, parameters['Figure defaults']['full-height']) + ('',)


@callback(
    Output('proteomics-comparison-table-upload-success', 'style'),
    Output({'type': 'data-store',
           'name': 'proteomics-comparison-table-data-store'}, 'data'),
    Input('proteomics-comparison-table-upload', 'contents'),
    State('proteomics-comparison-table-upload', 'filename'),
    State('proteomics-comparison-table-upload-success', 'style'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_check_comparison_table(contents, filename, current_style, data_dictionary):
    return parsing.check_comparison_file(contents, filename, data_dictionary['sample groups']['norm'], current_style)


@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-volcano-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-volcano-data-store'}, 'data'),
    Output('workflow-volcanoes-done-notifier', 'children'),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    Input('proteomics-control-dropdown', 'value'),
    Input({'type': 'data-store',
           'name': 'proteomics-comparison-table-data-store'}, 'data'),
    Input('proteomics-comparison-table-upload-success', 'style'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    Input('proteomics-fc-value-threshold', 'value'),
    Input('proteomics-p-value-threshold', 'value'),
    prevent_initial_call=True
)
def proteomics_volcano_plots(imputed_data, control_group, comparison_data, comparison_upload_success_style, data_dictionary, fc_thr, p_thr) -> tuple:
    if imputed_data is None:
        return no_update
    if control_group is None:
        if (comparison_data is None):
            logger.warning(f'Proteomics volcano: no comparison data: {datetime.now()}')
            return no_update
        if (len(comparison_data) == 0):
            logger.warning(f'Proteomics volcano: Comparison data len 0: {datetime.now()}')
            return no_update
        if comparison_upload_success_style['background-color'] in ('red', 'grey'):
            logger.warning(f'Proteomics volcano: comparison data failed validation: {datetime.now()}')
            return no_update
    sgroups: dict = data_dictionary['sample groups']['norm']
    comparisons: list = parsing.parse_comparisons(
        control_group, comparison_data, sgroups)
    return proteomics.volcano_plots(imputed_data, sgroups, comparisons, fc_thr, p_thr, parameters['Figure defaults']['full-height']) + ('',)

# Need to implement:
# GOBP mapping


@callback(
    Output('interactomics-choose-uploaded-controls', 'value'),
    [Input('interactomics-select-all-uploaded', 'value')],
    [State('interactomics-choose-uploaded-controls', 'options')],
    prevent_initial_call=True
)
def select_all_none_controls(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none


@callback(
    Output('interactomics-choose-additional-control-sets', 'value'),
    [Input('interactomics-select-all-inbuilt-controls', 'value')],
    [State('interactomics-choose-additional-control-sets', 'options')],
    prevent_initial_call=True
)
def select_all_none_inbuilt_controls(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none


@callback(
    Output('interactomics-choose-crapome-sets', 'value'),
    [Input('interactomics-select-all-crapomes', 'value')],
    [State('interactomics-choose-crapome-sets', 'options')],
    prevent_initial_call=True
)
def select_all_none_crapomes(all_selected, options) -> list:
    all_or_none: list = [option['value'] for option in options if all_selected]
    return all_or_none


@callback(
    Output({'type': 'workflow-plot',
           'id': 'interactomics-saint-container'}, 'children'),
    Output({'type': 'data-store',
           'name': 'interactomics-saint-input-data-store'}, 'data'),
    Output({'type': 'data-store',
           'name': 'interactomics-saint-crapome-data-store'}, 'data'),
    Input('button-run-saint-analysis', 'n_clicks'),
    State('interactomics-choose-uploaded-controls', 'value'),
    State('interactomics-choose-additional-control-sets', 'value'),
    State('interactomics-choose-crapome-sets', 'value'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('interactomics-nearest-control-filtering', 'value'),
    State('interactomics-num-controls', 'value'),
    prevent_initial_call=True
)
def interactomics_saint_analysis(nclicks, uploaded_controls: list, additional_controls: list, crapomes: list, uploaded_data: dict, proximity_filtering_checklist: list, n_controls: int) -> html.Div:
    if nclicks is None:
        return (no_update, no_update, no_update)
    if nclicks < 1:
        return (no_update, no_update, no_update)
    do_proximity_filtering: bool = (len(proximity_filtering_checklist) > 0)
    return interactomics.generate_saint_container(uploaded_data, uploaded_controls, additional_controls, crapomes, db_file, do_proximity_filtering, n_controls)


@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-output-data-store'}, 'data'),
    Output('interactomics-saint-running-loading', 'children'),
    Input({'type': 'data-store', 'name': 'interactomics-saint-input-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True,
    background=True
)
def interactomics_run_saint(saint_input, data_dictionary):
    return (interactomics.run_saint(
        saint_input,
        parameters['External tools']['SAINT tempdir'],
        data_dictionary['other']['session name'],
        data_dictionary['other']['bait uniprots']
    ), '')


@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'interactomics-saint-output-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-crapome-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_add_crapome_to_saint(saint_output, crapome):
    if saint_output == 'SAINT failed. Can not proceed.':
        return saint_output
    return interactomics.add_crapome(saint_output, crapome)


@callback(
    Output('workflow-done-notifier', 'children', allow_duplicate=True),
    Output('interactomics-saint-filtering-container', 'children'),
    Input({'type': 'data-store',
          'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_create_saint_filtering_container(saint_output_ready):
    if 'SAINT failed.' in saint_output_ready:
        return ('',html.Div(id='saint-failed', children=saint_output_ready))
    else:
        return ('',ui.saint_filtering_container(parameters['Figure defaults']['half-height']))


@callback(
    Output('interactomics-saint-bfdr-histogram', 'figure'),
    Output({'type': 'data-store',
           'name': 'interactomics-saint-bfdr-histogram-data-store'}, 'data'),
    Input('interactomics-saint-filtering-container', 'children'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data')
)
def interactomics_draw_saint_histogram(container_ready: list, saint_output: str, saint_output_filtered: str):
    if saint_output_filtered is not None:
        saint_output = saint_output_filtered
    return interactomics.saint_histogram(saint_output, parameters['Figure defaults']['half-height'])


@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    Input('interactomics-saint-bfdr-filter-threshold', 'value'),
    Input('interactomics-crapome-frequency-threshold', 'value'),
    Input('interactomics-crapome-rescue-threshold', 'value'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    State('interactomics-rescue-filtered-out', 'value')
    # prevent_initial_call=True
)
def interactomics_apply_saint_filtering(bfdr_threshold: float, crapome_percentage: int, crapome_fc: int, saint_output: str, rescue: list):
    return interactomics.saint_filtering(saint_output, bfdr_threshold, crapome_percentage, crapome_fc, len(rescue) > 0)


@callback(
    Output('interactomics-saint-graph', 'figure'),
    Output({'type': 'data-store',
           'name': 'interactomics-saint-graph-data-store'}, 'data'),
    Input({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_draw_saint_filtered_figure(filtered_output, replicate_colors):
    return interactomics.saint_counts(filtered_output, parameters['Figure defaults']['half-height'], replicate_colors)


@callback(
    Output({'type': 'analysis-div',
           'id': 'interactomics-analysis-post-saint-area'}, 'children'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    # State({'type': 'data-store', 'name': 'interactomics-saint-final-output-data-store'},'data'),
    prevent_initial_call=True
)
def interactomics_initiate_post_saint(_) -> html.Div:
    return ui.post_saint_cointainer()

@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-filtered-and-intensity-mapped-output-data-store'}, 'data'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_callback=True
)
def interactomics_map_intensity(n_clicks, unfiltered_saint_data, data_dictionary) -> str:
    if (n_clicks is None):
        return no_update
    if (n_clicks < 1):
        return no_update
    return interactomics.map_intensity(unfiltered_saint_data, data_dictionary['data tables']['intensity'], data_dictionary['sample groups']['norm'])

@callback( 
    Output('interactomics-known-loading', 'children'),
    Output({'type': 'data-store',
           'name': 'interactomics-saint-filt-int-known-data-store'}, 'data'),
    Input({'type': 'data-store',
          'name': 'interactomics-saint-filtered-and-intensity-mapped-output-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'replicate-colors-with-contaminants-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_known_plot(saint_output, rep_colors_with_cont) -> html.Div:
    return interactomics.known_plot(saint_output, db_file, rep_colors_with_cont, parameters['Figure defaults']['half-height'])

@callback(
    Output('interactomics-common-loading','children'),
    Output({'type': 'data-store', 'name': 'interactomics-common-protein-data-store'}, 'data'),
    Input({'type': 'data-store',
           'name': 'interactomics-saint-filt-int-known-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_common_proteins_plot(_, saint_data: dict) -> tuple:
    saint_data = interactomics.get_saint_matrix(saint_data)
    return qc_analysis.common_proteins(
        saint_data.to_json(orient='split'),
        db_file,
        parameters['Figure defaults']['full-height'],
        additional_groups = {
            'Other contaminants': contaminant_list
        }
    )

@callback(
    Output('interactomics-pca-loading', 'children'),
    Output({'type': 'data-store', 'name': 'interactomics-pca-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'interactomics-common-protein-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True

)
def interactomics_pca_plot(_, saint_data, replicate_colors) -> html.Div:
    return interactomics.pca(
        saint_data,
        parameters['Figure defaults']['full-height'],
        replicate_colors
    )

                #dcc.Loading(id='interactomics-volcano-loading'),
                
@callback(
    Output('interactomics-msmic-loading', 'children'),
    Output({'type': 'data-store',
           'name': 'interactomics-msmic-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'interactomics-pca-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_ms_microscopy_plots(_, saint_output) -> tuple:
    res =  interactomics.do_ms_microscopy(saint_output, db_file, parameters['Figure defaults']['full-height'], version='v1.0')
    return res

@callback(
    Output('workflow-done-notifier', 'children', allow_duplicate=True),
    Output('interactomics-network-loading', 'children'),
    Output({'type': 'data-store', 'name': 'interactomics-network-data-store'}, 'data'),
    Output({'type': 'data-store', 'name': 'interactomics-network-interactions-data-store'},'data'),
    Input({'type': 'data-store', 'name': 'interactomics-msmic-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_network_plot(_, saint_output):
    container, c_elements, interactions = interactomics.do_network(saint_output, parameters['Figure defaults']['full-height']['height'])
    return ('', container, c_elements, interactions)

@callback(
    Output('workflow-done-notifier', 'children', allow_duplicate=True),
    Output('interactomics-enrichment-loading', 'children'),
    Output({'type': 'data-store',
           'name': 'interactomics-enrichment-data-store'}, 'data'),
    Output({'type': 'data-store',
           'name': 'interactomics-enrichment-information-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'interactomics-network-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    State('interactomics-choose-enrichments', 'value'),
    prevent_initial_call=True,
    background=True
)
def interactomics_enrichment(_, saint_output, chosen_enrichments):
    return ('',) + interactomics.enrich(saint_output, chosen_enrichments, parameters['Figure defaults']['full-height'])

########################################
# Interactomics network plot callbacks #
########################################
## Visdcc pakcage is currently unused, because it's impossible to export anything sensible out of it.
## Export would require javascript, but there is no time to do any of this
## Perhaps in the future though? 
## This callback will be left here, same as the modifications in interactomics.network_display_data
## So that if export is possible in the future, we can just plug in the visdcc network plot.
@callback(
    Output("nodedata-div", "children"),
    Input("cytoscape", "tapNode"),
   # Input('visdcc-network','selection'),
    State({'type': 'data-store',
          'name': 'interactomics-network-interactions-data-store'},'data')
)
def display_tap_node(node_data, int_data, network_type: str = 'Cytoscape'):
    if not node_data:
        return None
    if network_type == 'Cytoscape':
        ret = interactomics.network_display_data(
            node_data,
            int_data,
            parameters['Figure defaults']['full-height']['height']
        )

@callback(
    Output("cytoscape", "layout"),
    Input("dropdown-layout", "value")
)
def update_cytoscape_layout(layout):
    ret_dic =  {"name": layout}
    if layout in parameters['Cytoscape layout parameters']:
        for k, v in parameters['Cytoscape layout parameters'][layout]:
            ret_dic[k] = v
    return ret_dic


########################################

@callback(
    Output('toc-div', 'children'),
    Input('qc-done-notifier', 'children'),
    Input('workflow-done-notifier', 'children'),
    Input({'type': 'done-notifier', 'name': 'proteomics-clustering-done-notifier'}, 'children'),
    Input('workflow-volcanoes-done-notifier', 'children'),
    State('main-content-div', 'children'),
    prevent_initial_call=True
)
def table_of_contents(_, __, ___, ____, main_div_contents: list) -> html.Div:
    """updates table of contents to reflect the main div"""
    return ui.table_of_contents(main_div_contents)


@callback(
    Output('workflow-specific-input-div', 'children'),
    Output('workflow-specific-div', 'children'),
    Input('qc-done-notifier', 'children'),
    State('workflow-dropdown', 'value'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True,
)
def workflow_area(_, workflow: str, data_dictionary: dict) -> html.Div:
    return ui.workflow_area(workflow, parameters['workflow parameters'], data_dictionary)


@callback(
    Output('download-sample_table-template', 'data'),
    Input('button-download-sample_table-template', 'n_clicks'),
    prevent_initial_call=True,
)
def sample_table_example_download(_) -> dict:
    return dcc.send_file(os.path.join(*parameters['Data paths']['Example sample table file']))


@callback(
    Output('download-proteomics-comparison-example', 'data'),
    Input('download-proteomics-comparison-example-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_example_comparison_file(n_clicks) -> dict:
    if n_clicks is None:
        return None
    if n_clicks == 0:
        return None
    return dcc.send_file(os.path.join(*parameters['Data paths']['Example proteomics comparison file']))


@callback(
    Output('download-datafile-example', 'data'),
    Input('button-download-datafile-example', 'n_clicks'),
    prevent_initial_call=True,
    background=True
)
def download_data_table_example(_) -> dict:
    logger.warning(f'received DT download request at {datetime.now()}')
    return dcc.send_file(os.path.join(*parameters['Data paths']['Example data file']))

@callback(
    Output('download-temp-dir-ready','children'),
    Output('button-download-all-data-text','children',allow_duplicate = True),
    Input('button-download-all-data', 'n_clicks'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('button-download-all-data-text','children'),
    prevent_initial_call=True,
    background=True
)
def prepare_for_download(_, main_data, button_text):
    export_dir: str = os.path.join(*parameters['Data paths']['Cache dir'],  main_data['other']['session name'], 'Proteogyver output')
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    import markdown
    with open(os.path.join('data','output_guide.md')) as fil:
        text = fil.read()
        html = markdown.markdown(text)
    with open(os.path.join(export_dir, 'README.html'),'w',encoding='utf-8') as fil:
        fil.write(html)
    return export_dir, button_text

@callback(
    Output('download_temp1', 'children'),
    Output('button-download-all-data-text','children', allow_duplicate=True),
    Input('download-temp-dir-ready','children'),
    State('input-stores', 'children'),
    State('button-download-all-data-text','children'),
    prevent_initial_call=True,
    background=True
)
def prepare_data1(export_dir, stores, button_text) -> dict:
    logger.warning(f'received download request data1 at {datetime.now()}')
    infra.save_data_stores(stores, export_dir)
    return 'data1 done', button_text

@callback(
    Output('download_temp2', 'children'),
    Output('button-download-all-data-text','children', allow_duplicate=True),
    Input('download-temp-dir-ready','children'),
    State('workflow-stores', 'children'),
    State('button-download-all-data-text','children'),
    prevent_initial_call=True,
    background=True
)
def prepare_data2(export_dir, stores, button_text) -> dict:
    logger.warning(f'received download request data2 at {datetime.now()}')
    infra.save_data_stores(stores, export_dir)
    return 'data2 done', button_text

@callback(
    Output('download_temp3', 'children'),
    Output('button-download-all-data-text','children', allow_duplicate=True),
    Input('download-temp-dir-ready','children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State('button-download-all-data-text','children'),
    State({'type': 'data-store', 'name': 'commonality-figure-pdf-data-store'}, 'data'),
    State('workflow-dropdown', 'value'),
    prevent_initial_call=True,
    background=True
)
def prepare_data3(export_dir, analysis_divs, button_text,commonality_pdf_data, workflow) -> dict:
    logger.warning(f'received download request data3 at {datetime.now()}')
    figure_output_formats = ['html', 'png', 'pdf']
    infra.save_figures(analysis_divs, export_dir,
                 figure_output_formats, commonality_pdf_data, workflow)
    return 'data3 done', button_text

@callback(
    Output('download_temp4', 'children'),
    Output('button-download-all-data-text','children', allow_duplicate=True),
    Input('download-temp-dir-ready','children'),
    State({'type': 'input-div', 'id': ALL}, 'children'),
    State('button-download-all-data-text','children'),
    prevent_initial_call=True,
    background=True
)
def prepare_data4(export_dir, input_divs, button_text) -> dict:
    logger.warning(f'received download request data4 at {datetime.now()}')
    infra.save_input_information(input_divs, export_dir)
    return 'data4 done',button_text
    
# NExt implement the below, but also move make_archive that way.

@callback(
    Output('download-all-data', 'data'),
    Output('button-download-all-data-text','children', allow_duplicate=True),
    Input('download-temp-dir-ready','children'),
    State('button-download-all-data-text','children'),
    Input('download_temp1', 'children'),
    Input('download_temp2', 'children'),
    Input('download_temp3', 'children'),
    Input('download_temp4', 'children'),
    prevent_initial_call=True,
    background=True,
)
def send_data(export_dir, button_text, *args) -> dict:
    for a in args:
        if not 'done' in a:
            return no_update, no_update
    export_zip_name: str = export_dir.rstrip(os.sep)
    shutil.make_archive(export_zip_name, 'zip', export_dir)
    export_zip_name +=  '.zip'
    return dcc.send_file(export_zip_name), button_text

    
