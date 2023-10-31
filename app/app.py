""" Restructured frontend for proteogyver app"""
import os
from celery import Celery
from uuid import uuid4
from datetime import datetime
from dash_bootstrap_components.themes import FLATLY
from dash.long_callback import CeleryLongCallbackManager
from dash import html, callback, Dash, no_update, ALL, dcc
from dash.dependencies import Input, Output, State
from components import ui_components as ui
from components.infra import data_stores, notifiers, prepare_download
from components import parsing, qc_analysis, proteomics, interactomics, db_functions
from components.figures.color_tools import get_assigned_colors
import plotly.io as pio

# import DbEngine


# db = DbEngine()
celery_app = Celery(
    __name__, broker="redis://localhost:6379/0", backend="redis://localhost:6379/1"
)
long_callback_manager = CeleryLongCallbackManager(celery_app)

app = Dash(__name__, external_stylesheets=[
           FLATLY], suppress_callback_exceptions=True, long_callback_manager=long_callback_manager)
server = app.server
parameters = parsing.parse_parameters('parameters.json')
db_file: str = os.path.join(*parameters['Data paths']['Database file'])
contaminant_list: list = db_functions.get_contaminants(db_file)


@callback(
    Output('upload-data-file-success', 'style'),
    Output('uploaded-data-table-info', 'data'),
    Output('uploaded-data-table', 'data'),
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
    Output('uploaded-sample-table-info', 'data'),
    Output('uploaded-sample-table', 'data'),
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
    Output('upload-data-store', 'data', allow_duplicate=True),
    Output('button-download-all-data', 'disabled'),
    Input('begin-analysis-button', 'n_clicks'),
    State('uploaded-data-table', 'data'),
    State('uploaded-data-table-info', 'data'),
    State('uploaded-sample-table', 'data'),
    State('uploaded-sample-table-info', 'data'),
    State('figure-theme-dropdown', 'value'),
    State('sidebar-remove-common-contaminants', 'value'),
    prevent_initial_call=True
)
def validate_data(_, data_tables, data_info, expdes_table, expdes_info, figure_template, remove_contaminants) -> tuple:
    """Sets the figure template, and \
        sends data to preliminary analysis and returns the resulting dictionary.
    """
    cont: list = []
    if len(remove_contaminants) > 0:
        cont = contaminant_list
    pio.templates.default = figure_template
    return (parsing.format_data(
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}--{uuid4()}',
        data_tables, data_info, expdes_table, expdes_info, cont), False)


@callback(
    Output('upload-data-store', 'data', allow_duplicate=True),
    Input('discard-samples', 'data'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def remove_samples(discard_samples_list, data_dictionary) -> dict:
    """Sends data to preliminary analysis and returns the resulting dictionary.
    """
    return parsing.delete_samples(discard_samples_list, data_dictionary)


@callback(
    Output({'type': 'analysis-div', 'id': 'qc-analysis-area'}, 'children'),
    Output('discard-samples-div', 'hidden'),
    Input('replicate-colors', 'data'),
    prevent_initial_call=True
)
def create_qc_area(_) -> tuple:
    """Creates the qc area div and unhides sample discard button"""
    return (ui.qc_area(), False)


@callback(
    Output('replicate-colors', 'data'),
    Output('replicate-colors-with-contaminants', 'data'),
    Input('upload-data-store', 'data'),
    prevent_initial_call=True
)
def assign_replicate_colors(data_dictionary) -> dict:
    return get_assigned_colors(data_dictionary['sample groups']['norm'])


@callback(
    Output('begin-analysis-button', 'disabled'),
    Input('uploaded-data-table-info',
          'data'), Input('uploaded-sample-table-info', 'data'),
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
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def open_discard_samples_modal(_, count_plot: list, data_dictionary: dict) -> tuple[bool, list]:
    return ui.discard_samples_checklist(
        count_plot,
        sorted(list(data_dictionary['sample groups']['rev'].keys()))
    )


@app.callback(
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
    Output('discard-samples', 'data'),
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
    Output({'type': 'qc-plot', 'id': 'count-plot-div'},
           'children'), Output('count-data-store', 'data'),
    Input('qc-area', 'children'),
    State('upload-data-store',
          'data'), State('replicate-colors-with-contaminants', 'data'),
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
    Output({'type': 'qc-plot', 'id': 'coverage-plot-div'},
           'children'), Output('coverage-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'count-plot-div'},
          'children'), State('upload-data-store', 'data'),
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
           'children'), Output('reproducibility-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'coverage-plot-div'},
          'children'), State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def reproducibility_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.reproducibility_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        data_dictionary['sample groups']['norm'],
        parameters['Figure defaults']['full-height']
    )


@callback(
    Output({'type': 'qc-plot', 'id': 'missing-plot-div'},
           'children'), Output('missing-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'reproducibility-plot-div'}, 'children'),
    State('upload-data-store', 'data'), State('replicate-colors', 'data'),
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
           'children'), Output('sum-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'missing-plot-div'}, 'children'),
    State('upload-data-store', 'data'), State('replicate-colors', 'data'),
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
           'children'), Output('mean-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'sum-plot-div'}, 'children'),
    State('upload-data-store', 'data'), State('replicate-colors', 'data'),
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
           'children'), Output('distribution-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'mean-plot-div'}, 'children'),
    State('upload-data-store', 'data'), State('replicate-colors', 'data'),
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
    Output({'type': 'qc-plot', 'id': 'commonality-plot-div'},
           'children'), Output('commonality-data-store', 'data'),
    Input({'type': 'qc-plot', 'id': 'distribution-plot-div'}, 'children'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def commonality_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.commonality_plot(
        data_dictionary['data tables'][data_dictionary['data tables']
                                       ['table to use']],
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
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
           'id': 'proteomics-filtering-plot-div'}, 'children'),
    Output('proteomics-na-filtered-data-store', 'data'),
    Input('proteomics-loading-filtering', 'children'),
    State('upload-data-store', 'data'),
    State('proteomics-filter-minimum-percentage', 'value'),
    prevent_initial_call=True
)
def proteomics_filtering_plot(_, uploaded_data: dict, filtering_percentage: int):
    return proteomics.na_filter(uploaded_data, filtering_percentage, parameters['Figure defaults']['full-height'])


@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-normalization-plot-div'}, 'children'),
    Output('proteomics-normalization-data-store', 'data'),
    Input('proteomics-na-filtered-data-store', 'data'),
    Input('proteomics-normalization-radio-option', 'value'),
    prevent_initial_call=True
)
def proteomics_normalization_plot(filtered_data: dict, normalization_option: str) -> html.Div:
    if filtered_data is None:
        return no_update
    return proteomics.normalization(filtered_data, normalization_option, parameters['Figure defaults']['full-height'])


@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-imputation-plot-div'}, 'children'),
    Output('proteomics-imputation-data-store', 'data'),
    Input('proteomics-normalization-data-store', 'data'),
    Input('proteomics-imputation-radio-option', 'value'),
    prevent_initial_call=True
)
def proteomics_imputation_plot(normalized_data: dict, imputation_option: str) -> html.Div:
    if normalized_data is None:
        return no_update
    return proteomics.imputation(normalized_data, imputation_option, parameters['Figure defaults']['full-height'])


@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-pca-plot-div'}, 'children'),
    Output('proteomics-pca-data-store', 'data'),
    Input('proteomics-imputation-data-store', 'data'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def proteomics_pca_plot(imputed_data: dict, upload_dict: dict) -> html.Div:
    return proteomics.pca(imputed_data, upload_dict['sample groups']['rev'], parameters['Figure defaults']['full-height'])


@callback(
    Output({'type': 'workflow-plot',
           'id': 'proteomics-clustermap-plot-div'}, 'children'),
    Output('proteomics-clustermap-data-store', 'data'),
    Output('workflow-done-notifier', 'children'),
    Input('proteomics-imputation-data-store', 'data'),
    prevent_initial_call=True
)
def proteomics_clustermap(imputed_data: dict) -> html.Div:
    return proteomics.clustermap(imputed_data, parameters['Figure defaults']['full-height']) + ('',)


@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-volcano-plot-div'}, 'children'),
    Output('proteomics-volcano-data-store', 'data'),
    Output('workflow-volcanoes-done-notifier', 'children'),
    Input('proteomics-imputation-data-store', 'data'),
    Input('proteomics-control-dropdown', 'value'),
    Input('proteomics-comparison-table-upload', 'data'),
    Input('proteomics-comparison-table-upload', 'filename'),
    State('upload-data-store', 'data'),
    Input('proteomics-fc-value-threshold', 'value'),
    Input('proteomics-p-value-threshold', 'value'),
    prevent_initial_call=True
)
def proteomics_volcano_plots(imputed_data, control_group, comparison_file, comparison_file_name, data_dictionary, fc_thr, p_thr) -> tuple:
    if (control_group is None) and (comparison_file is None):
        return no_update
    else:
        sgroups: dict = data_dictionary['sample groups']['norm']
        comparisons: list = parsing.parse_comparisons(
            control_group, comparison_file, comparison_file_name, sgroups)
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
    Output('interactomics-saint-input-data-store', 'data'),
    Output('interactomics-saint-crapome-data-store', 'data'),
    Input('button-run-saint-analysis', 'n_clicks'),
    State('interactomics-choose-uploaded-controls', 'value'),
    State('interactomics-choose-additional-control-sets', 'value'),
    State('interactomics-choose-crapome-sets', 'value'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def interactomics_saint_analysis(nclicks, uploaded_controls: list, additional_controls: list, crapomes: list, uploaded_data: dict) -> html.Div:
    if nclicks is None:
        return (no_update, no_update, no_update)
    if nclicks < 1:
        return (no_update, no_update, no_update)
    return interactomics.generate_saint_container(uploaded_data, uploaded_controls, additional_controls, crapomes, db_file)


@app.long_callback(
    Output('interactomics-saint-output-data-store', 'data'),
    Input('interactomics-saint-input-data-store', 'data'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def interactomics_run_saint(saint_input, data_dictionary):
    return interactomics.run_saint(
        saint_input,
        parameters['External tools']['SAINT']['spc'],
        parameters['Data paths']['Error log'],
        data_dictionary['other']['session name'],
        data_dictionary['other']['bait uniprots']
    )


@callback(
    Output('interactomics-saint-final-output-data-store', 'data'),
    Input('interactomics-saint-output-data-store', 'data'),
    State('interactomics-saint-crapome-data-store', 'data'),
    prevent_initial_call=True
)
def interactomics_add_crapome_to_saint(saint_output, crapome):
    if saint_output == 'SAINT failed. Can not proceed.':
        return saint_output
    return interactomics.add_crapome(saint_output, crapome)


@callback(
    Output('interactomics-saint-filtering-container', 'children'),
    Input('interactomics-saint-final-output-data-store', 'data'),
    prevent_initial_call=True
)
def interactomics_create_saint_filtering_container(saint_output_ready):
    if 'SAINT failed.' in saint_output_ready:
        return html.Div(id='saint-failed', children=saint_output_ready)
    else:
        return ui.saint_filtering_container(parameters['Figure defaults']['half-height'])


@callback(
    Output('interactomics-saint-bfdr-histogram', 'figure'),
    Input('interactomics-saint-filtering-container', 'children'),
    State('interactomics-saint-final-output-data-store', 'data')
)
def interactomics_draw_saint_histogram(container_ready: list, saint_output: str):
    return interactomics.saint_histogram(saint_output, parameters['Figure defaults']['half-height'])


@callback(
    Output('interactomics-saint-filtered-output-data-store', 'data'),
    Input('interactomics-saint-bfdr-filter-threshold', 'value'),
    Input('interactomics-crapome-frequency-threshold', 'value'),
    Input('interactomics-crapome-rescue-threshold', 'value'),
    State('interactomics-saint-final-output-data-store', 'data'),
    # prevent_initial_call=True
)
def interactomics_apply_saint_filtering(bfdr_threshold: float, crapome_percentage: int, crapome_fc: int, saint_output: str):
    return interactomics.saint_filtering(saint_output, bfdr_threshold, crapome_percentage, crapome_fc)


@callback(
    Output('interactomics-saint-graph', 'figure'),
    Input('interactomics-saint-filtered-output-data-store', 'data'),
    State('replicate-colors', 'data'),
    prevent_initial_call=True
)
def interactomics_draw_saint_filtered_figure(filtered_output, replicate_colors):
    return interactomics.saint_counts(filtered_output, parameters['Figure defaults']['half-height'], replicate_colors)


@callback(
    Output({'type': 'analysis-div',
           'id': 'interactomics-analysis-post-saint-area'}, 'children'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    # State('interactomics-saint-final-output-data-store','data'),
    prevent_initial_call=True
)
def interactomics_initiate_post_saint(_) -> html.Div:
    return ui.post_saint_cointainer()


@callback(
    Output('interactomics-saint-filtered-and-intensity-mapped-output-data-store', 'data'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    State('interactomics-saint-filtered-output-data-store', 'data'),
    State('upload-data-store', 'data'),
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
    Output('interactomics-saint-filt-int-known-output-data-store', 'data'),
    Input('interactomics-saint-filtered-and-intensity-mapped-output-data-store', 'data'),
    State('replicate-colors-with-contaminants', 'data'),
    prevent_initial_call=True
)
def interactomics_known_plot(saint_output, rep_colors_with_cont) -> html.Div:
    return interactomics.known_plot(saint_output, db_file, rep_colors_with_cont, parameters['Figure defaults']['half-height'])


@callback(
    Output('interactomics-pca-loading', 'children'),
    Output('interactomics-pca-data-store', 'data'),
    Input('interactomics-saint-filt-int-known-output-data-store', 'data'),
    State('interactomics-saint-filtered-output-data-store', 'data'),
    prevent_initial_call=True

)
def interactomics_pca_plot(_, saint_data) -> html.Div:
    return interactomics.pca(
        saint_data,
        parameters['Figure defaults']['full-height']
    )


@callback(
    Output('interactomics-enrichment-loading', 'children'),
    Output('interactomics-enrichment-data-store', 'data'),
    Output('interactomics-enrichment-information-data-store', 'data'),
    Input('interactomics-pca-data-store', 'data'),
    State('interactomics-saint-filtered-output-data-store', 'data'),
    State('interactomics-choose-enrichments', 'value'),
    prevent_initial_call=True
)
def interactomics_enrichment(_, saint_output, chosen_enrichments):
    return interactomics.enrich(saint_output, chosen_enrichments, parameters['Figure defaults']['full-height'])


@callback(
    Output('interactomics-network-loading', 'children'),
    Output('interactomics-network-data-store', 'data'),
    Input('interactomics-enrichment-data-store', 'data'),
    State('interactomics-saint-filtered-output-data-store', 'data'),
    prevent_initial_call=True
)
def interactomics_network_plot(_, saint_input):
    return ('', '')


@callback(
    Output('toc-div', 'children'),
    Input('qc-done-notifier', 'children'),
    Input('workflow-done-notifier', 'children'),
    Input('workflow-volcanoes-done-notifier', 'children'),
    State('main-content-div', 'children'),
    prevent_initial_call=True
)
def table_of_contents(_, __, ___, main_div_contents: list) -> html.Div:
    """updates table of contents to reflect the main div"""
    return ui.table_of_contents(main_div_contents)


@callback(
    Output('workflow-specific-input-div', 'children'),
    Output('workflow-specific-div', 'children'),
    Input('qc-done-notifier', 'children'),
    State('workflow-dropdown', 'value'),
    State('upload-data-store', 'data'),
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
    # DB dependent function
    pass


@callback(
    Output('download-datafile-example', 'data'),
    Input('button-download-datafile-example', 'n_clicks'),
    prevent_initial_call=True,
)
def download_data_table_example(_) -> dict:
    # DB dependent function
    pass


@callback(
    Output('download-all-data', 'data'),
    Input('button-download-all-data', 'n_clicks'),
    State('dcc-stores', 'children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State({'type': 'input-div', 'id': ALL}, 'children'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def download_all_data(nclicks, stores, analysis_divs, input_divs, main_data) -> dict:
    figure_output_formats = ['html', 'png', 'pdf']
    export_zip_name: str = prepare_download(
        stores, analysis_divs, input_divs, parameters['Data paths']['Cache dir'], main_data['other']['session name'], figure_output_formats)
    return dcc.send_file(export_zip_name)
    # DB dependent function


app.layout = html.Div([
    ui.main_sidebar(
        parameters['Possible values']['Figure templates'],
        parameters['Possible values']['Implemented workflows']),
    ui.modals(),
    ui.main_content_div(),
    data_stores(),
    notifiers()
])

if __name__ == '__main__':
    app.run(debug=True)
