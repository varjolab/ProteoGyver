""" Restructured frontend for proteogyver app"""

import json
from uuid import uuid4
from datetime import datetime
from dash_bootstrap_components.themes import FLATLY
from dash import html, callback, Dash, no_update
from dash.dependencies import Input, Output, State
from components import ui_components as ui
from components.infra import data_stores, notifiers
from components import parsing, qc_analysis, data_validation
from components.figures.color_tools import get_assigned_colors
import plotly.io as pio

#import DbEngine


#db = DbEngine()
app = Dash(external_stylesheets=[FLATLY],suppress_callback_exceptions=True)
with open('new_parameters.json', encoding='utf-8') as fil:
    parameters = json.load(fil)

@callback(
    Output('upload-data-file-success', 'style'),
    Output('uploaded-data-table-info','data'),
    Output('uploaded-data-table','data'),
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
        return parsing.parse_generic_table(file_contents, file_name, mod_date, current_upload_style)
@callback(
    Output('upload-data-store','data', allow_duplicate=True),
    Input('begin-analysis-button','n_clicks'),
    State('uploaded-data-table','data'),
    State('uploaded-data-table-info','data'),
    State('uploaded-sample-table', 'data'),
    State('uploaded-sample-table-info', 'data'),
    State('figure-theme-dropdown', 'value'),
    prevent_initial_call=True
)
def validate_data(_, data_tables, data_info, expdes_table, expdes_info, figure_template) -> tuple:
    """Sets the figure template, and \
        sends data to preliminary analysis and returns the resulting dictionary.
    """
    pio.templates.default = figure_template
    return (data_validation.format_data(
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}--{uuid4()}',
        data_tables, data_info, expdes_table, expdes_info))
@callback(
    Output('upload-data-store','data', allow_duplicate=True),
    Input('discard-samples','data'),
    State('upload-data-store','data'),
    prevent_initial_call=True
)
def remove_samples(discard_samples_list, data_dictionary) -> dict:
    """Sends data to preliminary analysis and returns the resulting dictionary.
    """
    return data_validation.delete_samples(discard_samples_list, data_dictionary)


@callback(
    Output({'type': 'analysis-div','id':'qc-analysis-area'},'children'),
    Output('discard-samples-div','hidden'),
    Input('replicate-colors','data'),
    prevent_initial_call=True
)
def create_qc_area(_) -> tuple:
    """Creates the qc area div and unhides sample discard button"""
    return (ui.qc_area(), False)

@callback(
    Output('replicate-colors','data'),
    Input('upload-data-store','data'),
    prevent_initial_call=True
)
def assign_replicate_colors(data_dictionary) -> dict:
    return get_assigned_colors(data_dictionary['sample groups']['norm'])

@callback(
    Output('begin-analysis-button','disabled'),
    Input('uploaded-data-table-info','data'), Input('uploaded-sample-table-info','data'),
    Input('workflow-dropdown','value'), Input('figure-theme-dropdown','value'),
    Input('upload-data-file-success', 'style'), Input('upload-data-file-success', 'style'),
    prevent_initial_call=True
)
def check_inputs(*args) -> bool:
    """Checks that all inputs are present and we can begin."""
    return data_validation.validate_basic_inputs(*args)

@callback(
    Output('discard-sample-checklist-container','children'),
    Input('discard-samples-button', 'n_clicks'), State({'type': 'qc-plot','id':'count-plot-div'},'children'),
    State('upload-data-store','data'),
    prevent_initial_call=True
)
def open_discard_samples_modal(_, count_plot: list, data_dictionary: dict) -> tuple[bool, list]:
    return ui.discard_samples_checklist(
            count_plot,
            sorted(list(data_dictionary['sample groups']['rev'].keys()))
    )


@app.callback(
    Output('discard-samples-modal','is_open'),
    Input('discard-samples-button', 'n_clicks'),
    Input('done-discarding-button','n_clicks'),
    State('discard-samples-modal','is_open'),
    prevent_initial_call=True
)
def toggle_discard_modal(n1, n2, is_open) -> bool:
    if (n1>0) or (n2>0):
        return not is_open
    return is_open

@callback(
    Output('discard-samples','data'),
    Input('done-discarding-button','n_clicks'),
    State('checklist-select-samples-to-discard','value'),
    prevent_initial_call=True
)
def add_samples_to_discarded(n_clicks, chosen_samples: list) -> list:
    if n_clicks < 1:
        return no_update
    return chosen_samples

@callback(
    Output({'type': 'qc-plot','id':'count-plot-div'},'children'), Output('count-data-store','data'),
    Input('qc-area','children'),
    State('upload-data-store','data'), State('replicate-colors','data'),
)
def count_plot(_, data_dictionary: dict, replicate_colors:dict) -> tuple:
    """Calls qc_analysis.count_plot function to generate a count plot from the samples."""
    return qc_analysis.count_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
        )
@callback(
    Output({'type': 'qc-plot','id':'coverage-plot-div'},'children'), Output('coverage-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'count-plot-div'},'children'), State('upload-data-store','data'),
)
def coverage_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.coverage_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        parameters['Figure defaults']['half-height']
        )
@callback(
    Output({'type': 'qc-plot','id':'reproducibility-plot-div'},'children'), Output('reproducibility-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'coverage-plot-div'},'children'), State('upload-data-store','data')
)
def reproducibility_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.reproducibility_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        data_dictionary['sample groups']['norm'],
        parameters['Figure defaults']['full-height']
        )
@callback(
    Output({'type': 'qc-plot','id':'missing-plot-div'},'children'), Output('missing-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'reproducibility-plot-div'},'children'),
    State('upload-data-store','data'), State('replicate-colors','data'),
)
def missing_plot(_, data_dictionary: dict, replicate_colors:dict) -> tuple:
    return qc_analysis.missing_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
        )
@callback(
    Output({'type': 'qc-plot','id':'sum-plot-div'},'children'), Output('sum-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'missing-plot-div'},'children'),
    State('upload-data-store','data'), State('replicate-colors','data'),
)
def sum_plot(_, data_dictionary: dict, replicate_colors:dict) -> tuple:
    return qc_analysis.sum_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
        )
@callback(
    Output({'type': 'qc-plot','id':'mean-plot-div'},'children'), Output('mean-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'sum-plot-div'},'children'),
    State('upload-data-store','data'), State('replicate-colors','data'),
)
def mean_plot(_, data_dictionary: dict, replicate_colors:dict) -> tuple:
    return qc_analysis.mean_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
        )
@callback(
    Output({'type': 'qc-plot','id':'distribution-plot-div'},'children'), Output('distribution-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'mean-plot-div'},'children'),
    State('upload-data-store','data'), State('replicate-colors','data'),
)
def distribution_plot(_, data_dictionary: dict, replicate_colors:dict) -> tuple:
    return qc_analysis.distribution_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
        parsing.get_distribution_title(data_dictionary['data tables']['table to use'])
        )
@callback(
    Output({'type': 'qc-plot','id':'commonality-plot-div'},'children'), Output('commonality-data-store','data'),
    Input({'type': 'qc-plot', 'id': 'distribution-plot-div'},'children'),
    State('upload-data-store','data'),
)
def commonality_plot(_, data_dictionary: dict) -> tuple:
    return qc_analysis.commonality_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
        )

@callback(
    Output('qc-done-notifier','children'),
    Input({'type': 'qc-plot', 'id': 'commonality-plot-div'}, 'children'),
)
def qc_done(_) -> str:
    return ''

@callback(
    Output('proteomics-analysis-area','children'),
    Input('qc-done-notifier','children')
)
def start_proteomics(_) -> list:
    return [

    ]
















@callback(
    Output('toc-div','children'),
    Input('qc-done-notifier', 'children'),
    Input('workflow-done-notifier','children'),
    State('main-content-div', 'children'),
    prevent_initial_call=True
)
def table_of_contents(_,__, main_div_contents: list) -> html.Div:
    """updates table of contents to reflect the main div"""
    return ui.table_of_contents(main_div_contents)

@callback(
    Output('workflow-specific-input-div','children'),
    Output('workflow-specific-div','children'),
    Input('qc-done-notifier', 'children'),
    State('workflow-dropdown', 'value'),
)
def workflow_area(_, workflow: str) -> html.Div:
    return ui.workflow_area(workflow, parameters['workflow parameters'])

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
    Input('button-export-all-data', 'n_clicks'),
    prevent_initial_call=True
)
def download_all_data(_, ) -> dict:
    # DB dependent function
    pass

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
