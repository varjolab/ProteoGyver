""" Restructured frontend for proteogyver app"""

import json
from uuid import uuid4
from datetime import datetime
from dash_bootstrap_components.themes import FLATLY
from dash import html, callback, Dash
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
    Output('toc-div','children'),
    Input('qc-done-notifier', 'children'),
    State('main-content-div', 'children'),
)
def table_of_contents(_, main_div_contents: list) -> html.Div:
    """updates table of contents to reflect the main div"""
    return ui.table_of_contents(main_div_contents)

@callback(
    Output('workflow-specific-div','children'),
    Input('qc-done-notifier', 'children'),
    State('workflow-dropdown', 'value'),
)
def workflow_area(_, workflow: str) -> html.Div:
    return ui.workflow_area(workflow)






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
@callback(
    Output('contents-div', 'children', allow_duplicate=True), #Added in dash version 2.9
    Input('call1-button','n_clicks'),
    State('contents-div','children'),
    prevent_initial_call=True
)
def call1(_, contents) -> html.Div:
    ret_text = "Nulla ut erat quis enim dignissim egestas. Quisque id porttitor leo. Aliquam vitae euismod massa, vel posuere mi. Curabitur vulputate pretium metus sed placerat. Nulla facilisi. Pellentesque sit amet sapien at leo cursus ullamcorper. Etiam quis sodales felis, hendrerit lobortis neque."
    new_contents: list = []
    idstr = 'call1-div'
    # The loop under makes sure that whatever the div contains, we will not duplicate what this call will add. 
    for element in contents:
        needed = True
        try:
            if element['props']['id']==idstr:
                needed = False
        except KeyError:
            pass
        if needed:
            new_contents.append(element)
    new_contents.append(
        html.Div(id=idstr, children=[
            html.H2(id='h12',children=ret_text[:12]),
            html.H3(id='h13',children=ret_text[24:36]),
            html.H2(id='h13',children=ret_text[36:48]),
            html.H3(id='h13',children=ret_text[48:55]),
            html.H4(id='h13',children=ret_text[48:55]),
            html.H3(id='h13',children=ret_text[48:55]),
            html.H2(id='h13',children=ret_text[55:65]),
            html.H3(id='h13',children=ret_text[55:65]),
            html.P(id='p1',children=ret_text)
        ])
    )
    return new_contents

app.layout = html.Div([
    ui.main_sidebar(
        parameters['Possible values']['Figure templates'],
        parameters['Possible values']['Implemented workflows']),
    ui.main_content_div(),
    data_stores(),
    notifiers()
])

if __name__ == '__main__':
    app.run(debug=True)
