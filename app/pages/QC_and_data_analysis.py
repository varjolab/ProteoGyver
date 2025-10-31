"""Restructured frontend for proteogyver app.

This module contains the main frontend logic for the Proteogyver application,
including callbacks for data processing, analysis, and visualization.

Attributes:
    parameters (dict): Application parameters loaded from parameters.toml
    db_file (str): Path to the database file
    contaminant_list (list): List of contaminant proteins
    figure_output_formats (list): Supported figure export formats
    layout (html.Div): Main application layout
"""
import os
import shutil
import traceback
import zipfile
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from dash import html, callback, no_update, ALL, dcc, register_page
from dash.dependencies import Input, Output, State
from components import ui_components as ui
from components import infra
from components import parsing, qc_analysis, proteomics, interactomics, db_functions
from components.figures.color_tools import get_assigned_colors, remove_unwanted_colors
from components.figures import tic_graph
from components.tools import utils
import plotly.io as pio
import logging
from element_styles import CONTENT_STYLE
from typing import Any, Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go


register_page(__name__, path='/')
logger = logging.getLogger(__name__)
logger.info(f'{__name__} loading')

parameters_file = 'config/parameters.toml'
parameters: Dict[str, Any] = parsing.parse_parameters(Path(parameters_file))
db_file: str = os.path.join(*parameters['Data paths']['Database file'])
contaminant_list: List[str] = db_functions.get_contaminants(db_file)
figure_output_formats: List[str] = ['html', 'png', 'pdf']

layout: html.Div = html.Div([
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
#TODO: implement clearing.
#TODO: Alternatively we could load the data store elements at this point, except for the ones needed to ingest files up to this point.
def clear_data_stores(begin_clicks: Optional[int]) -> str:
    """Clear all data stores before analysis begins.

    :param begin_clicks: Number of clicks on the begin analysis button.
    :returns: Empty string to clear notification.
    """
    #logger.info(
    #    f'Data cleared. Start clicks: {begin_clicks}: {datetime.now()}')
    return ''

@callback(
    Output('upload-data-file-success', 'style'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-data-table-info-data-store'}, 'data'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-data-table-data-store'}, 'data'),
    Output('input-warnings-data-table-div', 'children'),
    Output('input-warnings-data-table-div', 'hidden'),
    Input('upload-data-file', 'contents'),
    State('upload-data-file', 'filename'),
    State('upload-data-file', 'last_modified'),
    State('upload-data-file-success', 'style'),
    prevent_initial_call=True
)
def handle_uploaded_data_table(
    file_contents: Optional[str], 
    file_name: str, 
    mod_date: int, 
    current_upload_style: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], list, bool]:
    """Parse uploaded data table and populate stores.

    :param file_contents: Uploaded file contents.
    :param file_name: Uploaded filename.
    :param mod_date: File modification timestamp.
    :param current_upload_style: Upload success indicator style.
    :returns: Tuple of (upload style, data table info, data table contents, warnings, hide_warnings).
    """
    if file_contents is not None:
        upload_style, info, tables, warning_list = parsing.parse_data_file(
            file_contents, file_name, mod_date, current_upload_style, parameters['file loading']
        )
        warnings: list = [
            html.P(w) for w in warning_list
        ]
        if len(warnings) > 0:
            warnings.insert(0, html.H3('Data table warnings'))
            warnings.append(html.P('This might be due to file format. Supported formats are: csv (comma separated); tsv, txt, tab (tab separated); xlsx, xls (excel)'))
        return upload_style, info, tables, warnings, len(warnings)==0
    return no_update, no_update, no_update, no_update, no_update


@callback(
    Output('upload-sample_table-file-success', 'style'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-sample-table-info-data-store'}, 'data'),
    Output({'type': 'uploaded-data-store',
           'name': 'uploaded-sample-table-data-store'}, 'data'),
    Output('input-warnings-sample-table-div', 'children'),
    Output('input-warnings-sample-table-div', 'hidden'),
    Input('upload-sample_table-file', 'contents'),
    State('upload-sample_table-file', 'filename'),
    State('upload-sample_table-file', 'last_modified'),
    State('upload-sample_table-file-success', 'style'),
    prevent_initial_call=True
)
def handle_uploaded_sample_table(
    file_contents: Optional[str],
    file_name: str,
    mod_date: int,
    current_upload_style: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], list, bool]:
    """Parse uploaded sample table and populate stores.

    :param file_contents: Uploaded file contents.
    :param file_name: Uploaded filename.
    :param mod_date: File modification timestamp.
    :param current_upload_style: Upload success indicator style.
    :returns: Tuple of (upload style, sample table info, sample table contents, warnings, hide_warnings).
    """
    if file_contents is not None:
        upload_style, info, table_data = parsing.parse_sample_table(file_contents, file_name, mod_date, current_upload_style)
        exp_cols_found: list[str] = info['required columns found']
        warnings = []
        if len(exp_cols_found) < 2:
            req_cols: list[str] = ['sample name', 'sample group']
            fcols = ', '.join([info[col] for col in req_cols if col in info])
            warnings = [
                html.H3('Sample table warnings'),
                html.P(
                    f'- Experimental design table is missing required columns. Found columns: {fcols}, required columns: {", ".join(req_cols)}.'
                ),
                html.P('This might be due to file format. Supported formats are: csv (comma separated); tsv, txt, tab (tab separated); xlsx, xls (excel)')
            ]
        return upload_style, info, table_data, warnings, len(warnings)==0
    return no_update, no_update, no_update, no_update, no_update


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
    State('sidebar-options','value'),
    prevent_initial_call=True
)
def validate_data(
    _: str,
    data_tables: Dict[str, Any],
    data_info: Dict[str, Any],
    expdes_table: Dict[str, Any],
    expdes_info: Dict[str, Any],
    figure_template: str,
    additional_options: Optional[List[str]]
) -> Tuple[Dict[str, Any], bool, list[html.Div], bool]:
    """Validate and format uploaded data for analysis.

    :param _: Placeholder for start analysis notifier.
    :param data_tables: Uploaded data tables.
    :param data_info: Info about uploaded data tables.
    :param expdes_table: Experimental design table.
    :param expdes_info: Info about experimental design.
    :param figure_template: Selected figure template.
    :param additional_options: Selected additional processing options.
    :returns: Tuple of (data_dict, disable_download, warnings, hide_warnings).
    """
    logger.info(f'Validating data: {datetime.now()}')
    cont: List[str] = []
    repnames: bool = False
    uniq_only: bool = False
    if additional_options is not None:
        if 'Remove common contaminants' in additional_options:
            cont = contaminant_list
        if 'Rename replicates' in additional_options:
            repnames = True
        if 'Use unique proteins only (remove protein groups)' in additional_options:
            uniq_only = True
    pio.templates.default = remove_unwanted_colors(figure_template)
    data_dict: dict = parsing.format_data(
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}--{uuid4()}',
        data_tables,
        data_info,
        expdes_table,
        expdes_info,
        cont,
        repnames,
        uniq_only,
        parameters['workflow parameters']['interactomics']['control indicators'],
        parameters['file loading']['Bait ID column names']
    )

    return (
        data_dict,
        False
    )

@callback(
    Output({'type': 'data-store', 'name': 'upload-data-store'},
           'data', allow_duplicate=True),
    Input({'type': 'data-store', 'name': 'discard-samples-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def remove_samples(
    discard_samples_list: Optional[List[str]], 
    data_dictionary: Dict[str, Any]
) -> Dict[str, Any]:
    """Remove selected samples from the data dictionary.

    :param discard_samples_list: Sample names to remove.
    :param data_dictionary: Current data dictionary.
    :returns: Updated data dictionary without the selected samples.
    """
    return parsing.delete_samples(discard_samples_list, data_dictionary)


@callback(
    Output({'type': 'analysis-div', 'id': 'qc-analysis-area'}, 'children'),
    Output('discard-samples-div', 'hidden'),
    Output('workflow-specific-input-div', 'children',allow_duplicate = True),
    Output('workflow-specific-div', 'children',allow_duplicate = True),
    Input({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def create_qc_area(_: Dict[str, Any]) -> Tuple[html.Div, bool, str, str]:
    """Create the quality control area and show the discard button.

    :param _: Placeholder for replicate colors data store.
    :returns: Tuple of (QC area, discard button hidden flag, workflow input, workflow div).
    """
    return (ui.qc_area(), False,'','')


@callback(
    Output({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    Output({'type': 'data-store',
           'name': 'replicate-colors-with-contaminants-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def assign_replicate_colors(data_dictionary: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assign colors to sample replicates for visualization.

    :param data_dictionary: Data dictionary with sample groups.
    :returns: Tuple of (colors for samples, colors incl. contaminants).
    """
    return get_assigned_colors(data_dictionary['sample groups']['norm'])


@callback(
    Output('begin-analysis-button', 'disabled'),
    Input({'type': 'uploaded-data-store', 'name': 'uploaded-data-table-info-data-store'}, 'data'),
    Input({'type': 'uploaded-data-store', 'name': 'uploaded-sample-table-info-data-store'}, 'data'),
    Input('workflow-dropdown', 'value'), 
    Input('figure-theme-dropdown', 'value'),
    Input('upload-data-file-success', 'style'), 
    Input('upload-sample_table-file-success', 'style'),
    prevent_initial_call=True
)
def check_inputs(*args: Any) -> bool:
    """Validate that all required inputs are present before analysis.

    Returns True if inputs are invalid, to directly control the disabled state
    of the "Begin analysis" button.

    :param args: Data table info, sample table info, selected workflow, selected
        figure theme, and upload success styles.
    :returns: True if inputs are invalid, False if valid.
    """
    return parsing.validate_basic_inputs(*args)


@callback(
    Output('discard-sample-checklist-container', 'children'),
    Input('discard-samples-button', 'n_clicks'), 
    State({'type': 'qc-plot', 'id': 'count-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def open_discard_samples_modal(
    _: Optional[int], 
    count_plot: List[Any], 
    data_dictionary: Dict[str, Any]
) -> html.Div:
    """Create modal contents for selecting samples to discard.

    :param _: Number of clicks on discard samples button.
    :param count_plot: Current count plot components.
    :param data_dictionary: Data dictionary containing sample information.
    :returns: Checklist UI components for sample selection.
    """
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
def toggle_discard_modal(n1: Optional[int], n2: Optional[int], is_open: bool) -> bool:
    """Toggle visibility of the discard samples modal dialog.

    :param n1: Clicks on discard samples button.
    :param n2: Clicks on done discarding button.
    :param is_open: Current modal visibility state.
    :returns: New modal visibility state.
    """
    if (n1 > 0) or (n2 > 0):
        return not is_open
    return is_open


@callback(
    Output({'type': 'data-store', 'name': 'discard-samples-data-store'}, 'data'),
    Input('done-discarding-button', 'n_clicks'),
    State('checklist-select-samples-to-discard', 'value'),
    prevent_initial_call=True
)
def add_samples_to_discarded(n_clicks: Optional[int], chosen_samples: List[str]) -> Union[List[str], Any]:
    """Add selected samples to the list of discarded samples.

    :param n_clicks: Number of clicks on done discarding button.
    :param chosen_samples: Sample names selected for discarding.
    :returns: Updated list of discarded samples, or no_update if not triggered.
    """
    if n_clicks is None:
        return no_update
    if n_clicks < 1:
        return no_update
    return chosen_samples


@callback(
    Output({'type': 'qc-plot', 'id': 'tic-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'tic-data-store'}, 'data'),
    Input('qc-area', 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def parse_chromatogram_data(_: Any, data_dictionary: Dict[str, Any], replicate_colors: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate chromatogram plot data from sample information.

    :param _: Trigger placeholder from QC area.
    :param data_dictionary: Data dictionary containing sample information.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (TIC plot components, TIC data for storage).
    """
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
def plot_tic(chromatogram_data: Dict[str, Any], graph_type: str) -> go.Figure:
    """Create a chromatogram plot figure.

    :param chromatogram_data: Processed chromatogram data.
    :param graph_type: Type of chromatogram graph to display.
    :returns: Plotly Figure for the chromatogram plot.
    """
    return tic_graph.tic_figure(parameters['Figure defaults']['full-height'], chromatogram_data, graph_type)

@callback(
    Output({'type': 'qc-plot', 'id': 'count-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'count-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'tic-plot-div'}, 'children'), 
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-with-contaminants-data-store'}, 'data'),
    prevent_initial_call=True
)
def count_plot(
    _: Any, 
    data_dictionary: Dict[str, Any], 
    replicate_colors: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate protein count plot for samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :param replicate_colors: Color assignments for samples including contaminants.
    :returns: Tuple of (count plot components, count data for storage).
    """
    return qc_analysis.count_plot(
        data_dictionary['data tables']['with-contaminants'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        contaminant_list,
        parameters['Figure defaults']['full-height'],
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'common-protein-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'common-protein-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'count-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def common_proteins_plot(
    _: Any, 
    data_dictionary: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate plot showing common proteins across samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :returns: Tuple of (common proteins plot components, common proteins data).
    """
    return qc_analysis.common_proteins(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        db_file,
        parameters['Figure defaults']['full-height'],
        additional_groups = {
            'Other contaminants': contaminant_list
        },
        id_str = 'qc'
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'coverage-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'coverage-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'common-protein-plot-div'}, 'children'), 
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def coverage_plot(
    _: Any, 
    data_dictionary: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate protein coverage plot across samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :returns: Tuple of (coverage plot components, coverage data for storage).
    """
    return qc_analysis.coverage_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        parameters['Figure defaults']['half-height']
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'reproducibility-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'reproducibility-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'coverage-plot-div'}, 'children'), 
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def reproducibility_plot(
    _: Any, 
    data_dictionary: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate reproducibility plot between sample replicates.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :returns: Tuple of (reproducibility plot components, reproducibility data).
    """
    return qc_analysis.reproducibility_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        data_dictionary['sample groups']['norm'],
        data_dictionary['data tables']['table to use'],
        parameters['Figure defaults']['full-height']
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'missing-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'missing-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'reproducibility-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def missing_plot(
    _: Any, 
    data_dictionary: Dict[str, Any], 
    replicate_colors: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate missing values plot across samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (missing values plot components, missing values data).
    """
    return qc_analysis.missing_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'sum-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'sum-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'missing-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def sum_plot(
    _: Any, 
    data_dictionary: Dict[str, Any], 
    replicate_colors: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate plot showing sum of intensities across samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (sum plot components, sum data).
    """
    return qc_analysis.sum_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'mean-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'mean-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'sum-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def mean_plot(
    _: Any, 
    data_dictionary: Dict[str, Any], 
    replicate_colors: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate plot showing mean intensities across samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (mean plot components, mean data).
    """
    return qc_analysis.mean_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        parameters['Figure defaults']['half-height']
    )

@callback(
    Output({'type': 'qc-plot', 'id': 'distribution-plot-div'}, 'children'), 
    Output({'type': 'data-store', 'name': 'distribution-data-store'}, 'data'),
    Input({'type': 'qc-plot', 'id': 'mean-plot-div'}, 'children'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'), 
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def distribution_plot(
    _: Any, 
    data_dictionary: Dict[str, Any], 
    replicate_colors: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Generate plot showing distribution of intensities across samples.

    :param _: Trigger placeholder from upstream callback.
    :param data_dictionary: Data dictionary containing sample information.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (distribution plot components, distribution data).
    """
    return qc_analysis.distribution_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        replicate_colors,
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
        parsing.get_distribution_title(
            data_dictionary['data tables']['table to use'])
    )

@callback(
    Output({'type': 'data-store', 'name': 'qc-commonality-plot-visible-groups-data-store'}, 'data'),
    Input('qc-commonality-plot-update-plot-button','n_clicks'),
    State('qc-commonality-select-visible-sample-groups', 'value'),
)
def pass_selected_groups_to_data_store(
    _: Optional[int], 
    selection: List[str]
) -> Dict[str, List[str]]:
    """Store selected sample groups for commonality plot visibility.

    :param _: Clicks on update plot button.
    :param selection: Selected sample groups.
    :returns: Dict with selected groups for visibility.
    """
    return {'groups': selection}

@callback(
    Output({'type': 'qc-plot', 'id': 'commonality-plot-div'}, 'children'),
    Input({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
)
def generate_commonality_container(data_dictionary: Dict[str, Any]) -> html.Div:
    """Generate container for the commonality plot.

    :param data_dictionary: Data dictionary containing sample group information.
    :returns: Div container for commonality plot.
    """
    sample_groups = sorted(list(data_dictionary['sample groups']['norm'].keys()))
    return qc_analysis.generate_commonality_container(sample_groups)

@callback(
    Output('qc-commonality-graph-div','children'),
    Output({'type': 'data-store', 'name': 'commonality-data-store'}, 'data'),
    Output({'type': 'data-store', 'name': 'commonality-figure-pdf-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('toggle-additional-supervenn-options', 'value'),
    Input({'type': 'data-store', 'name': 'qc-commonality-plot-visible-groups-data-store'}, 'data'),
    prevent_initial_call=True
)
def commonality_plot(
    data_dictionary: Dict[str, Any], 
    additional_options: List[str], 
    show_only_groups: Optional[Dict[str, List[str]]]
) -> Tuple[html.Div, Dict[str, Any], Dict[str, Any]]:
    """Create a commonality plot showing protein overlap between groups.

    :param data_dictionary: Data dictionary containing sample information.
    :param additional_options: Selected additional plot options.
    :param show_only_groups: Optional dict specifying groups to display.
    :returns: Tuple of (plot components, plot data, PDF data).
    """
    show_groups: Union[str, List[str]] = None
    if show_only_groups is not None:
        show_groups = show_only_groups['groups']
    else:
        show_groups = 'all'
    return qc_analysis.commonality_plot(
        data_dictionary['data tables'][data_dictionary['data tables']['table to use']],
        data_dictionary['sample groups']['rev'],
        parameters['Figure defaults']['full-height'],
        ('Use supervenn' in additional_options), only_groups=show_groups
    )

@callback(
    Output('qc-done-notifier', 'children'),
    Input({'type': 'qc-plot', 'id': 'distribution-plot-div'},'children'),
    prevent_initial_call=True
)
def qc_done(_: Any) -> str:
    """Notify that QC analysis is complete.

    :param _: Trigger placeholder from distribution plot.
    :returns: Empty string to trigger completion notification.
    """
    return ''

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-na-filtered-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-na-filtered-data-store'}, 'data'),
    Input('proteomics-run-button', 'n_clicks'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State('proteomics-filter-minimum-percentage', 'value'),
    State('proteomics-filter-type', 'value'),
    prevent_initial_call=True
)
def proteomics_filtering_plot(nclicks: Optional[int], uploaded_data: Dict[str, Any], filtering_percentage: int, filter_type: str) -> Union[Tuple[html.Div, Dict[str, Any]], Tuple[Any, Any]]:
    """Create NA filtering results plot in proteomics workflow.

    :param nclicks: Number of clicks on run button.
    :param uploaded_data: Data dictionary containing proteomics data.
    :param filtering_percentage: Minimum percentage threshold for filtering.
    :param filter_type: Filter type ('sample-group' or 'sample-set').
    :returns: Tuple of (plot components, filtered data) or (no_update, no_update).
    """
    if nclicks is None:
        return (no_update, no_update)
    if nclicks < 1:
        return (no_update, no_update)
    return proteomics.na_filter(uploaded_data, filtering_percentage, parameters['Figure defaults']['full-height'], filter_type=filter_type)

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-normalization-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-normalization-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-na-filtered-data-store'}, 'data'),
    Input('proteomics-normalization-radio-option', 'value'),
    prevent_initial_call=True
)
def proteomics_normalization_plot(filtered_data: Optional[Dict[str, Any]], normalization_option: str) -> Union[Tuple[html.Div, Dict[str, Any]], Any]:
    """Create normalization results plot in proteomics workflow.

    :param filtered_data: NA-filtered proteomics data with intensities and metadata.
    :param normalization_option: Normalization method (e.g., 'median', 'quantile', 'vsn').
    :returns: Tuple of (plot components, normalized data) or no_update if no input.
    """
    if filtered_data is None:
        return no_update
    return proteomics.normalization(filtered_data, normalization_option, parameters['Figure defaults']['full-height'], parameters['Config']['R error file'])

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-missing-in-other-plot-div'}, 'children'),
    Input({'type': 'data-store', 'name': 'proteomics-normalization-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_missing_in_other_samples(normalized_data: Dict[str, Any]) -> html.Div:
    """Create plot showing missing values patterns after normalization.

    :param normalized_data: Normalized proteomics data with intensities and metadata.
    :returns: Div with plot components (half-height figure).
    """
    return proteomics.missing_values_in_other_samples(normalized_data, parameters['Figure defaults']['half-height'])

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-imputation-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-normalization-data-store'}, 'data'),
    Input('proteomics-imputation-radio-option', 'value'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_imputation_plot(normalized_data: Optional[Dict[str, Any]], imputation_option: str, data_dictionary: Dict[str, Any]) -> Union[Tuple[html.Div, Dict[str, Any]], Any]:
    """Create imputation results plot in proteomics workflow.

    :param normalized_data: Normalized proteomics data.
    :param imputation_option: Imputation method (e.g., 'minprob', 'minvalue', 'gaussian', 'qrilc', 'random_forest').
    :param data_dictionary: Full data dictionary (for sample group metadata).
    :returns: Tuple of (plot components, imputed data) or no_update if no input.
    """
    if normalized_data is None:
        return no_update
    return proteomics.imputation(
        normalized_data,
        imputation_option,
        parameters['Figure defaults']['full-height'],
        parameters['Config']['R error file'],
        sample_groups_rev=data_dictionary['sample groups']['rev']
        )



@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-cv-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-cv-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'},'data'),
    Input({'type': 'data-store', 'name': 'proteomics-na-filtered-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_cv_plot(uploaded_data: Dict[str, Any], na_filtered_data: Dict[str, Any], upload_dict: Dict[str, Any], replicate_colors: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Create coefficient of variation (CV) plot for proteomics data.

    :param uploaded_data: Original uploaded data dictionary with raw intensities.
    :param na_filtered_data: NA-filtered proteomics data for filtering raw data.
    :param upload_dict: Main data dictionary with sample grouping.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (CV plot components, CV analysis data).
    """
    return proteomics.perc_cvplot(uploaded_data['data tables']['raw intensity'], na_filtered_data, upload_dict['sample groups']['norm'], replicate_colors, parameters['Figure defaults']['full-height'])

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-pca-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-pca-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_pca_plot(imputed_data: Dict[str, Any], upload_dict: Dict[str, Any], replicate_colors: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Create PCA plot for proteomics data.

    :param imputed_data: Imputed proteomics data for PCA analysis.
    :param upload_dict: Data dictionary with sample grouping information.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (PCA plot components, PCA analysis data).
    """
    return proteomics.pca(imputed_data, upload_dict['sample groups']['rev'], parameters['Figure defaults']['full-height'], replicate_colors)

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-clustermap-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-clustermap-data-store'}, 'data'),
    Output({'type': 'done-notifier','name': 'proteomics-clustering-done-notifier'}, 'children', allow_duplicate=True),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_clustermap(imputed_data: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any], str]:
    """Create hierarchical clustering clustermap for proteomics data.

    :param imputed_data: Imputed proteomics data for clustering analysis.
    :returns: Tuple of (clustermap components, clustering data, completion notifier).
    """
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
def proteomics_check_comparison_table(
    contents: Optional[str], 
    filename: str, 
    current_style: Dict[str, str], 
    data_dictionary: Dict[str, Any]
) -> Tuple[Dict[str, str], Optional[Dict[str, Any]]]:
    """Validate uploaded comparison table for differential analysis.

    :param contents: Base64 contents of uploaded comparison file.
    :param filename: Uploaded filename.
    :param current_style: Current success indicator style.
    :param data_dictionary: Data dictionary with sample group information.
    :returns: Tuple of (updated style, parsed comparison data or None).
    """
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
    Input('proteomics-test-type', 'value'),
    prevent_initial_call=True
)
def proteomics_volcano_plots(
    imputed_data: Optional[Dict[str, Any]], 
    control_group: Optional[str], 
    comparison_data: Optional[Dict[str, Any]], 
    comparison_upload_success_style: Dict[str, str], 
    data_dictionary: Dict[str, Any], 
    fc_thr: float, 
    p_thr: float, 
    test_type: str
) -> Union[Tuple[html.Div, Dict[str, Any], str], Any]:
    """Create volcano plots for proteomics differential abundance analysis.

    :param imputed_data: Imputed proteomics data for analysis.
    :param control_group: Selected control group name for control-based comparisons.
    :param comparison_data: Comparison table data for custom comparisons.
    :param comparison_upload_success_style: Style indicating comparison table validation status.
    :param data_dictionary: Main data dictionary with sample information.
    :param fc_thr: Fold-change threshold for significance.
    :param p_thr: P-value threshold for significance.
    :param test_type: Statistical test type to use.
    :returns: Tuple of (volcano plot components, differential data, completion notifier) or no_update on invalid input.
    """
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
    sgroups: Dict[str, Any] = data_dictionary['sample groups']['norm']
    comparisons: List[Tuple[Any, ...]] = parsing.parse_comparisons(
        control_group, comparison_data, sgroups)
    return proteomics.differential_abundance(imputed_data, sgroups, comparisons, fc_thr, p_thr, parameters['Figure defaults']['full-height'], test_type, os.path.join(*parameters['Data paths']['Database file'])) + ('',)

# Need to implement:
# GOBP mapping


@callback(
    Output('interactomics-choose-uploaded-controls', 'value'),
    [Input('interactomics-select-all-uploaded', 'value')],
    [State('interactomics-choose-uploaded-controls', 'options')],
    prevent_initial_call=True
)
def select_all_none_controls(all_selected: bool, options: List[Dict[str, str]]) -> List[str]:
    """Select or deselect all uploaded control samples.

    :param all_selected: Whether the "select all" checkbox is checked.
    :param options: Available control sample options with 'value' keys.
    :returns: All values if selected, otherwise an empty list.
    """
    all_or_none: List[str] = [option['value'] for option in options if all_selected]
    return all_or_none

@callback(
    Output('interactomics-choose-enrichments', 'value'),
    [Input('interactomics-select-none-enrichments', 'n_clicks')],
    prevent_initial_call=True
)
def select_none_enrichments(deselect_click: Optional[int]) -> List[str]:
    """Deselect all enrichment options.

    :param deselect_click: Number of times the deselect button has been clicked.
    :returns: Empty list to clear all enrichment selections.
    """
    all_or_none: List[str] = []
    return all_or_none

@callback(
    Output('input-header', 'children'),
    Output('input-collapse','is_open'),
    Input('input-header','n_clicks'),
    Input('begin-analysis-button','n_clicks'),
    State('input-collapse','is_open'),
    prevent_initial_call=True
)
def collapse_or_uncollapse_input(
    header_click: Optional[int], 
    begin_click: Optional[int], 
    input_is_open: bool
) -> Tuple[str, bool]:
    """Toggle the collapse state of the input section.

    :param header_click: Number of clicks on the header (unused).
    :param begin_click: Number of clicks on the begin analysis button (unused).
    :param input_is_open: Current collapse state of the input section.
    :returns: Tuple of (new header text with arrow, new collapse state).
    """
    if input_is_open:
        return ('► Input', False)
    else:
        return ('▼ Input', True)


@callback(
    Output('interactomics-choose-additional-control-sets', 'value'),
    [Input('interactomics-select-all-inbuilt-controls', 'value')],
    [State('interactomics-choose-additional-control-sets', 'options')],
    prevent_initial_call=True
)
def select_all_none_inbuilt_controls(all_selected: bool, options: List[Dict[str, str]]) -> List[str]:
    """Select or deselect all inbuilt control sets.

    :param all_selected: Whether the "select all" checkbox is checked.
    :param options: Available inbuilt control set options with 'value' keys.
    :returns: All values if selected, otherwise an empty list.
    """
    all_or_none: List[str] = [option['value'] for option in options if all_selected]
    return all_or_none


@callback(
    Output('interactomics-choose-crapome-sets', 'value'),
    [Input('interactomics-select-all-crapomes', 'value')],
    [State('interactomics-choose-crapome-sets', 'options')],
    prevent_initial_call=True
)
def select_all_none_crapomes(all_selected: bool, options: List[Dict[str, str]]) -> List[str]:
    """Select or deselect all CRAPome control sets.

    :param all_selected: Whether the "select all" checkbox is checked.
    :param options: Available CRAPome control set options with 'value' keys.
    :returns: All values if selected, otherwise an empty list.
    """
    all_or_none: List[str] = [option['value'] for option in options if all_selected]
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
def interactomics_saint_analysis(
    nclicks: Optional[int], 
    uploaded_controls: List[str], 
    additional_controls: List[str], 
    crapomes: List[str], 
    uploaded_data: Dict[str, Any], 
    proximity_filtering_checklist: List[str], 
    n_controls: int
) -> Union[Tuple[html.Div, Dict[str, Any], Dict[str, Any]], Tuple[Any, Any, Any]]:
    """Initialize SAINT analysis with selected control samples and parameters.

    :param nclicks: Number of clicks on run analysis button.
    :param uploaded_controls: Selected uploaded control samples.
    :param additional_controls: Selected additional control sets.
    :param crapomes: Selected CRAPome control sets.
    :param uploaded_data: Uploaded experimental data dictionary.
    :param proximity_filtering_checklist: Selected filtering options.
    :param n_controls: Number of nearest controls if proximity filtering enabled.
    :returns: Tuple of (SAINT container Div, SAINT input data, CRAPome data) or no_update if not triggered.
    """
    if nclicks is None:
        return (no_update, no_update, no_update)
    if nclicks < 1:
        return (no_update, no_update, no_update)
    do_proximity_filtering: bool = ('Select' in proximity_filtering_checklist)
    return interactomics.generate_saint_container(uploaded_data, uploaded_controls, additional_controls, crapomes, db_file, do_proximity_filtering, n_controls)

@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-output-data-store'}, 'data'),
    Output('interactomics-saint-has-error','children'),
    Output('interactomics-saint-running-loading', 'children'),
    Input({'type': 'data-store', 'name': 'interactomics-saint-input-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True,
    background=True
)
def interactomics_run_saint(
    saint_input: Dict[str, Any], 
    data_dictionary: Dict[str, Any]
) -> Tuple[Union[Dict[str, Any], str], str, str]:
    """Execute SAINT analysis using prepared input data.

    :param saint_input: Prepared SAINT input data.
    :param data_dictionary: Data dictionary with session and bait information.
    :returns: Tuple of (SAINT output data or error string, error message, empty string).
    """
    saint_data, saint_not_found = interactomics.run_saint(
        saint_input,
        parameters['External tools']['SAINT tempdir'],
        data_dictionary['other']['session name'],
        data_dictionary['other']['bait uniprots']
    )
    sainterr = ''
    if saint_not_found:
        sainterr = 'SAINT executable was not found, scoring data is randomized'
    return (saint_data, sainterr, '')

@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'interactomics-saint-output-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-crapome-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_add_crapome_to_saint(
    saint_output: Union[Dict[str, Any], str], 
    crapome: Dict[str, Any]
) -> Union[Dict[str, Any], str]:
    """Integrate CRAPome data with SAINT analysis results.

    :param saint_output: Results from SAINT analysis.
    :param crapome: CRAPome control data to integrate.
    :returns: Combined SAINT and CRAPome data, or error string if SAINT failed.
    """
    if saint_output == 'SAINT failed. Can not proceed.':
        return saint_output
    return interactomics.add_crapome(saint_output, crapome)

@callback(
    Output('workflow-done-notifier', 'children', allow_duplicate=True),
    Output('interactomics-saint-filtering-container', 'children'),
    Input({'type': 'data-store',
          'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    State('interactomics-saint-has-error', 'children'),
    State('interactomics-rescue-filtered-out', 'value'),
    prevent_initial_call=True
)
def interactomics_create_saint_filtering_container(
    saint_output_ready: Union[Dict[str, Any], str], 
    saint_not_found: str,
    rescue: List[str]
) -> Tuple[str, html.Div]:
    """Create the filtering container for SAINT analysis results.

    :param saint_output_ready: Final SAINT analysis output data.
    :param saint_not_found: Error message if SAINT executable was not found.
    :param rescue: Selected rescue options.
    :returns: Tuple of (completion notifier, filtering container Div).
    """
    rescue_bool: bool = ('Rescue interactions that pass filter in any sample group' in rescue)
    saint_found: bool = len(saint_not_found) == 0
    if 'SAINT failed.' in saint_output_ready:
        return ('',html.Div(id='saint-failed', children=saint_output_ready))
    else:
        return ('',ui.saint_filtering_container(parameters['Figure defaults']['half-height'], rescue_bool, saint_found))

@callback(
    Output('interactomics-saint-bfdr-histogram', 'figure'),
    Output({'type': 'data-store',
           'name': 'interactomics-saint-bfdr-histogram-data-store'}, 'data'),
    Input({'type': 'input-div', 'id': 'interactomics-saint-filtering-area'}, 'children'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-final-output-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data')
)
def interactomics_draw_saint_histogram(
    container_ready: List[Any], 
    saint_output: str, 
    saint_output_filtered: Optional[str]
) -> Tuple[go.Figure, Dict[str, Any]]:
    """Generate histogram visualization of SAINT BFDR scores.

    :param container_ready: Trigger indicating the filtering container is ready.
    :param saint_output: Original SAINT analysis results.
    :param saint_output_filtered: Filtered SAINT results, if available.
    :returns: Tuple of (BFDR histogram Figure, histogram data JSON).
    """
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
)
def interactomics_apply_saint_filtering(
    bfdr_threshold: float, 
    crapome_percentage: int, 
    crapome_fc: int, 
    saint_output: str, 
    rescue: List[str]
) -> Dict[str, Any]:
    """Apply filtering criteria to SAINT analysis results.

    :param bfdr_threshold: BFDR score threshold.
    :param crapome_percentage: CRAPome frequency threshold.
    :param crapome_fc: CRAPome fold-change threshold.
    :param saint_output: SAINT results to filter.
    :param rescue: Selected rescue options.
    :returns: Filtered SAINT results.
    """
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
def interactomics_draw_saint_filtered_figure(filtered_output: Dict[str, Any], replicate_colors: Dict[str, Any]) -> Tuple[go.Figure, Dict[str, Any]]:
    """Create visualization of filtered SAINT analysis results.

    :param filtered_output: Filtered SAINT analysis results.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (Figure for filtered results, graph data JSON).
    """
    return interactomics.saint_counts(filtered_output, parameters['Figure defaults']['half-height'], replicate_colors)

@callback(
    Output({'type': 'analysis-div',
           'id': 'interactomics-analysis-post-saint-area'}, 'children'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    prevent_initial_call=True
)
def interactomics_initiate_post_saint(_: Optional[int]) -> html.Div:
    """Initialize the post-SAINT analysis interface container.

    :param _: Clicks on done filtering button (unused).
    :returns: Div containing post-SAINT analysis interface.
    """
    return ui.post_saint_container()

@callback(
    Output({'type': 'data-store',
           'name': 'interactomics-saint-filtered-and-intensity-mapped-output-data-store'}, 'data'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_callback=True
)
def interactomics_map_intensity(n_clicks: Optional[int], filtered_saint_data: Dict[str, Any], data_dictionary: Dict[str, Any]) -> Union[str, Any]:
    """Map averaged intensity to filtered SAINT results.

    :param n_clicks: Clicks on done filtering button.
    :param filtered_saint_data: Filtered SAINT results.
    :param data_dictionary: Data dictionary with intensity values and sample groups.
    :returns: JSON string of SAINT results with mapped intensity, or no_update if not triggered.
    """
    if (n_clicks is None):
        return no_update
    if (n_clicks < 1):
        return no_update
    return interactomics.map_intensity(filtered_saint_data, data_dictionary['data tables']['intensity'], data_dictionary['sample groups']['norm'])

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
def interactomics_known_plot(saint_output: Dict[str, Any], rep_colors_with_cont: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Create plot showing known interactions in SAINT results.

    :param saint_output: SAINT results with mapped intensity values.
    :param rep_colors_with_cont: Color assignments for samples including contaminants.
    :returns: Tuple of (plot components, known interactions data).
    """
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
def interactomics_common_proteins_plot(
    _: Dict[str, Any], 
    saint_data: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Create plot showing common proteins across SAINT results.

    :param _: Trigger input from known interactions plot (unused).
    :param saint_data: Filtered SAINT analysis results.
    :returns: Tuple of (common proteins plot components, data for storage).
    """
    saint_data = interactomics.get_saint_matrix(saint_data)
    return qc_analysis.common_proteins(
        saint_data.to_json(orient='split'),
        db_file,
        parameters['Figure defaults']['full-height'],
        additional_groups = {
            'Other contaminants': contaminant_list
        },
        id_str='interactomics'
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
def interactomics_pca_plot(
    _: Dict[str, Any], 
    saint_data: Dict[str, Any], 
    replicate_colors: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Create a PCA plot for interactomics data.

    :param _: Trigger input from common proteins plot (unused).
    :param saint_data: Filtered SAINT analysis results.
    :param replicate_colors: Color assignments for sample replicates.
    :returns: Tuple of (PCA plot components, PCA data).
    """
    return interactomics.pca(
        saint_data,
        parameters['Figure defaults']['full-height'],
        replicate_colors
    )

@callback(
    Output('interactomics-msmic-loading', 'children'),
    Output({'type': 'data-store',
           'name': 'interactomics-msmic-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'interactomics-pca-data-store'}, 'data'),
    State({'type': 'data-store',
          'name': 'interactomics-saint-filtered-output-data-store'}, 'data'),
    prevent_initial_call=True
)
def interactomics_ms_microscopy_plots(
    _: Dict[str, Any], 
    saint_output: Dict[str, Any]
) -> Tuple[html.Div, Dict[str, Any]]:
    """Create MS microscopy visualization plots.

    :param _: Trigger input from PCA plot (unused).
    :param saint_output: Filtered SAINT analysis results.
    :returns: Tuple of (MS microscopy plot components, analysis data).
    """
    res = interactomics.do_ms_microscopy(saint_output, db_file, 
                                       parameters['Figure defaults']['full-height'], 
                                       version='v1.0')

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
def interactomics_network_plot(
    _: Dict[str, Any], 
    saint_output: Dict[str, Any]
) -> Tuple[str, html.Div, Dict[str, Any], Dict[str, Any]]:
    """Create interactive network visualization of protein interactions.

    :param _: Trigger input from MS microscopy plot (unused).
    :param saint_output: Filtered SAINT analysis results.
    :returns: Tuple of (completion notifier, network container, elements, interactions).
    """
    container, c_elements, interactions = interactomics.do_network(
        saint_output, 
        parameters['Figure defaults']['full-height']['height']
    )
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
def interactomics_enrichment(
    _: Dict[str, Any], 
    saint_output: Dict[str, Any], 
    chosen_enrichments: List[str]
) -> Tuple[str, html.Div, Dict[str, Any], Dict[str, Any]]:
    """Perform enrichment analysis on filtered interactomics data.

    :param _: Trigger input from network plot (unused).
    :param saint_output: Filtered SAINT analysis results.
    :param chosen_enrichments: Selected enrichment analyses to perform.
    :returns: Tuple of (completion notifier, enrichment plots, results data, info data).
    """
    return ('',) + interactomics.enrich(
        saint_output, 
        chosen_enrichments, 
        parameters['Figure defaults']['full-height'],
        parameters_file=parameters_file
    )

########################################
# Interactomics network plot callbacks #
########################################
## Visdcc package is currently unused, because it's impossible to export anything sensible out of it.
## Export would require javascript, but there is no time to do any of this.
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
def display_tap_node(node_data: Optional[Dict[str, Any]], int_data: Dict[str, Any], network_type: str = 'Cytoscape') -> Optional[html.Div]:
    """Display detailed information for a selected node in the network visualization.

    :param node_data: Data associated with the tapped network node.
    :param int_data: Network interaction data store.
    :param network_type: Network visualization type (default 'Cytoscape').
    :returns: Div with node connection details, or None if not selected.
    """
    if not node_data:
        return None
    if network_type == 'Cytoscape':
        ret = interactomics.network_display_data(
            node_data,
            int_data,
            parameters['Figure defaults']['full-height']['height']
        )
    return ret

@callback(
    Output("cytoscape", "layout"),
    Input("dropdown-layout", "value")
)
def update_cytoscape_layout(layout: str) -> Dict[str, Any]:
    """Update Cytoscape network layout configuration.

    :param layout: Selected layout type from dropdown.
    :returns: Layout configuration dict including extra parameters if defined.
    """
    ret_dic: Dict[str, Any] = {"name": layout}
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
def table_of_contents(
    _: Any, 
    __: Any, 
    ___: Any, 
    ____: Any, 
    main_div_contents: List[Any]
) -> html.Div:
    """Update the table of contents based on main content.

    :param _: Trigger input (unused).
    :param __: Trigger input (unused).
    :param ___: Trigger input (unused).
    :param ____: Trigger input (unused).
    :param main_div_contents: Current contents of main div.
    :returns: Updated table of contents component.
    """
    return ui.table_of_contents(main_div_contents)

@callback(
    Output('workflow-specific-input-div', 'children',allow_duplicate = True),
    Output('workflow-specific-div', 'children',allow_duplicate = True),
    Input('qc-done-notifier', 'children'),
    State('workflow-dropdown', 'value'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True,
)
def workflow_area(
    _: Any, 
    workflow: str, 
    data_dictionary: Dict[str, Any]
) -> Tuple[html.Div, html.Div]:
    """Update workflow-specific areas based on selected workflow.

    :param _: Trigger input from QC completion notifier (unused).
    :param workflow: Selected workflow type.
    :param data_dictionary: Data dictionary containing uploaded data.
    :returns: Tuple of (workflow input area, workflow content area).
    """
    return ui.workflow_area(workflow, parameters['workflow parameters'], data_dictionary)


@callback(
    Output('download-proteomics-comparison-example', 'data'),
    Input('download-proteomics-comparison-example-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_example_comparison_file(n_clicks: Optional[int]) -> Optional[Dict[str, Any]]:
    """Provide example proteomics comparison file for download.

    :param n_clicks: Number of clicks on the download button.
    :returns: dcc.send_file payload for the example file, or None if not triggered.
    """
    if n_clicks is None:
        return None
    if n_clicks == 0:
        return None
    return dcc.send_file(os.path.join(*parameters['Data paths']['Example proteomics comparison file']))

@callback(
    Output('download-example-files', 'data'),
    Input('button-download-example-files', 'n_clicks'),
    prevent_initial_call=True,
)
def example_files_download(_: Optional[int]) -> Dict[str, Any]:
    """Provide example files bundle for download.

    :param _: Number of clicks on download button (unused).
    :returns: dcc.send_file payload for the zipped example files.
    """
    return dcc.send_file(utils.zipdir(os.path.join(*parameters['Data paths']['Example files'])))

def get_adiv_by_id(
    divs: List[Any], 
    idvals: List[Dict[str, str]], 
    idval_to_find: str
) -> Optional[Any]:
    """Retrieve a specific div element by matching its ID.

    :param divs: List of div elements.
    :param idvals: List of dicts containing ID values (with key 'id').
    :param idval_to_find: Target ID value to search for.
    :returns: Matching div element if found, otherwise None.
    """
    use_index: int = -1
    for i, idval in enumerate(idvals):
        if idval['id'] == idval_to_find:
            use_index = i
            break
    if use_index > -1:
        return divs[use_index]
    return None

##################################
##   Start of export section    ##
##################################
# Export needed to be split apart due to taking too long otherwise with background callbacks.
# Background callbacks were disabled due to some weird-ass bug that had something to do with volcano plots and excessive numbers of differentially abundant proteins.
# These could now be merged back into one, I guess

@callback(
    Output('download-temp-dir-ready','children'),
    Output('button-download-all-data-text','children',allow_duplicate = True),
    Input('button-download-all-data', 'n_clicks'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    prevent_initial_call=True
)
def prepare_for_download(
    _: Optional[int], 
    main_data: Dict[str, Any]
) -> Tuple[str, html.Div]:
    """Prepare directory structure and README for data export.

    :param _: Clicks on download button (unused).
    :param main_data: Data dictionary containing session information.
    :returns: Tuple of (export directory path, temporary loading indicators Div).
    """
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H-%M")
    export_dir: str = os.path.join(*parameters['Data paths']['Cache dir'],  
                                 main_data['other']['session name'], 
                                 f'{timestamp} Proteogyver output')
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    return export_dir, infra.temporary_download_button_loading_divs()

@callback(
    Output('download_temp1', 'children'),
    Output('download_loading_temp1', 'children'),
    Input('download-temp-dir-ready','children'),
    State('input-stores', 'children'),
    prevent_initial_call=True
)
def save_input_stores(export_dir: str, stores: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Save input data stores to export directory.

    :param export_dir: Destination directory.
    :param stores: Input data stores to save.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_input_stores at {start}')
    infra.write_README(export_dir, os.path.join('data','output_guide.md'))
    infra.save_data_stores(stores, export_dir)
    logger.info(f'done with download request save_input_stores, took {datetime.now()-start}')
    return 'save_input_stores done', ''

@callback(
    Output('download_temp2', 'children'),
    Output('download_loading_temp2', 'children'),
    Input('download-temp-dir-ready','children'),
    State('workflow-stores', 'children'),
    prevent_initial_call=True
)
def save_workflow_stores(export_dir: str, stores: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Save workflow data stores to export directory.

    :param export_dir: Destination directory.
    :param stores: Workflow data stores to save.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_workflow_stores at {start}')
    infra.save_data_stores(stores, export_dir)
    logger.info(f'done with download request save_workflow_stores, took {datetime.now()-start}')
    return 'save_workflow_stores done', ''

@callback(
    Output('download_temp3', 'children'),
    Output('download_loading_temp3', 'children'),
    Input('download-temp-dir-ready','children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State({'type': 'analysis-div', 'id': ALL}, 'id'),
    State({'type': 'data-store', 'name': 'commonality-figure-pdf-data-store'}, 'data'),
    State('workflow-dropdown', 'value'),
    prevent_initial_call=True
)
def save_qc_figures(
    export_dir: str, 
    analysis_divs: List[html.Div], 
    analysis_div_ids: List[Dict[str, str]], 
    commonality_pdf_data: Optional[Dict[str, Any]], 
    workflow: str
) -> Tuple[str, str]:
    """Save quality control figures to export directory.

    :param export_dir: Destination directory.
    :param analysis_divs: Analysis div elements.
    :param analysis_div_ids: IDs of analysis divs.
    :param commonality_pdf_data: Optional PDF data for commonality figures.
    :param workflow: Current workflow name.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_qc_figures at {start}')
    try:
        infra.save_figures([get_adiv_by_id(analysis_divs, analysis_div_ids, 'qc-analysis-area')], 
                          export_dir,
                          figure_output_formats, 
                          commonality_pdf_data, 
                          workflow)
    except Exception as e:
        logger.warning(f'save_qc_figures failed: {e}')
        with open(os.path.join(export_dir, 'save_qc_figures_errors'),'w') as fil:
            fil.write(f'{e}')
    logger.info(f'done with download request save_qc_figures, took {datetime.now()-start}')
    return 'save_qc_figures done', ''

@callback(
    Output('download_temp4', 'children'),
    Output('download_loading_temp4', 'children'),
    Input('download-temp-dir-ready','children'),
    State({'type': 'input-div', 'id': ALL}, 'children'),
    prevent_initial_call=True
)
def save_input_information(export_dir: str, input_divs: List[html.Div]) -> Tuple[str, str]:
    """Save input information to export directory.

    :param export_dir: Destination directory.
    :param input_divs: Input div elements to inspect.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_input_information at {start}')
    infra.save_input_information(input_divs, export_dir)
    logger.info(f'done with download request save_input_information, took {datetime.now()-start}')
    return 'save_input_information done',''

@callback(
    Output('download_temp5', 'children'),
    Output('download_loading_temp5', 'children'),
    Input('download-temp-dir-ready','children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State({'type': 'analysis-div', 'id': ALL}, 'id'),
    State('workflow-dropdown', 'value'),
    prevent_initial_call=True
)
def save_interactomics_figures(
    export_dir: str, 
    analysis_divs: List[html.Div], 
    analysis_div_ids: List[Dict[str, str]], 
    workflow: str
) -> Tuple[str, str]:
    """Save interactomics analysis figures to export directory.

    :param export_dir: Destination directory.
    :param analysis_divs: Analysis div elements.
    :param analysis_div_ids: IDs of analysis divs.
    :param workflow: Current workflow name.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_interactomics_figures at {start}')
    try:
        infra.save_figures([get_adiv_by_id(analysis_divs, analysis_div_ids, 'interactomics-analysis-results-area')], export_dir,
                    figure_output_formats, None, workflow)
    except Exception as e:
        logger.warning(f'save_interactomics_figures failed: {e}')
        tb = traceback.format_exc()
        logger.warning(f"save_interactomics_figures failed: {e}\n{tb}")
        with open(os.path.join(export_dir, "save_interactomics_figures_errors.txt"), "w") as fil:
            fil.write(f"save_interactomics_figures failed: {e}\n{tb}")
    logger.info(f'done with download request save_interactomics_figures, took {datetime.now()-start}')
    return 'save_interactomics_figures done', ''

@callback(
    Output('download_temp6', 'children'),
    Output('download_loading_temp6', 'children'),
    Input('download-temp-dir-ready','children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State({'type': 'analysis-div', 'id': ALL}, 'id'),
    State('workflow-dropdown', 'value'),
    prevent_initial_call=True
)
def save_interactomics_post_saint_figures(
    export_dir: str, 
    analysis_divs: List[html.Div], 
    analysis_div_ids: List[Dict[str, str]], 
    workflow: str
) -> Tuple[str, str]:
    """Save post-SAINT interactomics figures to export directory.

    :param export_dir: Destination directory.
    :param analysis_divs: Analysis div elements.
    :param analysis_div_ids: IDs of analysis divs.
    :param workflow: Current workflow name.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_interactomics_post_saint_figures at {start}')
    try:
        infra.save_figures([get_adiv_by_id(analysis_divs, analysis_div_ids, 'interactomics-analysis-post-saint-area')], 
                          export_dir, figure_output_formats, None, workflow)
    except Exception as e:
        logger.warning(f'save_interactomics_post_saint_figures failed: {e}')
        with open(os.path.join(export_dir, 'save_interactomics_post_saint_figures_errors'),'w') as fil:
            fil.write(f'{e}')
    logger.info(f'done with download request save_interactomics_post_saint_figures, took {datetime.now()-start}')
    return 'save_interactomics_post_saint_figures done', ''

@callback(
    Output('download_temp7', 'children'),
    Output('download_loading_temp7', 'children'),
    Input('download-temp-dir-ready','children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State({'type': 'analysis-div', 'id': ALL}, 'id'),
    State('workflow-dropdown', 'value'),
    prevent_initial_call=True
)
def save_proteomics_figures(
    export_dir: str, 
    analysis_divs: List[html.Div], 
    analysis_div_ids: List[Dict[str, str]], 
    workflow: str
) -> Tuple[str, str]:
    """Save proteomics analysis figures to export directory.

    :param export_dir: Destination directory.
    :param analysis_divs: Analysis div elements.
    :param analysis_div_ids: IDs of analysis divs.
    :param workflow: Current workflow name.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_proteomics_figures at {start}')
    try:
        infra.save_figures([get_adiv_by_id(analysis_divs, analysis_div_ids, 'proteomics-analysis-results-area')], 
                          export_dir, figure_output_formats, None, workflow)
    except Exception as e:
        logger.warning(f'save_proteomics_figures failed: {e}')
        with open(os.path.join(export_dir, 'save_proteomics_figures_errors'),'w') as fil:
            fil.write(f'{e}')
    logger.info(f'done with download request save_proteomics_figures, took {datetime.now()-start}')
    return 'save_proteomics_figures done', ''

@callback(
    Output('download_temp8', 'children'),
    Output('download_loading_temp8', 'children'),
    Input('download-temp-dir-ready','children'),
    State({'type': 'analysis-div', 'id': ALL}, 'children'),
    State({'type': 'analysis-div', 'id': ALL}, 'id'),
    State('workflow-dropdown', 'value'),
    prevent_initial_call=True
)
def save_phosphoproteomics_figures(
    export_dir: str, 
    analysis_divs: List[html.Div], 
    analysis_div_ids: List[Dict[str, str]], 
    workflow: str
) -> Tuple[str, str]:
    """Save phosphoproteomics analysis figures to export directory.

    :param export_dir: Destination directory.
    :param analysis_divs: Analysis div elements.
    :param analysis_div_ids: IDs of analysis divs.
    :param workflow: Current workflow name.
    :returns: Tuple of (completion message, loading indicator text).
    """
    start = datetime.now()
    logger.info(f'received download request save_phosphoproteomics_figures at {start}')
    try:
        infra.save_figures([get_adiv_by_id(analysis_divs, analysis_div_ids, 'phosphoproteomics-analysis-area')], 
                          export_dir, figure_output_formats, None, workflow)
    except Exception as e:
        logger.warning(f'save_phosphoproteomics_figures failed: {e}')
        with open(os.path.join(export_dir, 'save_phosphoproteomics_figures_errors'),'w') as fil:
            fil.write(f'{e}')
    logger.info(f'done with download request save_phosphoproteomics_figures, took {datetime.now()-start}')
    return 'save_phosphoproteomics_figures done', ''

    
@callback(
    Output('download-all-data', 'data'),
    Output('button-download-all-data-text','children', allow_duplicate=True),
    Input('download-temp-dir-ready','children'),
    Input('download_temp1', 'children'),
    Input('download_temp2', 'children'),
    Input('download_temp3', 'children'),
    Input('download_temp4', 'children'),
    Input('download_temp5', 'children'),
    Input('download_temp6', 'children'),
    Input('download_temp7', 'children'),
    Input('download_temp8', 'children'),
    prevent_initial_call=True
)
def send_data(export_dir: str, *args: str) -> Union[Tuple[Dict[str, Any], str], Tuple[Any, Any]]:
    """Create and send a ZIP archive containing all exported analysis data.

    :param export_dir: Temporary export directory path.
    :param args: Completion tokens from prior export steps (must include 'done').
    :returns: Tuple of (send_bytes payload, updated button text) or no_update while incomplete.
    """
    # Verify all export steps are complete
    for a in args:
        if not 'done' in a:
            return no_update, no_update
            
    start = datetime.now()    
    timestamp = start.strftime("%Y-%m-%d %H-%M")
    zip_filename = f"{timestamp} ProteoGyver output.zip"
    logger.info(f'Started packing data at {start}')
    
    try:
        # Create ZIP archive
        with zipfile.ZipFile(os.path.join(export_dir, zip_filename), 'w') as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    if file != zip_filename:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_dir)
                        zipf.write(file_path, arcname)
        logger.info(f'done packing data, took {datetime.now()-start}')
        
        # Read ZIP file and encode for download
        with open(os.path.join(export_dir, zip_filename), 'rb') as f:
            zip_data = f.read()
        
        # Clean up temporary files
        shutil.rmtree(export_dir)
    
    except Exception as e:
        logger.warning(f"Error creating download package: {str(e)}")
        return no_update, no_update

    return dcc.send_bytes(zip_data, zip_filename), 'Download all data'
##################################
##    End of export section     ##
##################################
