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
from io import StringIO
import os
import shutil
import zipfile
import markdown
import pandas as pd
from uuid import uuid4
from datetime import datetime
from dash import html, callback, no_update, ALL, dcc, register_page
from dash.dependencies import Input, Output, State
from components import ui_components as ui
from components import infra
from components import parsing, qc_analysis, proteomics, interactomics, db_functions
from components.figures.color_tools import get_assigned_colors
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

parameters_file = 'parameters.toml'
parameters: Dict[str, Any] = parsing.parse_parameters(parameters_file)
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
    """Clears all data stores before analysis begins.
    
    Args:
        begin_clicks (int): Number of times the begin analysis button has been clicked
        
    Returns:
        str: Empty string to clear notification
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
    current_upload_style: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Parses uploaded data table and sends data to data stores.
    
    Args:
        file_contents: Contents of the uploaded file
        file_name (str): Name of the uploaded file
        mod_date: Last modified date of the file
        current_upload_style (dict): Current style of the upload success indicator
        
    Returns:
        tuple: Contains:
            - Updated upload success style
            - Data table info for storage
            - Data table contents for storage
    """
    if file_contents is not None:
        return parsing.parse_data_file(
            file_contents, file_name, mod_date, current_upload_style, parameters['file loading']
        )
    return no_update, no_update, no_update


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
def handle_uploaded_sample_table(
    file_contents: Optional[str],
    file_name: str,
    mod_date: int,
    current_upload_style: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Parses uploaded sample table and sends data to data stores.
    
    Args:
        file_contents: Contents of the uploaded file
        file_name (str): Name of the uploaded file
        mod_date: Last modified date of the file
        current_upload_style (dict): Current style of the upload success indicator
        
    Returns:
        tuple: Contains:
            - Updated upload success style
            - Sample table info for storage
            - Sample table contents for storage
    """
    if file_contents is not None:
        return parsing.parse_sample_table(file_contents, file_name, mod_date, current_upload_style)
    return no_update, no_update, no_update


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
) -> Tuple[Dict[str, Any], bool]:
    """Validates and formats uploaded data for analysis.
    
    Args:
        _ (str): Placeholder for start analysis notifier
        data_tables: Uploaded data tables
        data_info: Information about uploaded data tables
        expdes_table: Experimental design table
        expdes_info: Information about experimental design
        figure_template (str): Selected figure template
        additional_options (list): Selected additional processing options
        
    Returns:
        tuple: Contains:
            - Formatted data dictionary
            - Boolean indicating if download button should be disabled
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
    pio.templates.default = figure_template
    return (
        parsing.format_data(
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
        ), 
        False)

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
    """Removes selected samples from the data dictionary.
    
    Args:
        discard_samples_list (list): List of sample names to remove
        data_dictionary (dict): Current data dictionary containing all samples
        
    Returns:
        dict: Updated data dictionary with selected samples removed
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
    """Creates the quality control analysis area and shows sample discard button.
    
    Args:
        _ (dict): Placeholder for replicate colors data store
        
    Returns:
        tuple: Contains:
            - QC area UI components
            - Boolean for sample discard button visibility
            - Empty string for workflow input div
            - Empty string for workflow div
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
    """Assigns colors to sample replicates for visualization.
    
    Args:
        data_dictionary (dict): Data dictionary containing sample information
        
    Returns:
        dict: Two color dictionaries:
            - One for regular samples
            - One including contaminant colors
    """
    return get_assigned_colors(data_dictionary['sample groups']['norm'])


@callback(
    Output('begin-analysis-button', 'disabled'),
    Input({'type': 'uploaded-data-store', 'name': 'uploaded-data-table-info-data-store'}, 'data'),
    Input({'type': 'uploaded-data-store', 'name': 'uploaded-sample-table-info-data-store'}, 'data'),
    Input('workflow-dropdown', 'value'), 
    Input('figure-theme-dropdown', 'value'),
    Input('upload-data-file-success', 'style'), 
    Input('upload-data-file-success', 'style'),
    prevent_initial_call=True
)
def check_inputs(*args: Any) -> bool:
    """Validates that all required inputs are present before analysis can begin.

    Returns True, if invalid so that the value can be used directly as input for dis/abling the begin analysis button.
    
    Args:
        *args: Variable length argument list containing:
            - Data table info
            - Sample table info
            - Selected workflow
            - Selected figure theme
            - Upload success styles
            
    Returns:
        bool: True if inputs are invalid, False if valid
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
    """Creates modal dialog for selecting samples to discard.
    
    Args:
        _ (int): Number of clicks on discard samples button
        count_plot (list): Current count plot components
        data_dictionary (dict): Data dictionary containing sample information
        
    Returns:
        tuple: Contains checklist UI components for sample selection
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
    """Toggles visibility of the discard samples modal dialog.
    
    Args:
        n1 (int): Number of clicks on discard samples button
        n2 (int): Number of clicks on done discarding button
        is_open (bool): Current modal visibility state
        
    Returns:
        bool: New modal visibility state
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
    """Adds selected samples to the list of discarded samples.
    
    Args:
        n_clicks (int): Number of clicks on done discarding button
        chosen_samples (list): List of sample names selected for discarding
        
    Returns:
        list: Updated list of discarded samples or no update if conditions not met
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
    """Generates chromatogram plot data from sample information.
    
    Args:
        _ (list): Placeholder for QC area children
        data_dictionary (dict): Data dictionary containing sample information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - TIC plot components
            - TIC data for storage
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
    """Creates chromatogram plot figure.
    
    Args:
        tic_data (dict): Processed chromatogram data
        graph_type (str): Type of chromatogram graph to display
        
    Returns:
        dict: Plotly figure object for chromatogram plot
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
    """Generates protein count plot for samples.
    
    Args:
        _ (list): Placeholder for TIC plot children
        data_dictionary (dict): Data dictionary containing sample information
        replicate_colors (dict): Color assignments for samples including contaminants
        
    Returns:
        tuple: Contains:
            - Count plot components
            - Count data for storage
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
    """Generates plot showing common proteins across samples.
    
    Args:
        _ (list): Placeholder for count plot children
        data_dictionary (dict): Data dictionary containing sample information
        
    Returns:
        tuple: Contains:
            - Common proteins plot components
            - Common proteins data for storage
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
    """Generates protein coverage plot across samples.
    
    Args:
        _ (list): Placeholder for common proteins plot children
        data_dictionary (dict): Data dictionary containing sample information
        
    Returns:
        tuple: Contains:
            - Coverage plot components
            - Coverage data for storage
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
    """Generates plot showing reproducibility between sample replicates.
    
    Args:
        _ (list): Placeholder for coverage plot children
        data_dictionary (dict): Data dictionary containing sample information
        
    Returns:
        tuple: Contains:
            - Reproducibility plot components
            - Reproducibility data for storage
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
    """Generates plot showing missing values across samples.
    
    Args:
        _ (list): Placeholder for reproducibility plot children
        data_dictionary (dict): Data dictionary containing sample information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - Missing values plot components
            - Missing values data for storage
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
    """Generates plot showing sum of intensities across samples.
    
    Args:
        _ (list): Placeholder for missing plot children
        data_dictionary (dict): Data dictionary containing sample information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - Sum plot components
            - Sum data for storage
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
    """Generates plot showing mean intensities across samples.
    
    Args:
        _ (list): Placeholder for sum plot children
        data_dictionary (dict): Data dictionary containing sample information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - Mean plot components
            - Mean data for storage
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
    """Generates plot showing distribution of intensities across samples.
    
    Args:
        _ (list): Placeholder for mean plot children
        data_dictionary (dict): Data dictionary containing sample information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - Distribution plot components
            - Distribution data for storage
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
    """Stores selected sample groups for commonality plot visibility.
    
    Args:
        _ (int): Number of clicks on update plot button
        selection (list): List of selected sample groups
        
    Returns:
        dict: Dictionary containing selected groups for visibility
    """
    return {'groups': selection}

@callback(
    Output({'type': 'qc-plot', 'id': 'commonality-plot-div'}, 'children'),
    Input({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
)
def generate_commonality_container(data_dictionary: Dict[str, Any]) -> html.Div:
    """Generates container for commonality plot showing shared proteins between samples.
    
    Args:
        data_dictionary (dict): Data dictionary containing sample group information
        
    Returns:
        html.Div: Container component for commonality plot
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
    """Creates commonality plot showing protein overlap between sample groups.
    
    Args:
        data_dictionary (dict): Data dictionary containing sample information
        additional_options (list): List of selected additional plot options
        show_only_groups (dict): Dictionary specifying which groups to display
        
    Returns:
        tuple: Contains:
            - Commonality plot components
            - Plot data for storage
            - PDF version of plot for export
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
    """Notifies that QC analysis is complete.
    
    Args:
        _ (list): Placeholder for distribution plot children
        
    Returns:
        str: Empty string to trigger completion notification
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
    """Creates plot showing results of NA filtering in proteomics workflow.
    
    Args:
        nclicks (int): Number of clicks on run button
        uploaded_data (dict): Data dictionary containing proteomics data
        filtering_percentage (int): Minimum percentage threshold for filtering
        
    Returns:
        tuple: Contains:
            - NA filtering plot components
            - Filtered data for storage
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
    """Creates plot showing results of data normalization in proteomics workflow.
    
    Args:
        filtered_data (dict): NA-filtered proteomics data containing intensity values
            and sample information
        normalization_option (str): Selected normalization method (e.g., 'median', 
            'mean', 'vsn', etc.)
        
    Returns:
        tuple: Contains:
            - html.Div: Normalization plot components showing data distribution
                before and after normalization
            - dict: Normalized data for storage and downstream analysis
            
    Notes:
        Returns no_update if filtered_data is None. Uses figure height parameters
        from config and writes any R errors to specified error file.
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
    """Creates plot showing patterns of missing values across samples after normalization.
    
    Args:
        normalized_data (dict): Normalized proteomics data containing intensity values
            and sample information
        
    Returns:
        html.Div: Plot components showing the distribution and patterns of missing
            values across different samples, using half-height figure parameters
    """
    return proteomics.missing_values_in_other_samples(normalized_data, parameters['Figure defaults']['half-height'])

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-imputation-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-normalization-data-store'}, 'data'),
    Input('proteomics-imputation-radio-option', 'value'),
    prevent_initial_call=True
)
def proteomics_imputation_plot(normalized_data: Optional[Dict[str, Any]], imputation_option: str) -> Union[Tuple[html.Div, Dict[str, Any]], Any]:
    """Creates plot showing results of missing value imputation in proteomics workflow.
    
    Args:
        normalized_data (dict): Normalized proteomics data containing intensity values
            and sample information
        imputation_option (str): Selected imputation method (e.g., 'knn', 'mean', 
            'median', etc.)
        
    Returns:
        tuple: Contains:
            - html.Div: Imputation plot components showing data distribution
                before and after imputation
            - dict: Imputed data for storage and downstream analysis
            
    Notes:
        Returns no_update if normalized_data is None. Uses figure height parameters
        from config and writes any R errors to specified error file.
    """
    if normalized_data is None:
        return no_update
    return proteomics.imputation(normalized_data, imputation_option, parameters['Figure defaults']['full-height'], parameters['Config']['R error file'])


# TODO: Either implement or get rid of. need to decide, which.
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
def proteomics_pertubation(imputed_data: Optional[Dict[str, Any]], data_dictionary: Dict[str, Any], control_group: Optional[str], comparison_data: Optional[Dict[str, Any]], comparison_upload_success_style: Dict[str, str], replicate_colors: Dict[str, Any]) -> Union[Tuple[html.Div, str], Any]:
    """Creates perturbation analysis plots for proteomics data.
    
    Args:
        imputed_data (dict): Imputed proteomics data
        data_dictionary (dict): Main data dictionary containing sample information
        control_group (str): Selected control group name
        comparison_data (dict): Comparison table data
        comparison_upload_success_style (dict): Style indicating comparison upload status
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - Perturbation plot components
            - Perturbation data for storage
    """
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
    sgroups: Dict[str, Any] = data_dictionary['sample groups']['norm']
    comparisons: List[Tuple[Any, ...]] = parsing.parse_comparisons(
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
    Output({'type': 'workflow-plot', 'id': 'proteomics-cv-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-cv-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'},'data'),
    Input({'type': 'data-store', 'name': 'proteomics-na-filtered-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_cv_plot(uploaded_data: Dict[str, Any], na_filtered_data: Dict[str, Any], upload_dict: Dict[str, Any], replicate_colors: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Creates coefficient of variation (CV) plot for proteomics data.
    
    Generates a plot showing the coefficient of variation across samples using raw intensity
    data filtered to match the NA-filtered dataset. Uses sample grouping information and
    replicate colors for visualization.
    
    Args:
        uploaded_data (dict): Original uploaded data dictionary containing raw intensity values
        na_filtered_data (dict): NA-filtered proteomics data for filtering raw data
        upload_dict (dict): Main data dictionary containing sample grouping information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - html.Div: CV plot components showing variation across samples
            - dict: CV analysis data for storage
    """
    raw_int_data: pd.DataFrame = pd.read_json(StringIO(uploaded_data['data tables']['raw intensity']), orient='split')
    na_filtered_table: pd.DataFrame = pd.read_json(StringIO(na_filtered_data), orient='split')
    # Drop rows that are no longer present in filtered data
    raw_int_data.drop(index=list(set(raw_int_data.index)-set(na_filtered_table.index)),inplace=True)
    return proteomics.perc_cvplot(raw_int_data, upload_dict['sample groups']['norm'], replicate_colors, parameters['Figure defaults']['full-height'])

@callback(
    Output({'type': 'workflow-plot', 'id': 'proteomics-pca-plot-div'}, 'children'),
    Output({'type': 'data-store', 'name': 'proteomics-pca-data-store'}, 'data'),
    Input({'type': 'data-store', 'name': 'proteomics-imputation-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'upload-data-store'}, 'data'),
    State({'type': 'data-store', 'name': 'replicate-colors-data-store'}, 'data'),
    prevent_initial_call=True
)
def proteomics_pca_plot(imputed_data: Dict[str, Any], upload_dict: Dict[str, Any], replicate_colors: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Creates principal component analysis (PCA) plot for proteomics data.
    
    Generates a PCA plot to visualize sample clustering and relationships using imputed
    proteomics data. Uses sample grouping information and replicate colors for visualization.
    
    Args:
        imputed_data (dict): Imputed proteomics data for PCA analysis
        upload_dict (dict): Main data dictionary containing sample grouping information
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - html.Div: PCA plot components showing sample relationships
            - dict: PCA analysis data for storage
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
    """Creates hierarchical clustering heatmap for proteomics data.
    
    Generates a clustermap visualization showing hierarchical clustering of samples and
    proteins based on imputed proteomics data. Uses default figure height parameters.
    
    Args:
        imputed_data (dict): Imputed proteomics data for clustering analysis
        
    Returns:
        tuple: Contains:
            - html.Div: Clustermap plot components showing hierarchical clustering
            - dict: Clustering analysis data for storage
            - str: Empty string for completion notification
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
    """Validates uploaded comparison table for proteomics differential analysis.
    
    Checks the uploaded comparison table file for correct format and compatibility
    with the sample groups in the dataset.
    
    Args:
        contents (str): Base64 encoded contents of uploaded comparison file
        filename (str): Name of uploaded comparison file
        current_style (dict): Current style settings for upload success indicator
        data_dictionary (dict): Main data dictionary containing sample group information
        
    Returns:
        tuple: Contains:
            - dict: Updated style for upload success indicator (green for success,
                   red for failure)
            - dict: Validated comparison table data if successful, None if validation fails
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
    """Creates volcano plots for differential abundance analysis in proteomics workflow.
    
    Generates volcano plots showing differential protein abundance between sample groups,
    using either a control group or comparison table approach. Includes statistical testing
    and fold change thresholding.
    
    Args:
        imputed_data (dict): Imputed proteomics data for analysis
        control_group (str): Selected control group name, if using control-based comparisons
        comparison_data (dict): Comparison table data for custom comparisons
        comparison_upload_success_style (dict): Style indicating comparison table validation status
        data_dictionary (dict): Main data dictionary containing sample information
        fc_thr (float): Fold change threshold for significance
        p_thr (float): P-value threshold for significance
        test_type (str): Statistical test type to use for comparisons
        
    Returns:
        tuple: Contains:
            - html.Div: Volcano plot components showing differential abundance results
            - dict: Differential analysis data for storage
            - str: Empty string for completion notification
            
    Notes:
        Returns no_update if input validation fails. Logs warnings for various
        failure conditions related to comparison data validation.
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
    return proteomics.differential_abundance(imputed_data, sgroups, comparisons, fc_thr, p_thr, parameters['Figure defaults']['full-height'], test_type, parameters['Data paths']['Database file']) + ('',)

# Need to implement:
# GOBP mapping


@callback(
    Output('interactomics-choose-uploaded-controls', 'value'),
    [Input('interactomics-select-all-uploaded', 'value')],
    [State('interactomics-choose-uploaded-controls', 'options')],
    prevent_initial_call=True
)
def select_all_none_controls(all_selected: bool, options: List[Dict[str, str]]) -> List[str]:
    """Handles selection/deselection of all uploaded control samples.
    
    Args:
        all_selected (bool): Whether the "select all" checkbox is checked
        options (list): List of available control sample options, each containing
            a 'value' key
            
    Returns:
        list: List of all control sample values if all_selected is True,
            empty list otherwise
    """
    all_or_none: List[str] = [option['value'] for option in options if all_selected]
    return all_or_none

@callback(
    Output('interactomics-choose-enrichments', 'value'),
    [Input('interactomics-select-none-enrichments', 'n_clicks')],
    prevent_initial_call=True
)
def select_none_enrichments(deselect_click: Optional[int]) -> List[str]:
    """Deselects all enrichment options.
    
    Args:
        deselect_click (int): Number of times the deselect button has been clicked. Unused.
            
    Returns:
        list: Empty list to clear all enrichment selections
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
    """Toggles the collapse state of the input section.
    
    Args:
        header_click (int): Number of clicks on the header. Not used. 
        begin_click (int): Number of clicks on the begin analysis button. Nod used.
        input_is_open (bool): Current collapse state of the input section
            
    Returns:
        tuple: Contains:
            - str: Updated header text with arrow indicator (► or ▼)
            - bool: New collapse state (True for open, False for closed)
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
    """Handles selection/deselection of all inbuilt control sets.
    
    Args:
        all_selected (bool): Whether the "select all" checkbox is checked
        options (list): List of available inbuilt control set options, each containing
            a 'value' key
            
    Returns:
        list: List of all inbuilt control set values if all_selected is True,
            empty list otherwise
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
    """Handles selection/deselection of all CRAPome control sets.
    
    Args:
        all_selected (bool): Whether the "select all" checkbox is checked
        options (list): List of available CRAPome control set options, each containing
            a 'value' key
            
    Returns:
        list: List of all CRAPome control set values if all_selected is True,
            empty list otherwise
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
    """Initializes SAINT analysis with selected control samples and parameters.
    
    Args:
        nclicks (int): Number of clicks on run analysis button
        uploaded_controls (list): List of selected uploaded control samples
        additional_controls (list): List of selected additional control sets
        crapomes (list): List of selected CRAPome control sets
        uploaded_data (dict): Dictionary containing uploaded experimental data
        proximity_filtering_checklist (list): List of selected filtering options
        n_controls (int): Number of nearest controls to use if proximity filtering enabled
        
    Returns:
        tuple: Contains:
            - html.Div: SAINT analysis container components
            - dict: SAINT input data for storage
            - dict: CRAPome data for storage
            
    Notes:
        Returns no_update if button not clicked. Uses proximity filtering if 'Select'
        is in the filtering checklist.
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
    """Executes SAINT analysis using prepared input data.
    
    Args:
        saint_input (dict): Prepared SAINT input data
        data_dictionary (dict): Main data dictionary containing session and bait information
        
    Returns:
        tuple: Contains:
            - dict: SAINT analysis output data
            - bool: Whether SAINT executable was found
            - str: Empty string to clear loading indicator
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
    """Integrates CRAPome data with SAINT analysis results.
    
    Args:
        saint_output (dict): Results from SAINT analysis
        crapome (dict): CRAPome control data to integrate
        
    Returns:
        dict: Combined SAINT and CRAPome data, or error message if SAINT failed
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
    """Creates the filtering interface container for SAINT analysis results.
    
    Args:
        saint_output_ready (dict): Final SAINT analysis output data
        rescue (list): Selected rescue options for filtered interactions
        
    Returns:
        tuple: Contains:
            - str: Empty string for workflow completion notification
            - html.Div: Either error message if SAINT failed or filtering interface container
            
    Notes:
        Enables rescue functionality if "Rescue interactions that pass filter in any 
        sample group" is selected.
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
    """Generates histogram visualization of SAINT BFDR scores.
    
    Args:
        container_ready (list): Trigger input indicating filtering container is ready
        saint_output (str): Original SAINT analysis results
        saint_output_filtered (str): Filtered SAINT results if available
        
    Returns:
        tuple: Contains:
            - dict: Plotly figure object for BFDR histogram
            - dict: Histogram data for storage
            
    Notes:
        Uses filtered output data if available, otherwise uses original SAINT output.
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
    """Applies filtering criteria to SAINT analysis results.
    
    Args:
        bfdr_threshold (float): BFDR score threshold for filtering
        crapome_percentage (int): CRAPome frequency percentage threshold
        crapome_fc (int): CRAPome fold change threshold
        saint_output (str): SAINT analysis results to filter
        rescue (list): Selected rescue options for filtered interactions
        
    Returns:
        dict: Filtered SAINT analysis results based on specified thresholds
        
    Notes:
        Applies multiple filtering criteria:
        - BFDR score threshold
        - CRAPome frequency threshold
        - CRAPome fold change threshold
        Rescue functionality is enabled if rescue list is not empty.
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
    """Creates visualization of filtered SAINT analysis results.
    
    Args:
        filtered_output (dict): Filtered SAINT analysis results
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - dict: Plotly figure object showing filtered SAINT results
            - dict: Graph data for storage
    """
    return interactomics.saint_counts(filtered_output, parameters['Figure defaults']['half-height'], replicate_colors)

@callback(
    Output({'type': 'analysis-div',
           'id': 'interactomics-analysis-post-saint-area'}, 'children'),
    Input('interactomics-button-done-filtering', 'n_clicks'),
    prevent_initial_call=True
)
def interactomics_initiate_post_saint(_: Optional[int]) -> html.Div:
    """Initializes the post-SAINT analysis interface container.
    
    Args:
        _ (int): Number of clicks on done filtering button (unused)
        
    Returns:
        html.Div: Container component for post-SAINT analysis interface
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
def interactomics_map_intensity(n_clicks: Optional[int], unfiltered_saint_data: Dict[str, Any], data_dictionary: Dict[str, Any]) -> Union[str, Any]:
    """Maps intensity values to filtered SAINT results.
    
    Args:
        n_clicks (int): Number of clicks on done filtering button
        unfiltered_saint_data (dict): Filtered SAINT analysis results
        data_dictionary (dict): Main data dictionary containing intensity values
            and sample group information
            
    Returns:
        str: JSON string containing SAINT results with mapped intensity values
        
    Notes:
        Returns no_update if button not clicked or clicked less than once.
    """
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
def interactomics_known_plot(saint_output: Dict[str, Any], rep_colors_with_cont: Dict[str, Any]) -> Tuple[html.Div, Dict[str, Any]]:
    """Creates plot showing known interactions in SAINT results.
    
    Args:
        saint_output (dict): SAINT results with mapped intensity values
        rep_colors_with_cont (dict): Color assignments for samples including contaminants
        
    Returns:
        tuple: Contains:
            - html.Div: Plot components showing known interactions
            - dict: Known interactions data for storage
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
    """Creates plot showing common proteins across SAINT results.
    
    Args:
        _ (dict): Trigger input from known interactions plot (unused)
        saint_data (dict): Filtered SAINT analysis results
        
    Returns:
        tuple: Contains:
            - html.Div: Plot components showing common proteins
            - dict: Common proteins data for storage
            
    Notes:
        Converts SAINT data to matrix format before analysis. Includes contaminant
        proteins as an additional group in the visualization.
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
    """Creates principal component analysis (PCA) plot for interactomics data.
    
    Args:
        _ (dict): Trigger input from common proteins plot (unused)
        saint_data (dict): Filtered SAINT analysis results
        replicate_colors (dict): Color assignments for sample replicates
        
    Returns:
        tuple: Contains:
            - html.Div: PCA plot components showing sample relationships
            - dict: PCA analysis data for storage
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
    """Creates MS microscopy visualization plots.
    
    Args:
        _ (dict): Trigger input from PCA plot (unused)
        saint_output (dict): Filtered SAINT analysis results
        
    Returns:
        tuple: Contains:
            - html.Div: MS microscopy plot components
            - dict: MS microscopy analysis data for storage
            
    Notes:
        Uses version 1.0 of MS microscopy visualization.
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
    """Creates interactive network visualization of protein interactions.
    
    Args:
        _ (dict): Trigger input from MS microscopy plot (unused)
        saint_output (dict): Filtered SAINT analysis results
        
    Returns:
        tuple: Contains:
            - str: Empty string for workflow completion notification
            - html.Div: Network plot container components
            - dict: Network visualization elements data
            - dict: Interaction data for network
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
    """Performs enrichment analysis on filtered interactomics data.
    
    Args:
        _ (dict): Trigger input from network plot (unused)
        saint_output (dict): Filtered SAINT analysis results
        chosen_enrichments (list): List of selected enrichment analyses to perform
        
    Returns:
        tuple: Contains:
            - str: Empty string for workflow completion notification
            - html.Div: Enrichment analysis plot components
            - dict: Enrichment analysis results data
            - dict: Additional enrichment information data
            
    Notes:
        Runs as a background callback to handle potentially long computation times.
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
    """Displays detailed information for a selected node in the network visualization.
    
    Args:
        node_data (dict): Data associated with the tapped network node
        int_data (dict): Network interaction data store
        network_type (str, optional): Type of network visualization. Defaults to 'Cytoscape'
        
    Returns:
        html.Div: Component containing detailed node information, or None if no node selected
        
    Notes:
        Currently only supports Cytoscape network type. visdcc network support is commented out.
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
    """Updates the layout of the Cytoscape network visualization.
    
    Args:
        layout (str): Selected layout type from dropdown
        
    Returns:
        dict: Layout configuration dictionary with name and optional parameters
        
    Notes:
        Applies additional layout parameters if defined in parameters['Cytoscape layout parameters']
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
    """Updates table of contents based on main content.
    
    Args:
        _,__,___,____ (any): Trigger inputs from various workflow completion notifiers (unused)
        main_div_contents (list): Current contents of main div
        
    Returns:
        html.Div: Updated table of contents component
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
    """Updates workflow-specific areas based on selected workflow.
    
    Args:
        _ (any): Trigger input from QC completion notifier (unused)
        workflow (str): Selected workflow type
        data_dictionary (dict): Main data dictionary containing uploaded data
        
    Returns:
        tuple: Contains workflow-specific input and content areas
    """
    return ui.workflow_area(workflow, parameters['workflow parameters'], data_dictionary)


@callback(
    Output('download-proteomics-comparison-example', 'data'),
    Input('download-proteomics-comparison-example-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_example_comparison_file(n_clicks: Optional[int]) -> Optional[Dict[str, Any]]:
    """Handles download of example proteomics comparison file.
    
    Args:
        n_clicks (int): Number of clicks on download button
        
    Returns:
        dict: File download configuration for example comparison file, or None if button not clicked
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
    """Handles download of example files.
    
    Args:
        _ (int): Number of clicks on download button (unused)
        
    Returns:
        dict: File download configuration for example files
    """
    return dcc.send_file(utils.zipdir(os.path.join(*parameters['Data paths']['Example files'])))

def get_adiv_by_id(
    divs: List[Any], 
    idvals: List[Dict[str, str]], 
    idval_to_find: str
) -> Optional[Any]:
    """Retrieves a specific div element from a list by matching its ID.
    
    Args:
        divs (list): List of div elements
        idvals (list): List of dictionaries containing ID values
        idval_to_find (str): Target ID value to search for
        
    Returns:
        Any: The matching div element if found, None otherwise
        
    Notes:
        - Used for finding specific analysis div elements in the UI
        - Returns None if no matching ID is found
        - Assumes idvals list contains dictionaries with 'id' key
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
    """Prepares directory structure and README for data export.
    
    Args:
        _ (int): Number of clicks on download button (unused)
        main_data (dict): Main data dictionary containing session information
        
    Returns:
        tuple: Contains:
            - str: Path to created export directory
            - html.Div: Temporary loading indicator components
            
    Notes:
        - Creates timestamped export directory
        - Removes existing directory if present
        - Converts output guide markdown to HTML for README
    """
    timestamp: str = datetime.now().strftime("%Y-%m-%d %H-%M")
    export_dir: str = os.path.join(*parameters['Data paths']['Cache dir'],  
                                 main_data['other']['session name'], 
                                 f'{timestamp} Proteogyver output')
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    with open(os.path.join('data','output_guide.md')) as fil:
        text: str = fil.read()
        html_content: str = markdown.markdown(text, extensions=['markdown.extensions.nl2br', 'markdown.extensions.sane_lists'])
    with open(os.path.join(export_dir, 'README.html'),'w',encoding='utf-8') as fil:
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                }}
                ul, ol {{
                    padding-left: 20px;
                    margin-bottom: 20px;
                }}
                li {{
                    margin-bottom: 8px;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        fil.write(html_template)
    return export_dir, infra.temporary_download_button_loading_divs()

@callback(
    Output('download_temp1', 'children'),
    Output('download_loading_temp1', 'children'),
    Input('download-temp-dir-ready','children'),
    State('input-stores', 'children'),
    prevent_initial_call=True
)
def save_input_stores(export_dir: str, stores: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Saves input data stores to export directory.
    
    Args:
        export_dir (str): Path to export directory
        stores (list): List of input data stores to save
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Uses infrastructure function to save data stores
    """
    start = datetime.now()
    logger.info(f'received download request save_input_stores at {start}')
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
    """Saves workflow data stores to export directory.
    
    Args:
        export_dir (str): Path to export directory
        stores (list): List of workflow data stores to save
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Uses infrastructure function to save data stores
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
    """Saves quality control figures to export directory.
    
    Args:
        export_dir (str): Path to export directory
        analysis_divs (list): List of analysis div elements
        analysis_div_ids (list): List of analysis div IDs
        commonality_pdf_data (dict): PDF data for commonality figures
        workflow (str): Current workflow type
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Handles errors by writing to error file
        - Uses infrastructure function to save figures
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
    """Saves input information to export directory.
    
    Args:
        export_dir (str): Path to export directory
        input_divs (list): List of input div elements
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Uses infrastructure function to save input information
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
    """Saves interactomics analysis figures to export directory.
    
    Args:
        export_dir (str): Path to export directory
        analysis_divs (list): List of analysis div elements
        analysis_div_ids (list): List of analysis div IDs
        workflow (str): Current workflow type
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Handles errors by writing to error file
        - Uses infrastructure function to save figures
    """
    start = datetime.now()
    logger.info(f'received download request save_interactomics_figures at {start}')
    try:
        infra.save_figures([get_adiv_by_id(analysis_divs, analysis_div_ids, 'interactomics-analysis-results-area')], export_dir,
                    figure_output_formats, None, workflow)
    except Exception as e:
        logger.warning(f'save_interactomics_figures failed: {e}')
        with open(os.path.join(export_dir, 'save_interactomics_figures_errors'),'w') as fil:
            fil.write(f'{e}')
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
    """Saves post-SAINT interactomics analysis figures to export directory.
    
    Args:
        export_dir (str): Path to export directory
        analysis_divs (list): List of analysis div elements
        analysis_div_ids (list): List of analysis div IDs
        workflow (str): Current workflow type
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Handles errors by writing to error file
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
    """Saves proteomics analysis figures to export directory.
    
    Args:
        export_dir (str): Path to export directory
        analysis_divs (list): List of analysis div elements
        analysis_div_ids (list): List of analysis div IDs
        workflow (str): Current workflow type
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Handles errors by writing to error file
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
    """Saves phosphoproteomics analysis figures to export directory.
    
    Args:
        export_dir (str): Path to export directory
        analysis_divs (list): List of analysis div elements
        analysis_div_ids (list): List of analysis div IDs
        workflow (str): Current workflow type
        
    Returns:
        tuple: Contains:
            - str: Completion message
            - str: Empty string to clear loading indicator
            
    Notes:
        - Logs start and completion times
        - Handles errors by writing to error file
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
    """Creates and sends a ZIP archive containing all exported analysis data.
    
    Args:
        export_dir (str): Path to the temporary export directory
        *args: Variable number of completion status inputs from previous export steps:
            - download_temp1: Input stores
            - download_temp2: Workflow stores
            - download_temp3: QC figures
            - download_temp4: Input information
            - download_temp5: Interactomics figures
            - download_temp6: Post-SAINT figures
            - download_temp7: Proteomics figures
            - download_temp8: Phosphoproteomics figures
        
    Returns:
        tuple: Contains:
            - dict: Download configuration for ZIP file using dcc.send_bytes
            - str: Updated button text indicating download is ready
            
    Notes:
        - Verifies all export steps are complete by checking for 'done' in status
        - Creates timestamped ZIP archive containing all exported files
        - Preserves directory structure relative to export directory
        - Excludes the ZIP file itself from the archive
        - Cleans up temporary export directory after ZIP creation
        - Logs start time and duration of packaging process
        - Returns no_update if any export step is incomplete
        - Handles errors with logging and returns no_update on failure
        
    Example ZIP structure:
        timestamp ProteoGyver output.zip/
        ├── README.html
        ├── figures/
        │   ├── qc/
        │   ├── interactomics/
        │   ├── proteomics/
        │   └── phosphoproteomics/
        ├── data/
        │   ├── input_stores/
        │   └── workflow_stores/
        └── errors/
            └── *_errors (if any occurred)
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
