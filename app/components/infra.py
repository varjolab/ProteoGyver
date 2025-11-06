"""Infrastructure components for the Proteogyver web application.

This module provides core infrastructure components and utilities for the Proteogyver web app,
including data storage, figure export, and utility functions.

Key functionality includes:

*   Data store configuration and management
*   Figure export in multiple formats (HTML, PDF, PNG)
*   Input parameter tracking and export
*   Utility functions for component traversal and data formatting
*   Hidden utility components for app functionality

The module defines configurations for data store exports and figure directories, and provides
functions for saving data, figures, and input parameters to files. It also creates various
Dash components used throughout the application.

Functions
---------
.. py:func:: save_data_stores
    Saves data from data stores to files.
.. py:func:: save_figures
    Exports figures in various formats.
.. py:func:: save_input_information
    Saves input parameters to TSV.
.. py:func:: get_all_props
    Utility function for traversing Dash components.
.. py:func:: get_all_types
    Utility function for traversing Dash components.
.. py:func:: upload_data_stores
    Creates Dash components for uploaded data stores.
.. py:func:: working_data_stores
    Creates Dash components for processed/working data stores.

Constants
---------
.. py:data:: DATA_STORE_IDS
    List of all data store IDs.
.. py:data:: data_store_export_configuration
    Export settings for each data store.
.. py:data:: figure_export_directories
    Output directory mapping for figures.
"""

from dash import dcc, html
import os
from plotly import io as pio
from plotly import graph_objects as go
from io import StringIO
import re
import markdown
import json
import pandas as pd
from base64 import b64decode
from datetime import datetime
import logging
from _version import __version__
logger = logging.getLogger(__name__)

data_store_export_configuration: dict = {
    'interactomics-network-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'interactomics-network-interactions-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'interactomics-saint-bfdr-histogram-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'interactomics-saint-graph-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'interactomics-common-protein-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'proteomics-comparison-table-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'commonality-figure-pdf-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'proteomics-cv-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'qc-commonality-plot-visible-groups-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'tic-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'version-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],

    'uploaded-data-table-info-data-store': ['txt', 'Input data info', 'Data table', 'input-file'],
    'uploaded-sample-table-info-data-store': ['txt', 'Input data info', 'Sample table', 'input-file'],
    'upload-data-store': ['json', 'Debug', '', ''],
    'replicate-colors-data-store': ['json', 'Debug', '', ''],
    'replicate-colors-with-contaminants-data-store': ['json', 'Debug', '', ''],
    'discard-samples-data-store': ['json', 'Debug', '', ''],
    'commonality-data-store': ['txt', 'Data', 'Shared proteins', ''],
    'reproducibility-data-store': ['json', 'Data', 'Reproducibility data', ''],
    
    'uploaded-sample-table-data-store': ['tsv', 'Data', ['Input data tables', 'Uploaded expdesign'],'noindex'],
    'uploaded-data-table-data-store': ['tsv', 'Data', 'Input data tables', 'upload-split'],

    'interactomics-msmic-data-store': ['tsv', 'Data', 'MS microscopy results', ''],
    'interactomics-volcano-data-store': ['tsv', 'Data', 'Differential abundance', 'volc-split'],
    
    'count-data-store': ['tsv', 'Data', ['Summary data', 'Protein counts'],''],
    'common-protein-data-store': ['tsv', 'Data', ['Summary data', 'Common proteins'],''],
    'coverage-data-store': ['tsv', 'Data', ['Summary data', 'Protein coverage'],''],
    'missing-data-store': ['tsv', 'Data', ['Summary data', 'Missing counts'],''],
    'sum-data-store': ['tsv', 'Data', ['Summary data', 'Value sums'],''],
    'mean-data-store': ['tsv', 'Data', ['Summary data', 'Value means'],''],
    'proteomics-distribution-data-store': ['tsv', 'Data', ['Summary data', 'sample distribution'],''],
    'distribution-data-store': ['tsv', 'Data', ['Summary data', 'Value distribution'],''],

    'proteomics-pca-data-store': ['tsv', 'Data', ['Figure data', 'Proteomics PCA'],''],
    'proteomics-clustermap-data-store': ['tsv', 'Data', ['Figure data', 'Proteomics Clustermap'],''],
    'proteomics-pertubation-data-store': ['tsv', 'Data', ['Figure data', 'Proteomics pertubation'],''],

    'proteomics-na-filtered-data-store': ['tsv', 'Data', ['Proteomics data tables', 'NA filtered data'],''],
    'proteomics-normalization-data-store': ['tsv', 'Data', ['Proteomics data tables', 'NA-Normalized data'],''],
    'proteomics-imputation-data-store': ['tsv', 'Data', ['Proteomics data tables', 'NA-Norm-Imputed data'],''],
    
    'interactomics-pca-data-store': ['tsv', 'Data', ['Interactomics data tables', 'PCA'],''],
    'interactomics-imputed-and-normalized-intensity': ['tsv', 'Data', ['Interactomics data tables', 'ImpNorm intensities'],''],
    'interactomics-saint-crapome-data-store': ['tsv', 'Data', ['Interactomics data tables', 'Crapome'],''],
    'interactomics-saint-output-data-store': ['tsv', 'Data', ['Interactomics data tables', 'Saint output'],'noindex'],
    'interactomics-saint-final-output-data-store': ['tsv', 'Data', ['Interactomics data tables', 'Saint output with crapome'],'noindex'],
    'interactomics-saint-filtered-output-data-store': ['tsv', 'Data', ['Interactomics data tables', 'Filtered saint output'],'noindex'],
    'interactomics-saint-input-data-store': ['tsv', 'Data', 'SAINT input', 'saint-split'],
    'interactomics-enrichment-data-store': ['tsv', 'Data', 'Enrichment', 'enrichment-split'],
    'proteomics-volcano-data-store': ['tsv', 'Data', 'Differential abundance', 'volc-split'],
    'interactomics-saint-filtered-and-intensity-mapped-output-data-store': ['tsv', 'Data', 'Filtered interactomics results with intensity', 'rename-int'],
    'interactomics-saint-filt-int-known-data-store': ['tsv', 'Data', 'Filtered interactomics results with intensity and knowns', 'rename-int'],

    'interactomics-enrichment-information-data-store': ['txt', 'Data', 'Enrichment information', 'enrich-split'],
}

figure_export_directories: dict = {
    'Filtered Prey counts per bait': 'Interactomics figures',
    'Identified known interactions': 'Interactomics figures',
    'Sample run TICs': 'QC figures',
    'Sample run BPCs': 'QC figures',
    'Sample run MSns': 'QC figures',
    'Missing values per sample': 'QC figures',
    'SPC PCA': 'Interactomics figures',
    'Protein identification coverage': 'QC figures',
    'Proteins per sample': 'QC figures',
    'SAINT BFDR value distribution': 'Interactomics figures',
    'Sample reproducibility': 'QC figures',
    'Shared identifications': 'QC figures',
    'Sum of values per sample': 'QC figures',
    'Value distribution per sample': 'QC figures',
    'Value mean': 'QC figures',
    'Imputation': 'Proteomics figures',
    'Missing value filtering': 'Proteomics figures',
    'Coefficients of variation': 'Proteomics figures',
    'High-confidence interactions and identified known interactions': 'Interactomics figures',
    'Common proteins in data (qc)': 'QC figures',
    'Common proteins in data (interactomics)': 'Interactomics figures',
    'Intensity of proteins with missing values in other samples': 'Proteomics figures',
    'Normalization': 'Proteomics figures',
    'PCA': 'Proteomics figures',
    'Sample correlation clustering': 'Proteomics figures',
}

DATA_STORE_IDS = list(data_store_export_configuration.keys())

def save_data_stores(data_stores, export_dir) -> dict:
    """Save data from data stores into files under an export directory.

    :param data_stores: List of Store-like elements with ``props.data``.
    :param export_dir: Destination directory path.
    :returns: Dict of failures keyed by store id (empty if none).
    """
    prev_time: datetime = datetime.now()
    logger.info(f'save data stores - started: {prev_time}')
    fails = {}
    fails_errors = {}
    for d in data_stores:
        if not 'data' in d['props']:
            continue
        if isinstance(d['props']['data'], str):
            if d['props']['data'].strip() == '':
                continue
        export_format: str
        export_subdir: str
        file_name: str
        file_config: str
        export_format, export_subdir, file_name, file_config = data_store_export_configuration[
            d['props']['id']['name']]
        output_index=True
        if 'noindex' in file_config:
            file_config = file_config.replace('noindex','')
            output_index = False
        if export_format == 'NO EXPORT':
            continue
        try:
            export_destination: str = os.path.join(export_dir, export_subdir)
            if isinstance(file_name, list):
                export_destination = os.path.join(export_destination, *file_name[:-1])
                file_name = file_name[-1]
            if not os.path.isdir(export_destination):
                os.makedirs(export_destination)
            if export_format == 'json':
                if file_name == '':
                    file_name = d['props']['id']['name']
                with open(os.path.join(export_destination, file_name+'.json'), 'w', encoding='utf-8') as fil:
                    dict_to_write: dict = {}
                    if isinstance(d['props']['data'], dict):
                        dict_to_write = d['props']['data']
                    elif isinstance(d['props']['data'], str):
                        dict_to_write = json.loads(d['props']['data'])
                    json.dump(dict_to_write, fil, indent=2)
            elif export_format == 'txt':
                if file_config == 'enrich-split':
                    for enrichment_name, file_contents in d['props']['data']:
                        with open(os.path.join(export_destination, 'Enrichment',f'{file_name} {enrichment_name}.{export_format}'), 'w', encoding='utf-8') as fil:
                            fil.write(file_contents)
                elif file_config == 'input-file':
                    with open(os.path.join(export_destination, f'{file_name}.{export_format}'),'w',encoding='utf-8') as fil:
                        if file_name == 'Data table':
                            fil.write('\n'.join([
                                f'File modified timestamp: {d["props"]["data"]["Modified time"]}',
                                f'File name: {d["props"]["data"]["File name"]}',
                                f'Data type: {d["props"]["data"]["Data type"]}',
                            ]))
                            if 'Data source guess' in d['props']['data']:
                                fil.write(f'\nData source guess: {d["props"]["data"]["Data source guess"]}\n')
                        if file_name == 'Sample table':
                            fil.write('\n'.join([
                                f'File modified timestamp: {d["props"]["data"]["Modified time"]}',
                                f'File name: {d["props"]["data"]["File name"]}'
                            ]))
                else:
                    with open(os.path.join(export_destination, f'{file_name}.{export_format}'),'w',encoding = 'utf-8') as fil:
                        fil.write(d['props']['data'])
            elif export_format == 'tsv':
                if not 'split' in file_config:
                    pd_df: pd.DataFrame = pd.read_json(StringIO(d['props']['data']),orient='split')
                    if 'pertubation' in file_config:
                        continue
                    if 'rename-int' in file_config:
                        if not 'Averaged intensity' in pd_df.columns:
                            if ('intensity' in file_name)  and ('knowns' in file_name):
                                file_name = 'Filtered interactomics results with knowns'
                            else:
                                continue
                    use_name = os.path.join(export_destination, '.'.join([file_name, export_format]))
                    pd_df.to_csv(use_name, sep='\t', index=output_index)
                elif 'enrichment-split' in file_config:
                    export_destination = os.path.join(export_destination, file_name)
                    if not os.path.isdir(export_destination): os.makedirs(export_destination)
                    for enrichment_name, enrichment_dict in d['props']['data'].items():
                        use_name = os.path.join(export_destination, '.'.join([f'{enrichment_name}', export_format]))
                        pd.read_json(StringIO(enrichment_dict['result']),orient='split').to_csv(use_name,sep='\t',index=False)
                elif file_config == 'upload-split':
                    export_destination = os.path.join(export_destination, file_name)
                    if not os.path.isdir(export_destination): os.makedirs(export_destination)
                    for key, data_table in d['props']['data'].items():
                        use_name = os.path.join(export_destination, '.'.join([f'{key}', export_format]))
                        pd_df:pd.DataFrame = pd.read_json(StringIO(data_table),orient='split')
                        if pd_df.shape[1] < 2: continue
                        pd_df.to_csv(use_name,sep='\t')
                elif file_config == 'saint-split':
                    for saint_name, saint_filelines in d['props']['data'].items():
                        use_name = os.path.join(export_destination, '.'.join([f'{file_name} {saint_name}', export_format]))
                        with open(use_name,'w',encoding='utf-8') as fil:
                            for line in saint_filelines:
                                fil.write('\t'.join(line)+'\n')
                elif 'volc-split' in file_config:
                    df: pd.DataFrame = pd.read_json(StringIO(d['props']['data']),orient='split')
                    df_dicts: list = []
                    df_dicts_all: list = []
                    for _, s_c_row in df[['Sample', 'Control']].drop_duplicates().iterrows():
                        sample: str = s_c_row['Sample']
                        control: str = s_c_row['Control']
                        compname: str = f'{sample} vs {control}'
                        sig_comp: str = f'{compname} sig only'
                        if len(sig_comp) > 31:
                            compname = compname.replace(' ', '')
                            replace = re.compile(
                                re.escape('samplegroup'), re.IGNORECASE)
                            compname = replace.sub('SG', compname)
                            if 'samplegroup' in compname.lower():
                                compname = compname.replace()
                            sig_comp = f'{compname} significant only'
                        sig_df = df[(df['Sample'] == sample) & (df['Control'] == control) & df['Significant']]
                        if sig_df.shape[0] > 0:
                            df_dicts.append({
                                'name': sig_comp,
                                'data': sig_df,
                                'headers': True,
                                'index': False
                            })
                        df_dicts_all.append({
                            'name': f'{compname}',
                            'data': df[(df['Sample'] == sample) & (df['Control'] == control)],
                            'headers': True,
                            'index': False
                        })
                    export_destination = os.path.join(export_destination, file_name)
                    if not os.path.isdir(export_destination): os.makedirs(export_destination)
                    for df_dict in df_dicts:
                        use_name = os.path.join(export_destination, '.'.join([f'{df_dict["name"]}', export_format]))
                        df_dict['data'].to_csv(use_name,sep='\t',index=False)
                    for df_dict in df_dicts_all:
                        use_name = os.path.join(export_destination, '.'.join([f'{df_dict["name"]}', export_format]))
                        df_dict['data'].to_csv(use_name,sep='\t',index=False)
                else:
                    with open('DEBUG_INFRA_SAVE_DATA_STORES','w', encoding = 'utf-8') as fil:
                        fil.write(f'{d}')
        except Exception as e: # pylint: disable=bare-except
            fails[d['props']['id']['name']] = d
            fails_errors[d['props']['id']['name']] = f'{e}'
        
        logger.info(
            f'save data stores - export {d["props"]["id"]["name"]} done: {datetime.now() - prev_time}')
        prev_time: datetime = datetime.now()
    if len(fails) > 0:
        logger.warning(f'Failed to save data stores: {", ".join(fails.keys())}')
        with open('FAILED_DATA_STORES_errors.json', 'w', encoding = 'utf-8') as fil:
            json.dump(fails_errors, fil, indent = 4)
        with open('FAILED_DATA_STORES.json', 'w', encoding = 'utf-8') as fil:
            json.dump(fails, fil, indent = 4)
    logger.info(
        f'save data stores - Done with export: {datetime.now() - prev_time}')

def get_all_props(elements, marker_key, match_partial=True) -> list:
    """Find all elements whose props contain a marker key.

    :param elements: Nested dash component-like dict/list structure.
    :param marker_key: Key to search for within ``props``.
    :param match_partial: Whether to allow partial matches (unused).
    :returns: List of tuples ``(marker_key, element)``.
    """
    ret: list = []
    if isinstance(elements, dict):
        mkey: str = None
        for key in elements['props'].keys():
            if marker_key in key:
                mkey = marker_key
        if mkey is not None:
            ret.append((mkey, elements))
        else:
            try:
                ret.extend(get_all_props(
                    elements['props']['children'], marker_key, match_partial))
            except KeyError:
                pass
    elif isinstance(elements, list):
        for e in elements:
            ret.extend(get_all_props(e, marker_key, match_partial))
    return ret


def get_all_types(elements, get_types) -> list:
    """Find all elements of specified types in a nested structure.

    :param elements: Nested dash component-like dict/list structure.
    :param get_types: List of element type strings to collect (e.g., ``['h4','graph']``).
    :returns: List of matching elements.
    """
    ret = []
    if isinstance(elements, dict):
        # return [elements]
        for gt in get_types:
            if elements['type'].lower() == gt.lower():
                ret.append(elements)
                break
        else:
            try:
                ret.extend(get_all_types(
                    elements['props']['children'], get_types))
            except KeyError:
                pass
    elif isinstance(elements, list):
        for e in elements:
            ret.extend(get_all_types(e, get_types))
    return ret

def write_README(save_dir, guide_file) -> None:
    """Write a README.html file rendered from markdown.

    :param save_dir: Target directory for README.
    :param guide_file: Path to markdown file to render.
    :returns: None
    """

    with open(guide_file) as fil:
        text: str = fil.read()
        html_content: str = markdown.markdown(text, extensions=['markdown.extensions.nl2br', 'markdown.extensions.sane_lists'])
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
    with open(os.path.join(save_dir, 'README.html'),'w',encoding='utf-8') as fil:
        fil.write(html_template)

def save_figures(analysis_divs, export_dir, output_formats, commonality_pdf_data, workflow) -> None:
    """Save figures found in analysis divs in requested formats.

    :param analysis_divs: Analysis container elements to scan.
    :param export_dir: Base directory to save figures to.
    :param output_formats: Formats to export (e.g., ``['html','pdf','png']``).
    :param commonality_pdf_data: Optional PDF bytes (base64) for commonality figure.
    :param workflow: Workflow name used in paths.
    :returns: None
    """
    logger.info(f'saving figures: {datetime.now()}')
    prev_time: datetime = datetime.now()
    headers_and_figures: list = get_all_types(
        analysis_divs, ['h4', 'h5', 'graph', 'img', 'P'])
    figure_names_and_figures: list = []
    prev_time: datetime = datetime.now()
    # if there is pdf data, it means that matplotlib engine was used to generate this particular figure, and we need to add it to the queue in this form:
    if (commonality_pdf_data is not None) and (len(commonality_pdf_data) > 0):
        header: str = 'Shared identifications'
        figure_names_and_figures.append([
            figure_export_directories[header],
            header,
            '',
            'NO HTML',
            commonality_pdf_data,
            'pdf_text'
        ])

    for i, header in enumerate(headers_and_figures):
        if i < (len(headers_and_figures)-2):
            if header['type'].lower() == 'h4':
                graph: dict = headers_and_figures[i+1]
                if graph['type'].lower() not in {'graph', 'img'}:
                    continue
                legend: dict = headers_and_figures[i+2]
                header_str: str = header['props']['children']
                fdir: str = ''
                if header_str in figure_export_directories:
                    fdir = figure_export_directories[header_str]
                elif ('volcano' in header_str.lower()) or ('all significant differences vs' in header_str.lower()):
                    fdir = 'Volcano plots'
                if graph['type'].lower() == 'graph':
                    figure: dict = graph['props']['figure']
                    figure_html: str = pio.to_html(
                        figure, config=graph['props']['config'])
                    figure_names_and_figures.append(
                        [fdir, header_str, legend['props']['children'], figure_html, figure, 'graph'])
                elif graph['type'].lower() == 'img':
                    img_str: str = graph['props']['src']
                    figure_html = f'<Img id={graph["props"]["id"]}, src="{img_str}">'
                    figure_names_and_figures.append(
                        [fdir, header_str, legend['props']['children'], figure_html, graph, 'img'])
            elif header['type'].lower() == 'h5':  # Only used in enrichment plots and MS-microscopy plots
                graph: dict = headers_and_figures[i+1]
                legend: dict = headers_and_figures[i+2]
                figure: dict = graph['props']['figure']
                figure_html: str = pio.to_html(
                    figure, config=graph['props']['config'])
                fig_subdir: str = 'Enrichment figures'
                if 'microscopy' in legend['props']['children'].lower():
                    fig_subdir = 'MS-microscopy'
                figure_names_and_figures.append([
                    fig_subdir, header['props']['children'], legend[
                        'props']['children'], figure_html, figure, 'graph'
                ])

    logger.info(
        f'saving figures - figures identified: {len(figure_names_and_figures)}: {datetime.now() - prev_time}')
    prev_time: datetime = datetime.now()

    for subdir, name, legend, fig_html, fig, figtype in figure_names_and_figures:
        while ';REP' in subdir:
            _, rep_what, __ = subdir.split(';',maxsplit=2)
            replace = rep_what.split(':',maxsplit=1)[1]
            if replace == 'WORKFLOW':
                subdir = subdir.replace(f';{rep_what};', workflow)
        target_dir: str = os.path.join(export_dir, subdir)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        if ('html' in output_formats) and (fig_html != 'NO HTML'):
            new_html: list = []
            split_html: list = fig_html.split('\n')
            add_header = False
            for line in split_html[:-2]:
                if '<body>' in line:
                    add_header = True
                new_html.append(line)
                if add_header:
                    new_html.append(f'<div><H4>{name}</H4></div>')
                    add_header = False
            new_html.append(f'<div><p>{legend}')
            new_html.extend(split_html[-2:])
            with open(os.path.join(target_dir, f'{name}.html'), 'w', encoding='utf-8') as fil:
                fil.write('\n'.join(new_html))
        for output_format in output_formats:
            fig_path:str = os.path.join(target_dir, f'{name}.{output_format}')
            if output_format == 'html':
                continue
            if figtype == 'graph':
                try:
                    go.Figure(fig).write_image(fig_path)
                except Exception as e:
                    with open(fig_path.replace(f'.{output_format}', ' ERROR.txt'),'w',encoding='utf-8') as fil:
                        fil.write(f'Error Message:\n{e}')
            elif figtype == 'img':
                if output_format == 'pdf':
                    continue
                with open(fig_path, 'wb') as fil:
                    fil.write(b64decode(fig['props']['src'].replace(
                        'data:image/png;base64,', '')))
            elif (figtype == 'pdf_text'):
                if output_format != 'pdf':
                    continue
                with open(fig_path, 'wb') as fil:
                    fil.write(b64decode(fig))

        logger.info(
            f'saving figures - writing: {name}.{output_format}: {datetime.now() - prev_time}')
        prev_time: datetime = datetime.now()
    logger.info(f'saving figures - done: {datetime.now() - prev_time}')


def format_nested_list(input_list: list):
    """Format a nested list into a comma-separated string.

    :param input_list: Nested list to format.
    :returns: Comma-separated string representation.
    """
    if not isinstance(input_list, list):
        return str(input_list)
    rlist: list = []
    for entry in input_list:
        if isinstance(entry, list):
            rlist.append(format_nested_list(entry))
        else:
            rlist.append(str(entry))
    return ', '.join(rlist)


def save_input_information(input_divs, version_data_store, export_dir) -> None:
    """Save user input parameters/settings to a TSV file.

    :param input_divs: Input container elements to scan.
    :param export_dir: Destination directory.
    :returns: None
    """
    logger.info(f'saving input info: {datetime.now()}')
    prev_time: datetime = datetime.now()
    these: list = [
        'Slider',
        'Select',
        'Label',
        'Checklist',
        'RadioItems',
        'Input'
    ]
    input_options: list = [['Proteogyver version', __version__]]
    labels_and_inputs: list = get_all_types(input_divs, these)
    for i, label in enumerate(labels_and_inputs):
        if label['type'] != 'Label':
            continue
        if label['props']['children'] == 'of:':
            label['props']['children'] = 'NA filtering based on'
        if len(label['props']['children']) < 4:
            continue
        try:
            input_options.append(
                [label['props']['children'], labels_and_inputs[i+1]['props']['value']])
        except KeyError:
            continue
    # Have to iterate over the inputs again to gather some trickier input options. Janky.
    for i, input in enumerate(labels_and_inputs):
        if not 'id' in input['props']:
            continue
        if input['props']['id'] == 'interactomics-nearest-control-filtering':
            usebool: bool = False
            if len(input['props']['value']) > 0:
                if input['props']['value'][0] is not None:
                    usebool = True
            input_options.append(
                ['Use only most-similar controls', usebool])
        elif input['props']['id'] == 'interactomics-num-controls':
            input_options.append(
                ['Number of used controls (If similarity scoring used for controls)', input['props']['value']])
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    with open(os.path.join(export_dir, 'Workflow information and parameters.tsv'), 'w', encoding='utf-8') as fil:
        fil.write('Parameter\tValue\n')
        fil.write('Tool\tProteoGyver QC and preliminary analysis\n')
        fil.write(f'Export time\t{timestamp}\n')
        for name, values in input_options:
            val_str: str = format_nested_list(values)
            if len(val_str) == 0:
                val_str = 'None'
            fil.write(f'{name}\t{val_str}\n')
    with open(os.path.join(export_dir, 'Software and database versions.tsv'), 'w', encoding='utf-8') as fil:
        fil.write('Entity\tVersion\n')
        for name, version in version_data_store.items():
            fil.write(f'{name}\t{version}\n')
    logger.info(f'saving input info - done: {datetime.now() - prev_time}')

def upload_data_stores() -> html.Div:
    """Create Store components for uploaded data.

    :returns: Div containing uploaded data Stores.
    """
    stores: list = []
    for ID_STR in DATA_STORE_IDS:
        if 'uploaded' in ID_STR:
            stores.append(
                dcc.Store(id={'type': 'uploaded-data-store', 'name': ID_STR}))
    return html.Div(
        id='input-stores',
        children=stores
    )


def working_data_stores() -> html.Div:
    """Create Store components for processed/working data.

    :returns: Div containing working data Stores.
    """
    stores: list = []
    for ID_STR in DATA_STORE_IDS:
        if 'uploaded' in ID_STR:
            continue
        stores.append(dcc.Store(id={'type': 'data-store', 'name': ID_STR}))
    return html.Div(id='workflow-stores', children=stores)

def temporary_download_divs():
    """Create temporary divs used for downloads.

    :returns: Div containing temporary download children.
    """
    return html.Div(
        id='download-temporary-things',
        children=[html.Div(id=f'download_temp{i}',children='') for i in range(1,9)]+ [html.Div(id='download-temp-dir-ready',children='')]
    )

def temporary_download_button_loading_divs():
    """Create hidden loading indicators for download buttons.

    :returns: Div with Loading components used during downloads.
    """
    return html.Div(
        [
            html.Div([dcc.Loading(id=f'download_loading_temp{i}',children='') for i in range(1,9)], hidden=True),
            '   '
        ]
    )

def invisible_utilities() -> html.Div:
    """Create hidden utility components container.

    :returns: Hidden Div with utility components.
    """
    return html.Div(
        id='utils-div',
        children = [
            notifiers(),
            working_data_stores(),
            upload_data_stores(),
            temporary_download_divs(),
            html.Div(id='interactomics-saint-has-error',children=''),
        ],
        hidden=True
    )

def notifiers() -> html.Div:
    """Create notification components used by callbacks.

    :returns: Hidden Div with notifier children.
    """
    return html.Div(
        id='notifiers-div',
        children=[
            html.Div(id='start-analysis-notifier'),
            html.Div(id='qc-done-notifier'),

            html.Div(id={'type': 'done-notifier',
                     'name': 'proteomics-clustering-done-notifier'}),
            html.Div(id={'type': 'done-notifier',
                     'name': 'proteomics-volcanoes-done-notifier'}),
            html.Div(id={'type': 'done-notifier',
                     'name': 'interactomics-saint-done-notifier'}),
            html.Div(id={'type': 'done-notifier',
                     'name': 'interactomics-pre-enrich-done-notifier'}),
            html.Div(id={'type': 'done-notifier',
                     'name': 'interactomics-enrichment-done-notifier'}),

            # Replace these two with the above
            html.Div(id='workflow-done-notifier'),
            html.Div(id='workflow-volcanoes-done-notifier')
        ],
        hidden=True
    )
