"""Infrastructure components for Proteogyver"""

from dash import dcc, html
import os
from plotly import io as pio
from plotly import graph_objects as go
import shutil
import json
import pandas as pd
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


data_store_export_configuration: dict = {
    'uploaded-data-table-info-data-store': ['json', 'Debug', '', ''],
    'uploaded-data-table-data-store': ['xlsx', 'Data', 'Input data tables', 'upload-split'],
    'uploaded-sample-table-info-data-store': ['json', 'Debug', '', ''],
    'uploaded-sample-table-data-store': ['xlsx', 'Data', 'Input data tables', 'Uploaded expdesign;Sheet 3'],
    'upload-data-store': ['json', 'Debug', '', ''],
    'replicate-colors-data-store': ['json', 'Debug', '', ''],
    'replicate-colors-with-contaminants-data-store': ['json', 'Debug', '', ''],
    'discard-samples-data-store': ['json', 'Debug', '', ''],
    'count-data-store': ['xlsx', 'Data', 'Summary data', 'Protein counts;'],
    'coverage-data-store': ['xlsx', 'Data', 'Summary data', 'Protein coverage;'],
    'reproducibility-data-store': ['json', 'Data', 'Reproducibility data', ''],
    'missing-data-store': ['xlsx', 'Data', 'Summary data', 'Missing counts;'],
    'sum-data-store': ['xlsx', 'Data', 'Summary data', 'Value sums;'],
    'mean-data-store': ['xlsx', 'Data', 'Summary data', 'Value means;'],
    'distribution-data-store': ['xlsx', 'Data', 'Summary data', 'Value distribution;'],
    'commonality-data-store': ['json', 'Data', 'Commonality data', ''],
    'proteomics-na-filtered-data-store': ['xlsx', 'Data', 'Proteomics data tables', 'NA filtered data;Sheet 2'],
    'proteomics-normalization-data-store': ['xlsx', 'Data', 'Proteomics data tables', 'NA-Normalized data;Sheet 1'],
    'proteomics-imputation-data-store': ['xlsx', 'Data', 'Proteomics data tables', 'NA-Norm-Imputed data;Sheet 0'],
    'proteomics-distribution-data-store': ['xlsx', 'Data', 'Summary data', 'sample distribution;'],
    'proteomics-pca-data-store': ['xlsx', 'Data', 'Figure data', 'Proteomics PCA;'],
    'proteomics-clustermap-data-store': ['xlsx', 'Data', 'Figure data', 'Proteomics Clustermap;'],
    'proteomics-volcano-data-store': ['xlsx', 'Data', 'Significant differences between sample groups', 'volc-split;significants [sg] vs [cg]'],
    'interactomics-saint-input-data-store': ['xlsx', 'Data', 'SAINT input', 'saint-split'],
    'interactomics-enrichment-data-store': ['xlsx', 'Data', 'Enrichment', 'enrichment-split'],
    'interactomics-saint-bfdr-histogram-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'interactomics-saint-graph-data-store': ['NO EXPORT', 'NO EXPORT', 'NO EXPORT', 'NO EXPORT'],
    'interactomics-network-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Network data;Sheet 8'],
    'interactomics-pca-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'PCA;Sheet 7'],
    'interactomics-imputed-and-normalized-intensity': ['xlsx', 'Data', 'Interactomics data tables', 'ImpNorm intensities;Sheet 6'],
    'interactomics-saint-crapome-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Crapome;Sheet 5'],
    'interactomics-saint-output-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Saint output;Sheet 4'],
    'interactomics-saint-final-output-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Saint output with crapome;Sheet 3'],
    'interactomics-saint-filtered-output-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Filtered saint output;Sheet 2'],
    'interactomics-saint-filtered-and-intensity-mapped-output-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Filt saint w intensities;Sheet 1'],
    'interactomics-saint-filt-int-known-data-store': ['xlsx', 'Data', 'Interactomics data tables', 'Filt int saint w knowns;Sheet 0'],
    'interactomics-enrichment-information-data-store': ['txt', 'Data', 'Enrichment information', 'enrich-split']
}

DATA_STORE_IDS = list(data_store_export_configuration.keys())


def save_data_stores(data_stores, export_dir) -> dict:
    export_excels: dict = {}
    timestamps: dict = {}
    prev_time: datetime = datetime.now()
    logger.debug(f'save data stores - started: {prev_time}')
    for d in data_stores:
        if not 'data' in d['props']:
            continue
        timestamps[d['props']['id']['name']] = d['props']['modified_timestamp']
        export_format: str
        export_subdir: str
        file_name: str
        file_config: str
        export_format, export_subdir, file_name, file_config = data_store_export_configuration[
            d['props']['id']['name']]
        if export_format == 'NO EXPORT':
            continue
        export_destination: str = os.path.join(export_dir, export_subdir)
        if not os.path.isdir(export_destination):
            os.makedirs(export_destination)
        if export_format == 'json':
            if file_name == '':
                file_name = d['props']['id']['name']
            with open(os.path.join(export_destination, file_name+'.json'), 'w', encoding='utf-8') as fil:
                json.dump(d['props']['data'], fil, indent=2)
        elif export_format == 'txt':
            if file_config == 'enrich-split':
                for enrichment_name, file_contents in d['props']['data']:
                    with open(os.path.join(export_destination, f'{file_name} {enrichment_name}.{export_format}'), 'w', encoding='utf-8') as fil:
                        fil.write(file_contents)
        elif export_format == 'xlsx':
            file_name = os.path.join(
                export_destination, '.'.join([file_name, export_format]))
            if file_name not in export_excels:
                export_excels[file_name] = {}
            if not 'split' in file_config:
                sheet_name: str
                sheet_name, sheet_index = file_config.split(';')
                try:
                    sheet_index: int = int(sheet_index.split()[-1])
                except IndexError:
                    sheet_index = 0
                    while sheet_index in export_excels[file_name]:
                        sheet_index += 1
                if not file_name in export_excels:
                    export_excels[file_name] = {}
                assert sheet_index not in export_excels[
                    file_name], f'DUPLICATE INDEX IN EXCEL EXPORT: {sheet_name} and {export_excels[file_name][sheet_index]["name"]}'
                try:
                    export_excels[file_name][sheet_index] = {
                        'name': sheet_name,
                        'data': pd.read_json(d['props']['data'], orient='split'),
                        'headers': True,
                        'index': True
                    }
                except ValueError:
                    logger.debug(
                        f'save data stores - ValueError: {file_name} {sheet_index} {sheet_name}: {prev_time}')
                    continue
            else:
                if file_config == 'upload-split':
                    sheet_index = 0
                    for key, data_table in d['props']['data'].items():
                        data_table_pd: pd.DataFrame = pd.read_json(
                            data_table, orient='split')
                        if data_table_pd.shape[1] < 2:
                            continue
                        sheet_name = f'Input table {key}'
                        if not file_name in export_excels:
                            export_excels[file_name] = {}
                        assert sheet_index not in export_excels[
                            file_name], f'DUPLICATE INDEX IN EXCEL EXPORT: {sheet_name} and {export_excels[file_name][sheet_index]["name"]}'
                        export_excels[file_name][sheet_index] = {
                            'name': sheet_name,
                            'data': data_table_pd,
                            'headers': True,
                            'index': True,
                        }
                        sheet_index += 1
                elif file_config == 'saint-split':
                    export_excels[file_name] = {
                        i: {'name': key, 'index': False,
                            'data': pd.DataFrame(value), 'headers': False}
                        for i, (key, value) in enumerate(d['props']['data'].items())
                    }
                elif 'volc-split' in file_config:
                    df: pd.DataFrame = pd.read_json(
                        d['props']['data'], orient='split')
                    df_dicts: list = []
                    df_dicts_all: list = []
                    for _, s_c_row in df[['Sample', 'Control']].drop_duplicates().iterrows():
                        sample: str = s_c_row['Sample']
                        control: str = s_c_row['Control']
                        df_dicts.append({
                            'name': f'{sample} vs {control} significant only',
                            'data': df[(df['Sample'] == sample) & (df['Control'] == control) & df['Significant']],
                            'headers': True,
                            'index': True
                        })
                        df_dicts_all.append({
                            'name': f'{sample} vs {control}',
                            'data': df[(df['Sample'] == sample) & (df['Control'] == control)],
                            'headers': True,
                            'index': True
                        })
                    if len(df_dicts) > 20:
                        for df_dict_index, df_dict in enumerate(df_dicts):
                            new_filename = file_name.replace(
                                f'.{export_format}',
                                f'{df_dict["name"]}.{export_format}'
                            )
                            export_excels[new_filename] = {
                                0: df_dict,
                                1: df_dicts_all[df_dict_index]
                            }
                    else:
                        df_dicts.extend(df_dicts_all)
                        export_excels[file_name] = {
                            i: df_dict for i, df_dict in enumerate(df_dicts)
                        }
        logger.debug(
            f'save data stores - export {d["props"]["id"]["name"]} done: {datetime.now() - prev_time}')
        prev_time: datetime = datetime.now()

    no_index: set = {'Uploaded expdesign'}

    logger.debug(
        f'save data stores - writing excels: {datetime.now() - prev_time}')
    prev_time: datetime = datetime.now()
    for excel_name, excel_dict in export_excels.items():
        start_excel_time: datetime = prev_time
        if len(excel_dict.keys()) == 0:
            print('No sheets', excel_name)
            continue
        with pd.ExcelWriter(excel_name, engine="xlsxwriter") as writer:
            # writer = pd.ExcelWriter(excel_name)
            for df_dict_index in sorted(list(excel_dict.keys())):
                if df_dict_index == 'config_info':
                    continue
                dic: dict = excel_dict[df_dict_index]
                index_bool = dic['index']
                if (dic['name'] in no_index):
                    index_bool = False
                logger.debug(
                    f'save data stores - sheet {excel_name}: {dic["name"]} shape: {dic["data"].shape}: {datetime.now() - prev_time}')
                logger.debug(
                    f'save data stores - sheet {excel_name}: {dic["name"]} headers: {dic["headers"]}: {datetime.now() - prev_time}')
                logger.debug(
                    f'save data stores - sheet {excel_name}: {dic["name"]} index: {index_bool}: {datetime.now() - prev_time}')
                dic['data'].to_excel(writer, sheet_name=dic['name'],
                                     header=dic['headers'], index=index_bool)
                logger.debug(
                    f'save data stores - sheet {excel_name}: {dic["name"]} done: {datetime.now() - prev_time}')
                prev_time: datetime = datetime.now()

                # excel_dict[df_dict_index]['data'].to_excel(writer, sheet_name = dic['name'], header = dic['headers'])
            logger.debug(
                f'save data stores - closing writer {excel_name}: {datetime.now() - start_excel_time}')
            prev_time: datetime = datetime.now()
        # writer.close()
        logger.debug(
            f'save data stores - finished with {excel_name}. Took: {datetime.now() - start_excel_time}')
        prev_time: datetime = datetime.now()
    print(json.dumps(timestamps, indent=2))
    return timestamps


def get_all_props(elements, marker_key, match_partial=True) -> list:
    logger.debug(f'fetching props {marker_key}: {datetime.now()}')
    prev_time: datetime = datetime.now()
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
    logger.debug(
        f'fetching props {marker_key} - done: {datetime.now() - prev_time}')
    return ret


def get_all_types(elements, get_types) -> list:
    logger.debug(f'fetching types {get_types}: {datetime.now()}')
    prev_time: datetime = datetime.now()
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
    logger.debug(
        f'fetching types {get_types} - done: {datetime.now() - prev_time}')
    return ret


def save_figures(analysis_divs, export_dir, output_formats) -> None:
    logger.debug(f'saving figures: {datetime.now()}')
    prev_time: datetime = datetime.now()
    headers_and_figures: list = get_all_types(
        analysis_divs, ['h4', 'graph', 'P'])
    figure_names_and_figures: list = []
    for i, header in enumerate(headers_and_figures):
        if i < (len(headers_and_figures)-2):
            if header['type'].lower() == 'h4':
                graph: dict = headers_and_figures[i+1]
                legend: dict = headers_and_figures[i+2]
                if graph['type'].lower() == 'graph':
                    figure: dict = graph['props']['figure']
                    figure_html: str = pio.to_html(
                        figure, config=graph['props']['config'])
                    figure_names_and_figures.append(
                        [header['props']['children'], legend['props']['children'], figure_html, figure])
    logger.debug(
        f'saving figures - figures identified: {len(figure_names_and_figures)}: {datetime.now() - prev_time}')
    prev_time: datetime = datetime.now()

    for name, legend, fig_html, fig in figure_names_and_figures:
        if 'html' in output_formats:
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
            with open(os.path.join(export_dir, f'{name}.html'), 'w', encoding='utf-8') as fil:
                fil.write('\n'.join(new_html))
        for output_format in output_formats:
            if output_format == 'html':
                continue
            go.Figure(fig).write_image(os.path.join(
                export_dir, f'{name}.{output_format}'))
        logger.debug(
            f'saving figures - writing: {name}.{output_format}: {datetime.now() - prev_time}')
        prev_time: datetime = datetime.now()
    logger.debug(f'saving figures - done: {datetime.now() - prev_time}')


def format_nested_list(input_list: list):
    if not isinstance(input_list, list):
        return str(input_list)
    rlist: list = []
    for entry in input_list:
        if isinstance(entry, list):
            rlist.append(format_nested_list(entry))
        else:
            rlist.append(str(entry))
    return ', '.join(rlist)


def save_input_information(input_divs, export_dir) -> None:
    logger.debug(f'saving input info: {datetime.now()}')
    prev_time: datetime = datetime.now()
    these: list = [
        'Slider',
        'Select',
        'Label',
        'Checklist',
        'RadioItems'
    ]
    input_options: list = []
    labels_and_inputs: list = get_all_types(input_divs, these)
    for i, label in enumerate(labels_and_inputs):
        if label['type'] != 'Label':
            continue
        if len(label['props']['children']) < 4:
            continue
        try:
            input_options.append(
                [label['props']['children'], labels_and_inputs[i+1]['props']['value']])
        except KeyError:
            continue
    with open(os.path.join(export_dir, 'Options used in analysis.txt'), 'w', encoding='utf-8') as fil:
        for name, values in input_options:
            val_str: str = format_nested_list(values)
            if len(val_str) == 0:
                val_str = 'None'
            fil.write(f'{name} {val_str}\n')
    logger.debug(f'saving input info - done: {datetime.now() - prev_time}')


def prepare_download(data_stores, analysis_divs, input_divs, cache_dir, session_name, figure_output_formats) -> str:
    logger.debug(f'preparing download: {datetime.now()}')
    prev_time: datetime = datetime.now()
    export_dir: str = os.path.join(*cache_dir, session_name)
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir)
    timestamps: dict = save_data_stores(data_stores, export_dir)
    save_figures(analysis_divs, export_dir, figure_output_formats)
    save_input_information(input_divs, export_dir)
    logger.debug(
        f'preparing download - finished exporting, making zip now: {datetime.now() - prev_time}')
    prev_time: datetime = datetime.now()
    export_zip_name: str = export_dir.rstrip(os.sep) + '.zip'
    if os.path.isfile(export_zip_name):
        os.remove(export_zip_name)
    shutil.make_archive(export_dir.rstrip(os.sep), 'zip', export_dir)
    logger.debug(
        f'preparing download - zip made, removing leftover data: {datetime.now() - prev_time}')
    prev_time: datetime = datetime.now()
    shutil.rmtree(export_dir)
    logger.debug(f'preparing download - done: {datetime.now() - prev_time}')
    return export_zip_name


def data_stores() -> html.Div:
    """Returns all the needed data store components"""
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
    """Returns all the needed data store components"""
    stores: list = []
    for ID_STR in DATA_STORE_IDS:
        if 'uploaded' in ID_STR:
            continue
        stores.append(dcc.Store(id={'type': 'data-store', 'name': ID_STR}))
    return html.Div(id='workflow-stores', children=stores)


def notifiers() -> html.Div:
    """Returns divs used for various callbacks only."""
    return html.Div(
        id='notifiers-div',
        children=[
            html.Div(id='start-analysis-notifier'),
            html.Div(id='qc-done-notifier'),
            html.Div(id='workflow-done-notifier'),
            html.Div(id='workflow-volcanoes-done-notifier')
        ],
        hidden=True
    )
