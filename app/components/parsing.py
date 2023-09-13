"""File parsing functions for Proteogyver"""

import base64
import io
import pandas as pd
import numpy as np
from collections.abc import Mapping
import os
import json
from components import db_functions
from importlib import util as import_util

def update_nested_dict(base_dict, update_dict) -> dict:
    for key, value in update_dict.items():
        if isinstance(value, Mapping):
            base_dict[key] = update_nested_dict(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict
def parse_parameters(parameters_file: str) -> dict:
    with open(parameters_file, encoding='utf-8') as fil:
        parameters: dict = json.load(fil)
    db_conn = db_functions.create_connection(os.path.join(*parameters['Data paths']['Database file']))
    control_sets: list = db_functions.get_from_table(db_conn, 'control_sets', select_col = 'control_set_name')
    default_control_sets: list = db_functions.get_from_table(
        db_conn,
        'control_sets',
        'control_set_name',
        'is_default',
        1
        )
    disabled_control_sets: list = db_functions.get_from_table(
        db_conn,
        'control_sets',
        'control_set_name',
        'is_disabled',
        1
        )
    crapome_sets: list = db_functions.get_from_table(db_conn, 'control_sets', select_col = 'control_set_name')
    default_crapome_sets: list = db_functions.get_from_table(
        db_conn,
        'crapome_sets',
        'crapome_set_name',
        'is_default',
        1
        )
    disabled_crapome_sets: list = db_functions.get_from_table(
        db_conn,
        'crapome_sets',
        'crapome_set_name',
        'is_disabled',
        1
        )
    db_conn.close()
    handler_names: list = []
    enrichments: list = []
    default_enrichments: list = []
    disabled_enrichments: list = []
    for enricher_mod in os.listdir(os.path.join(*parameters['Module paths']['Enrichers'])):
        if enricher_mod.split('.')[-1] != 'py':
            continue
        filepath = os.path.join(*parameters['Module paths']['Enrichers'], enricher_mod)
        spec = import_util.spec_from_file_location(
            'module.name', filepath)
        api_module = import_util.module_from_spec(spec)
        spec.loader.exec_module(api_module)
        handler = api_module.handler()

        handler_names.append((handler.nice_name, filepath))
        handler_enrichments:list = [
            f'{handler.nice_name}: {e_type}' for e_type in handler.get_available()['enrichment']
        ]
        enrichments.extend(handler_enrichments)
        if not 'david' in handler.nice_name.lower():
            default_enrichments.extend([
                f'{handler.nice_name}: {e_type}' for e_type in handler.get_default_panel()
            ])
    parameters['workflow parameters']['interactomics'] = {}
    parameters['workflow parameters']['interactomics']['crapome'] = {
        'available': crapome_sets, 
        'disabled': disabled_crapome_sets,
        'default': default_crapome_sets
    }
    parameters['workflow parameters']['interactomics']['controls'] = {
        'available': control_sets,
        'disabled': disabled_control_sets,
        'default': default_control_sets
    }
    parameters['workflow parameters']['interactomics']['enrichment'] = {
        'available': enrichments,
        'default': default_enrichments,
        'disabled': disabled_enrichments,
        'handler names': handler_names
    }

    return parameters

def get_distribution_title(used_table_type: str) -> str:
    if used_table_type == 'intensity':
        title: str = 'Log2 transformed value distribution'
    else:
        title = 'Value distribution'
    return title

def read_dia_nn(data_table: pd.DataFrame) -> pd.DataFrame:
    """Reads dia-nn report file into an intensity matrix"""
    protein_col: str = 'Protein.Group'
    protein_lengths:dict = None
    if 'Protein Length' in data_table.columns:
        protein_lengths = {}
        for _,row in data_table[[protein_col,'Protein Length']].drop_duplicates().iterrows():
            protein_lengths[row[protein_col]] = row['Protein Length']
    is_report: bool = False
    for column in data_table.columns:
        if column == 'Run':
            is_report = True
            break
    if is_report:
        table: pd.DataFrame = pd.pivot_table(
            data=data_table, index=protein_col, columns='Run', values='PG.MaxLFQ')
    else:
        data_cols: list = []
        for column in data_table.columns:
            col: list = column.split('.')
            if col[-1] == 'd':
                data_cols.append(column)
        if len(data_cols) == 0:
            gather: bool = False
            for column in data_table.columns:
                if gather:
                    data_cols.append(column)
                elif column == 'First.Protein.Description':
                    gather = True
        table: pd.DataFrame = data_table[data_cols]
        table.index = data_table['Protein.Group']
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    return [table, pd.DataFrame({'No data': ['No data']}), protein_lengths]

def read_fragpipe(data_table: pd.DataFrame) -> pd.DataFrame:
    """Reads a fragpipe report into spc and intensity tables (if intensity values are available)"""
    intensity_cols: list = []
    spc_cols: list = []
    uniq_intensity_cols: list = []
    uniq_spc_cols: list = []
    has_maxlfq: bool = False
    for column in data_table.columns:
        if 'Total' in column:
            continue
        if 'Combined' in column:
            continue
        if 'Intensity' in column:
            if 'maxlfq' in column.lower():
                has_maxlfq = True
            if 'unique' in column.lower():
                uniq_intensity_cols.append(column)
            else:
                intensity_cols.append(column)
        elif 'Spectral Count' in column:
            if 'unique' in column.lower():
                uniq_spc_cols.append(column)
            else:
                spc_cols.append(column)
    if len(uniq_intensity_cols) > 0:
        intensity_cols = uniq_intensity_cols
    if len(uniq_spc_cols) > 0:
        spc_cols = uniq_spc_cols
    if has_maxlfq:
        intensity_cols = [i for i in intensity_cols if 'maxlfq' in i.lower()]
    protein_col: str = 'Protein ID'
    if 'Protein Length' in data_table.columns:
        protein_lengths: dict = {}
        for _,row in data_table[[protein_col,'Protein Length']].drop_duplicates().iterrows():
            protein_lengths[row[protein_col]] = row['Protein Length']
    else:
        protein_lengths = None
    table: pd.DataFrame = data_table
    # Replace zeroes with missing valuese
    table.replace(0, np.nan, inplace=True)
    table.index = table[protein_col]
    intensity_table: pd.DataFrame = table[intensity_cols]
    replace_str: str = ''
    if len(uniq_spc_cols) > 0:
        replace_str = 'Unique '
    spc_table: pd.DataFrame = table[spc_cols].rename(
        columns={ic: ic.replace(f'{replace_str}Spectral Count', '').strip()
                for ic in spc_cols}
    )
    replace_str = ''
    if len(uniq_intensity_cols) > 0:
        replace_str = 'Unique '
    if intensity_table[intensity_cols[0:2]].sum().sum() == 0:
        intensity_table = pd.DataFrame({'No data': ['No data']})
    else:
        intensity_table.rename(
        columns={ic: ic.replace(f'{replace_str}Intensity', '').replace('MaxLFQ', '').strip()
                for ic in intensity_cols},
        inplace=True)
    return (intensity_table, spc_table, protein_lengths)

def read_matrix(
        data_table: pd.DataFrame,
        is_spc_table:bool=False,
        max_spc_ever:int=0
        ) -> pd.DataFrame:
    """Reads a generic matrix into a data table. Either the returned SPC or intensity table is
    an empty dataframe.
    
    Matrix is assumed to be SPC matrix, if the maximum value is smaller than max_spc_ever.
    """
    protein_id_column: str = 'Protein.Group'
    table: pd.DataFrame = data_table
    if protein_id_column not in table.columns:
        protein_id_column = table.columns[0]
    protein_lengths:dict = None
    protein_length_cols:list = ['PROTLEN','Protein Length','Protein.Length']
    protein_length_cols.extend([x.lower() for x in protein_length_cols])
    for plencol in protein_length_cols:
        if plencol in table.columns:
            protein_lengths = {}
            for _,row in table[[protein_id_column,plencol]].drop_duplicates().iterrows():
                protein_lengths[row[protein_id_column]] = row[plencol]
            table = table.drop(columns=plencol)
            break
    table.index = table[protein_id_column]
    table = table[table.index != 'na']
    for column in table.columns:
        isnumber: bool = np.issubdtype(table[column].dtype, np.number)
        if not isnumber:
            try:
                table[column] = pd.to_numeric(table[column])
            except ValueError:
                continue
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    table.drop(columns=[protein_id_column,],inplace=True)
    spc_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
    intensity_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
    if is_spc_table:
        spc_table = table
    else:
        if table.select_dtypes(include=[np.number]).max().max() <= max_spc_ever:
            spc_table = table
        else:
            intensity_table = table
    return (intensity_table, spc_table, protein_lengths)

def read_df_from_content(content, filename, lowercase_columns = False) -> pd.DataFrame:
    """Reads a dataframe from uploaded content.
    
    Filenames ending with ".csv" are read as comma separated, filenames ending with ".tsv", ".tab"
    or ".txt" are read as tab-delimed files, and ".xlsx" and ".xls" are read as excel files.
    Filename ending identification is case-insensitive.
    """
    _: str
    content_string: str
    _, content_string = content.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    f_end: str = filename.rsplit('.', maxsplit=1)[-1].lower()
    data = None
    if f_end == 'csv':
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), index_col=False)
    elif f_end in ['tsv', 'tab', 'txt']:
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), sep='\t', index_col=False)
    elif f_end in ['xlsx', 'xls']:
        data: pd.DataFrame = pd.read_excel(io.StringIO(decoded_content))
    if lowercase_columns:
        data.columns = [c.lower() for c in data.columns]
    return data

def read_data_from_content(file_contents, filename, maxpsm) -> pd.DataFrame:
    """Determines and applies the appropriate read function to use for the given data file."""
    table: pd.DataFrame = read_df_from_content(file_contents, filename)
    read_funcs: dict[tuple[str, str]] = {
            ('DIA', 'DIA-NN'): read_dia_nn,
            ('DDA', 'FragPipe'): read_fragpipe,
            ('DDA/DIA', 'Unknown'): read_matrix,
        }
    data_type: tuple = None
    keyword_args: dict = {}
    if 'Protein.Ids' in table.columns:
        if 'First.Protein.Description' in table.columns:
            data_type = ('DIA', 'DIA-NN')
    elif 'Top Peptide Probability' in table.columns:
        if 'Protein Existence' in table.columns:
            data_type = ('DDA', 'FragPipe')
    if data_type is None:
        data_type = ('DDA/DIA', 'Unknown')
        keyword_args['max_spc_ever'] = maxpsm
    intensity_table: pd.DataFrame
    spc_table: pd.DataFrame
    protein_length_dict: dict
    intensity_table, spc_table, protein_length_dict = read_funcs[data_type](table, **keyword_args)
    intensity_table = remove_duplicate_protein_groups(intensity_table)
    spc_table = remove_duplicate_protein_groups(spc_table)

    info_dict: dict = {
        'protein lengths': protein_length_dict,
        'data type': data_type
        }
    table_dict: dict = {
        'spc': spc_table.to_json(orient='split'),
        'int': intensity_table.to_json(orient='split'),
    }
    return table_dict, info_dict

def guess_control_samples(sample_names: list) -> list:
    possible_control_samples: list = []
    for group_name in sample_names:
        if 'gfp' in group_name.lower():
            possible_control_samples.append(group_name)
    return possible_control_samples

def parse_comparisons(control_group, comparison_file, comparison_file_name, sgroups) -> list:
    comparisons: list = []
    if control_group is None:
        dataframe: pd.DataFrame = read_df_from_content(comparison_file, comparison_file_name, lowercase_columns = True)
        scol:str = 'sample'
        ccol:str = 'control'
        if ('sample' not in dataframe.columns) or ('control' not in dataframe.columns):
            scol, ccol = dataframe.columns[:2]
        for _, row in dataframe.iterrows():
            comparisons.append([row[scol], row[ccol]])
    else:
        comparisons = [(sample, control_group) for sample in sgroups.keys()if sample != control_group]
    return comparisons

def remove_duplicate_protein_groups(data_table: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate protein groups from a given data table by summing the values
    of numerical columns and using the first value of non-numerical columns."""
    aggfuncs: dict = {}
    numerical_columns: set = set(data_table.select_dtypes(include=np.number).columns)
    for column in data_table.columns:
        if column in numerical_columns:
            aggfuncs[column] = sum
        else:
            aggfuncs[column] = 'first'
    return data_table.groupby(data_table.index).agg(aggfuncs).replace(0,np.nan)

def parse_data_file(
        data_file_contents,
        data_file_name,
        data_file_modified_data,
        new_upload_style,
        parameters) -> tuple:
    """Parses the given data file and returns a tuple of 
    (new_upload_style, file info dict, tables dict with tables in split orientation)
    """
    info: dict = {
        'Modified time': data_file_modified_data,
        'File name': data_file_name
    }
    tables: dict
    more_info:dict
    tables, more_info = read_data_from_content(
        data_file_contents,
        data_file_name,
        parameters['Maximum psm ever theoretically encountered']
        )
    for key, value in more_info.items():
        info[key] = value
    has_data: bool = False
    for _, table_data in tables.items():
        if isinstance(table_data, str):
            if table_data.count('No data') != 2:
                data_table: pd.DataFrame = pd.read_json(table_data,orient='split')
                numeric_columns: set = set(data_table.select_dtypes(include=np.number).columns)
                if len(numeric_columns) >= 3:
                    has_data = True
    new_upload_style['background-color'] = 'green'
    if not has_data:
        new_upload_style['background-color'] = 'red'
    return (new_upload_style, info, tables)

def parse_generic_table(
        data_file_contents,
        data_file_name,
        data_file_modified_data,
        new_upload_style) -> tuple:
    """Parses the given table file and returns a tuple of 
    (new_upload_style, file info dict, table_json in split orientation)
    """
    info: dict = {
        'Modified time': data_file_modified_data,
        'File name': data_file_name
    }
    decoded_table: pd.DataFrame = read_df_from_content(data_file_contents, data_file_name)
    indicator_color: str = 'green'
    if not ( (decoded_table.shape[1]>1) and (decoded_table.shape[0]>1) ):
        indicator_color = 'red'
    needed_column_count: int = len([
            colname for colname in ['Sample name','Sample group'] if colname in decoded_table.columns
        ])
    if needed_column_count != 2:
        indicator_color = 'red'
    new_upload_style['background-color'] = indicator_color

    return (new_upload_style, info, decoded_table.to_json(orient='split'))
