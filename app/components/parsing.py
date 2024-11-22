"""File parsing functions for Proteogyver"""

import base64
import io
import pandas as pd
import numpy as np
from collections.abc import Mapping
import os
import json
from components import db_functions
from components import EnrichmentAdmin as ea


def update_nested_dict(base_dict, update_dict) -> dict:
    for key, value in update_dict.items():
        if isinstance(value, Mapping):
            base_dict[key] = update_nested_dict(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict

def _to_str(val, nan_str: str = '', float_precision: int = 2):
    """Return a string representation of the given integer, rounded float, or otherwise a string.
    """
    if pd.isna(val):
        return nan_str
    if isinstance(val, float):
        if (val % 1 == 0.0):
            return str(int(val))
        else:
            return f'{val:.{float_precision}f}'
    if isinstance(val, int):
        return str(val)
    assert isinstance(val, str)
    return val

def check_numeric(st: str):
    if isinstance(st, np.number):
        return {'success': True, 'value': st}
    val = None
    try:
        sts = st.split('.')
        if len(sts) > 1:
            if sts[-1]=='0':
                val = int(sts[0])
        if val is None:
            val = int(st)
    except ValueError:
        try:
            val = float(st)
        except ValueError:
            return {'success': False, 'value': st}
    return {'success': True, 'value': val}

def unmix_dtypes(df: pd.DataFrame) -> None:
    """Convert mixed dtype columns in the given dataframe to strings.

    Ref: https://stackoverflow.com/a/61826020/
    """
    for col in df.columns:
        if not (orig_dtype := pd.api.types.infer_dtype(df[col])).startswith("mixed"):
            continue
        df[col].fillna(value=np.nan, inplace=True)
        df[col] = df[col].apply(_to_str)
        if (new_dtype := pd.api.types.infer_dtype(df[col])).startswith("mixed"):
            raise TypeError(f"Unable to convert {col} to a non-mixed dtype. Its previous dtype was {orig_dtype} and new dtype is {new_dtype}.")


def parse_parameters(parameters_file: str) -> dict:
    with open(parameters_file, encoding='utf-8') as fil:
        parameters: dict = json.load(fil)
    db_conn = db_functions.create_connection(
        os.path.join(*parameters['Data paths']['Database file']))
    control_sets: list = db_functions.get_from_table(
        db_conn, 'control_sets', select_col='control_set_name')
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
    crapome_sets: list = db_functions.get_from_table(
        db_conn, 'crapome_sets', select_col='crapome_set_name')
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
    parameters['External tools']['SAINT tempdir'] = [
        os.getcwd()]+parameters['External tools']['SAINT tempdir']
    if not 'interactomics' in parameters['workflow parameters'].keys():
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
        'available': ea.get_available(),
        'default': ea.get_default(),
        'disabled': ea.get_disabled()
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
    protein_lengths: dict = None
    if 'Protein Length' in data_table.columns:
        protein_lengths = {}
        for _, row in data_table[[protein_col, 'Protein Length']].drop_duplicates().iterrows():
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
        for _, row in data_table[[protein_col, 'Protein Length']].drop_duplicates().iterrows():
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
        is_spc_table: bool = False,
        max_spc_ever: int = 0
) -> pd.DataFrame:
    """Reads a generic matrix into a data table. Either the returned SPC or intensity table is
    an empty dataframe.

    Matrix is assumed to be SPC matrix, if the maximum value is smaller than max_spc_ever.
    """
    protein_id_column: str = 'Protein.Group'
    table: pd.DataFrame = data_table
    if protein_id_column not in table.columns:
        protein_id_column = table.columns[0]
    protein_lengths: dict = None
    protein_length_cols: list = ['PROTLEN', 'Protein Length', 'Protein.Length']
    protein_length_cols.extend([x.lower() for x in protein_length_cols])
    for plencol in protein_length_cols:
        if plencol in table.columns:
            protein_lengths = {}
            for _, row in table[[protein_id_column, plencol]].drop_duplicates().iterrows():
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
    table.drop(columns=[protein_id_column,], inplace=True)
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


def read_df_from_content(content, filename, lowercase_columns=False) -> pd.DataFrame:
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
    elif f_end == 'xlsx':
        data: pd.DataFrame = pd.read_excel(
            io.BytesIO(decoded_content), engine='openpyxl')
    elif f_end == 'xls':
        data: pd.DataFrame = pd.read_excel(
            io.BytesIO(decoded_content), engine='xlrd')
    if lowercase_columns:
        data.columns = [c.lower() for c in data.columns]
    return data


def read_data_from_content(file_contents, filename, maxpsm) -> pd.DataFrame:
    """Determines and applies the appropriate read function to use for the given data file."""
    table: pd.DataFrame = read_df_from_content(file_contents, filename)
    table.columns = table.columns.astype(str)

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
    intensity_table, spc_table, protein_length_dict = read_funcs[data_type](
        table, **keyword_args)
    intensity_table = remove_duplicate_protein_groups(intensity_table)
    spc_table = remove_duplicate_protein_groups(spc_table)

    info_dict: dict = {
        'protein lengths': protein_length_dict,
        'data type': data_type,
    }
    table_dict: dict = {
        'spc': spc_table.to_json(orient='split'),
        'int': intensity_table.to_json(orient='split'),
    }
    return table_dict, info_dict


def guess_controls(sample_groups: dict, ctrl_indicators: list) -> tuple:
    """Guesses controls from sample groups.

    Any samples with GFP in the name are assumed to be controls.
    :returns: tuple of (list of control sample groups, list of control samples)
    """
    control_groups: list = []
    control_samples: list = []
    for group_name, samples in sample_groups.items():
        might_be_control: bool = False
        for ctrl_ind in ctrl_indicators:
            if ctrl_ind in group_name.lower():
                might_be_control = True
                break
        if might_be_control:
            control_groups.append(group_name)
            control_samples.append(samples)
    return (control_groups, control_samples)

def parse_comparisons(control_group, comparison_data, sgroups) -> list:
    """Parses control group, sample group, and comparison data into a list of pairwise [sample, control] comparisons"""
    comparisons: list = []
    if (control_group is not None) and (control_group != ''):
        comparisons.extend([(sample, control_group)
                            for sample in sgroups.keys()if sample != control_group])
    if comparison_data is not None:
        if len(comparison_data) > 0:
            comparisons.extend(comparison_data)
    return comparisons


def remove_duplicate_protein_groups(data_table: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate protein groups from a given data table by summing the values
    of numerical columns and using the first value of non-numerical columns."""
    aggfuncs: dict = {}
    numerical_columns: set = set(
        data_table.select_dtypes(include=np.number).columns)
    for column in data_table.columns:
        if column in numerical_columns:
            aggfuncs[column] = 'sum'
        else:
            aggfuncs[column] = 'first'
    return data_table.groupby(data_table.index).agg(aggfuncs).replace(0, np.nan)


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
    more_info: dict
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
                data_table: pd.DataFrame = pd.read_json(
                    io.StringIO(table_data), orient='split')
                numeric_columns: set = set(
                    data_table.select_dtypes(include=np.number).columns)
                if len(numeric_columns) >= 3:
                    has_data = True
    new_upload_style['background-color'] = 'green'
    if not has_data:
        new_upload_style['background-color'] = 'red'
    return (new_upload_style, info, tables)


def check_sample_table_column(column, accepted_values) -> str:
    for candidate in accepted_values:
        if candidate == column.lower():
            return column
    return None


def check_required_columns(columns) -> tuple:
    reqs_found: set = set()
    needed_sample_info_columns: set = {('req', ('sample name', 'sample_name')), ('req', (
        'sample group', 'sample_group')), ('opt', ('bait uniprot', 'bait_uniprot', 'bait_id', 'bait id'))}
    infodict: dict = {}
    for n in needed_sample_info_columns:
        for c in columns:
            found: str = check_sample_table_column(c, n[1])
            if found is not None:
                valname: str = n[1][0]
                infodict[valname] = found
                if n[0] == 'req':
                    reqs_found.add(valname)
                break
    return (infodict, reqs_found)


def parse_sample_table(
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
    decoded_table: pd.DataFrame = read_df_from_content(
        data_file_contents, data_file_name)
    indicator_color: str = 'green'
    if not ((decoded_table.shape[1] > 1) and (decoded_table.shape[0] > 1)):
        indicator_color = 'red'
    reqs_found: set
    additional_info: dict
    additional_info, reqs_found = check_required_columns(decoded_table.columns)
    for k, v in additional_info.items():
        info[k] = v
    if len(reqs_found) < 2:
        indicator_color = 'red'
    elif 'bait uniprot' in info:
        indicator_color = 'blue'
    new_upload_style['background-color'] = indicator_color

    return (new_upload_style, info, decoded_table.to_json(orient='split'))


def check_bait(bait_entry: str) -> str:
    """Checks if a string contains a valid bait name.

    :returns: a string representation of the bait. 
    """
    bval: str = ''
    if bait_entry is not None:
        bval = str(bait_entry)
    if (len(bval) == 0) or (bval == 'nan'):
        bval = 'No bait uniprot'
    return bval



def format_data(
        session_uid: str, data_tables: dict, data_info: dict, expdes_table: dict,
        expdes_info: dict, contaminants_to_remove: list, replace_replicate_names: bool,
        use_unique_only: bool, control_indicators: list, bait_id_column_names: list) -> dict:
    """Formats data formats into usable form and produces a data dictionary for later use"""

    intensity_table: pd.DataFrame = pd.read_json(
        io.StringIO(data_tables['int']), orient='split')
    spc_table: pd.DataFrame = pd.read_json(io.StringIO(data_tables['spc']),orient='split')
    expdesign: pd.DataFrame = pd.read_json(io.StringIO(expdes_table),orient='split')

    sample_groups: dict
    discarded_columns: list
    used_columns: list
    sample_groups, discarded_columns, used_columns, expdesign = rename_columns_and_update_expdesign(
        expdesign,
        [intensity_table, spc_table],
        bait_id_column_names,
        replace_names = replace_replicate_names
    )
    spc_table = spc_table[sorted(list(spc_table.columns))]
    if use_unique_only:
        for table in [spc_table, intensity_table]:
            drop_ind = [i for i in table.index if ';' in str(i)]
            if len(drop_ind)>0:
                table.drop(index=drop_ind,inplace=True)
    if len(discarded_columns) > 0:
        for table in [spc_table, intensity_table]:
            table.drop(columns=[c for c in discarded_columns if c in table.columns],inplace=True)
    if len(intensity_table.columns) > 1:
        intensity_table = intensity_table[sorted(
            list(intensity_table.columns))]
        untransformed_intensity_table: pd.DataFrame = intensity_table
        intensity_table = intensity_table.apply(np.log2)
    else:
        untransformed_intensity_table = intensity_table

    wcont_spc_table: pd.DataFrame = spc_table
    wcont_untransformed_intensity_table: pd.DataFrame = untransformed_intensity_table
    wcont_intensity_table: pd.DataFrame = intensity_table
    if len(contaminants_to_remove) > 0:
        spc_table = spc_table.loc[[
            i for i in spc_table.index if i not in contaminants_to_remove]]
        untransformed_intensity_table = untransformed_intensity_table.loc[[
            i for i in untransformed_intensity_table.index if i not in contaminants_to_remove]]
        intensity_table = intensity_table.loc[[
            i for i in intensity_table.index if i not in contaminants_to_remove]]
    spc_table = spc_table.replace(0, np.nan)
    intensity_table = intensity_table.replace(0, np.nan)
    experiment_type = 'Proteomics/Phosphoproteomics'
    if 'bait uniprot' in expdes_info:
        experiment_type = 'Interactomics'
    return_dict: dict = {
        'sample groups': sample_groups,
        'data tables': {
            'raw intensity': untransformed_intensity_table.to_json(orient='split'),
            'spc': spc_table.to_json(orient='split'),
            'intensity': intensity_table.to_json(orient='split'),
            'experimental design': expdesign.to_json(orient='split'),
            'with-contaminants': {
                'raw intensity': wcont_untransformed_intensity_table.to_json(orient='split'),
                'spc': wcont_spc_table.to_json(orient='split'),
                'intensity': wcont_intensity_table.to_json(orient='split'),
            }
        },
        'info': {
            'discarded columns': discarded_columns,
            'used columns': used_columns,
            'data type': data_info['data type'],
            'Expdes based experiment type': experiment_type
        },
        'file info': {
            'Data': {
                'File modified': data_info['Modified time'],
                'File name': data_info['File name']
            },
            'Sample table': {
                'File modified': expdes_info['Modified time'],
                'File name': expdes_info['File name']
            }
        },
        'other': {
            'session name': session_uid,
            'protein lengths': data_info['protein lengths'],
        }
    }
    return_dict['other']['bait uniprots'] = {}
    if 'Bait uniprot' in expdesign.columns:
        for _, row in expdesign.iterrows():
            return_dict['other']['bait uniprots'][row['Sample group']
                                                  ] = check_bait(row['Bait uniprot'])
        return_dict['info']['Expdes based experiment type'] = 'Interactomics'

    if len(intensity_table.columns) < 2:
        return_dict['data tables']['table to use'] = 'spc'
        return_dict['other']['all proteins'] = list(spc_table.index)
    else:
        return_dict['data tables']['table to use'] = 'intensity'
        return_dict['other']['all proteins'] = list(intensity_table.index)

    return_dict['sample groups']['guessed control samples'] = guess_controls(
        sample_groups, control_indicators)

    return return_dict


def remove_from_table(table_name, table, discard_samples):
    if table_name == 'experimental design':
        table_without_discarded_samples = table[
            ~table['Sample name'].isin(discard_samples)
        ]
    else:
        table_without_discarded_samples = table[
            [c for c in table.columns if c not in discard_samples]
        ]
    return table_without_discarded_samples


def delete_samples(discard_samples, data_dictionary) -> dict:
    for table_name, table_json in data_dictionary['data tables'].items():
        if table_name == 'table to use':
            continue
        elif table_name == 'with-contaminants':
            for real_table_name, table_json in data_dictionary['data tables'][table_name].items():
                table_without_discarded_samples: pd.DataFrame = remove_from_table(
                    real_table_name,
                    pd.read_json(io.StringIO(table_json),orient='split'),
                    discard_samples
                )
                data_dictionary['data tables']['with-contaminants'][real_table_name] = table_without_discarded_samples.to_json(
                    orient='split'
                )
        else:
            table_without_discarded_samples: pd.DataFrame = remove_from_table(
                table_name,
                pd.read_json(io.StringIO(table_json),orient='split'),
                discard_samples
            )
            data_dictionary['data tables'][table_name] = table_without_discarded_samples.to_json(
                orient='split'
            )
    sg_dict: dict = {'norm': {}, 'rev': {}}
    for sample_group_name, sample_group_samples in data_dictionary['sample groups']['norm'].items():
        group_samples: list = [
            s_name for s_name in sample_group_samples if s_name not in discard_samples]
        if len(group_samples) == 0:
            continue
        sg_dict['norm'][sample_group_name] = group_samples
    for group, samples in sg_dict['norm'].items():
        for sample in samples:
            sg_dict['rev'][sample] = group
    data_dictionary['sample groups'] = sg_dict
    data_dictionary['user-discarded samples'] = discard_samples

    return data_dictionary

def clean_sample_names(expdesign: pd.DataFrame, bait_id_column_names: list) -> pd.DataFrame:
    """Clean and validate the experimental design dataframe."""
    # Remove rows with missing required values
    expd_columns = ['Sample group','Sample name']
    expdesign = expdesign[~(expdesign[expd_columns].isna().sum(axis=1)>0)].copy()
    expd_columns.extend([c for c in expdesign.columns if c not in expd_columns])
    # Convert all values to strings
    for col in expd_columns:
        expdesign[col] = expdesign[col].apply(_to_str)
    # Remove file paths from sample names (handles both Windows and Unix paths)
    expdesign.loc[:, 'Sample name'] = expdesign['Sample name'].apply(
        lambda x: x.rsplit('\\', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1]
    )
    
    # Standardize bait column name if it exists
    for bid in bait_id_column_names:
        matching_cols = [c for c in expdesign.columns if c.lower().strip() == bid]
        if matching_cols:
            expdesign.rename(columns={matching_cols[0]: 'Bait uniprot'}, inplace=True)
            break
    return expdesign
def clean_column_name(col_name: str) -> str:
    """Remove file paths and extensions from column names."""
    col = col_name.rsplit('\\', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1]
    return col.rsplit('.d', maxsplit=1)[0]

def format_sample_group_name(sample_group) -> str:
    """Format sample group names, handling numeric cases."""
    if pd.isna(sample_group):
        return None
    
    try_num = check_numeric(sample_group)
    if try_num['success']:
        return f'SampleGroup_{try_num["value"]}'
    return str(sample_group)

def generate_replicate_name(group_name: str, sample_name: str, existing_names: set, replace_names: bool) -> str:
    """Generate unique replicate names."""
    if replace_names:
        i = 1
        while f'{group_name}_Rep_{i}' in existing_names:
            i += 1
        return f'{group_name}_Rep_{i}'
    else:
        basename = clean_column_name(sample_name)
        i = 0
        while f'{basename}_{i}' in existing_names:
            i += 1
        return f'{basename}_{i}' if i > 0 else basename
    
def rename_columns_and_update_expdesign(
        expdesign: pd.DataFrame,
        tables: list,
        bait_id_column_names: list,
        replace_names: bool = True) -> tuple:
    """Rename columns and update experimental design table.
    
    Returns:
        tuple: (sample_groups, discarded_columns, used_columns, expdesign)
    """
    # Initial cleanup
    expdesign = clean_sample_names(expdesign, bait_id_column_names)
    discarded_columns = []
    sample_group_columns = {}
    column_mappings = []  # List of dicts for each table's column mappings
    
    # First pass: Map original columns to cleaned names and group assignments
    for table in tables:
        if len(table.columns) < 2:
            column_mappings.append({})
            continue
            
        table_mapping = {}
        for col in table.columns:
            clean_col = clean_column_name(col)
            
            # Skip if column not in experimental design
            if clean_col not in expdesign['Sample name'].values:
                discarded_columns.append(clean_col)
                continue
                
            # Get and format sample group
            sample_group = expdesign[expdesign['Sample name'] == clean_col].iloc[0]['Sample group']
            group_name = format_sample_group_name(sample_group)
            if not group_name:
                continue
                
            # Initialize group if needed
            if group_name not in sample_group_columns:
                sample_group_columns[group_name] = [[] for _ in tables]
            
            table_mapping[col] = {'clean_name': clean_col, 'group': group_name}
            
        column_mappings.append(table_mapping)
    
    # Second pass: Generate final column names and build sample groups
    sample_groups = {'norm': {}, 'rev': {}}
    used_columns = [{} for _ in tables]
    
    for table_idx, mapping in enumerate(column_mappings):
        final_names = {}
        for orig_col, info in mapping.items():
            group = info['group']
            new_name = generate_replicate_name(
                group, 
                info['clean_name'],
                set(final_names.values()), 
                replace_names
            )
            
            final_names[orig_col] = new_name
            
            if group not in sample_groups['norm']:
                sample_groups['norm'][group] = []
            sample_groups['norm'][group].append(new_name)
            sample_groups['rev'][new_name] = group
            
            used_columns[table_idx][new_name] = orig_col
            
        # Apply renames to table
        tables[table_idx] = tables[table_idx].rename(columns=final_names)
    
    # Get rid of duplicates introduced due to multiple tables being processed in previous step
    for group in sample_groups['norm']:
        sample_groups['norm'][group] = sorted(
            list(
                set(sample_groups['norm'][group])
            )
        )
    # Final cleanup: Remove unused samples from expdesign
    used_cols = set().union(*[set(table.columns) for table in tables if len(table.columns) > 0])
    expdesign = expdesign[expdesign['Sample name'].isin(used_cols)]
    
    return (sample_groups, discarded_columns, used_columns, expdesign)


def check_comparison_file(file_contents, file_name, sgroups, new_upload_style) -> list:
    indicator: str = 'green'
    try:
        comparisons: list = []
        dataframe: pd.DataFrame = read_df_from_content(
            file_contents, file_name, lowercase_columns=True)
        scol: str = 'sample'
        ccol: str = 'control'
        if ('sample' not in dataframe.columns) or ('control' not in dataframe.columns):
            scol, ccol = dataframe.columns[:2]
        for _, row in dataframe.iterrows():
            samplename: str = row[scol]
            controlname: str = row[ccol]
            try_num = check_numeric(samplename)
            if try_num['success']:
                samplename = f'SampleGroup_{try_num["value"]}'
                
            try_num = check_numeric(controlname)
            if try_num['success']:
                controlname = f'SampleGroup_{try_num["value"]}'
            else:
                controlname: str = str(controlname)
            # parse sample and control names based on the same rules as in parsing of the group names. Here we can do a lazier version and just try the SampleGroup_ format, if the group is not found to begin with.
            if samplename not in sgroups:
                samplename = f'SampleGroup_{samplename}'
            if samplename not in sgroups:
                continue
            if controlname not in sgroups:
                controlname = f'SampleGroup_{controlname}'
            if controlname not in sgroups:
                continue
            comparisons.append([samplename, controlname])
        if len(comparisons) == 0:
            indicator = 'red'
        elif len(comparisons) != dataframe.shape[0]:
            indicator = 'yellow'
    except AttributeError as e:  # If content is None, we get an attribute error.
        indicator = 'grey'
    new_upload_style['background-color'] = indicator
    return (new_upload_style, comparisons)


def validate_basic_inputs(*args, fail_on_None: bool = True) -> bool:
    """Validates the basic inputs of proteogyver"""
    not_valid: bool = False
    if fail_on_None:
        for arg in args:
            if arg is None:
                not_valid = True
    for style_arg in args[-2:]:
        if style_arg['background-color'] == 'red':
            not_valid = True
    return not_valid
