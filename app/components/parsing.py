"""File parsing functions for Proteogyver.

This module contains functions for parsing and processing various data file formats,
handling data type conversions, and managing parameter configurations.
"""

from typing import Any, Dict, List, Tuple, Union, Optional, Set
import base64
import io
import pandas as pd
import numpy as np
from collections.abc import Mapping
import os
from pathlib import Path
from components import db_functions, text_handling
from components import EnrichmentAdmin
from components.tools import utils

def update_nested_dict(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Updates a nested dictionary with values from another dictionary.
    
    Args:
        base_dict (dict): Base dictionary to update
        update_dict (dict): Dictionary containing update values
        
    Returns:
        dict: Updated base dictionary
        
    Notes:
        - Recursively handles nested dictionary structures
        - Creates new nested dictionaries if they don't exist in base_dict
    """
    for key, value in update_dict.items():
        if isinstance(value, Mapping):
            base_dict[key] = update_nested_dict(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict

def _to_str(val: Any, nan_str: str = '', float_precision: int = 2) -> str:
    """Returns a string representation of numeric or string values.
    
    Args:
        val: Value to convert to string
        nan_str (str, optional): String to use for NaN values. Defaults to ''
        float_precision (int, optional): Decimal places for float values. Defaults to 2
        
    Returns:
        str: String representation of the value
        
    Notes:
        - Handles NaN values
        - Converts integers without decimal places
        - Formats floats with specified precision
        - Validates string type for non-numeric values
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

def check_numeric(st: Union[str, np.number]) -> Dict[str, Union[bool, Union[int, float, str]]]:
    """Checks if a string can be converted to a numeric value.
    
    Args:
        st (str): String to check for numeric conversion
        
    Returns:
        dict: Contains:
            - success (bool): Whether conversion was successful
            - value: Converted numeric value or original string
            
    Notes:
        - Attempts integer conversion first
        - Falls back to float conversion
        - Handles special case of integers with .0 suffix
    """
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
    """Converts mixed dtype columns in a dataframe to strings.
    
    Args:
        df (pd.DataFrame): DataFrame to process
        
    Returns:
        None: Modifies DataFrame in place
        
    Notes:
        - Identifies columns with mixed data types
        - Converts mixed columns to consistent string format
        - Raises TypeError if conversion fails
        
    Reference:
        https://stackoverflow.com/a/61826020/
    """
    for col in df.columns:
        if not (orig_dtype := pd.api.types.infer_dtype(df[col])).startswith("mixed"):
            continue
        df[col].fillna(value=np.nan, inplace=True)
        df[col] = df[col].apply(_to_str)
        if (new_dtype := pd.api.types.infer_dtype(df[col])).startswith("mixed"):
            raise TypeError(f"Unable to convert {col} to a non-mixed dtype. Its previous dtype was {orig_dtype} and new dtype is {new_dtype}.")


def parse_parameters(parameters_file: Union[str, Path]) -> Dict[str, Any]:
    """Parses and enriches parameters from a TOML configuration file.
    
    Args:
        parameters_file (Union[str, Path]): Path to parameters TOML file
        
    Returns:
        dict: Enriched parameters dictionary containing:
            - Original parameters
            - Control set configurations
            - CRAPome set configurations
            - Enrichment configurations
            
    Notes:
        - Loads database configurations
        - Retrieves control and CRAPome set information
        - Adds enrichment analysis options
        - Updates SAINT temporary directory path
    """
    parameters = utils.read_toml(Path(parameters_file))
    
    if not os.path.exists(os.path.join(*parameters['Data paths']['Database file'])):
        parameters['Data paths']['Database file'] = parameters['Data paths']['Minimal database file']
    
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
    ea = EnrichmentAdmin.EnrichmentAdmin(parameters_file)
    parameters['workflow parameters']['interactomics']['enrichment'] = {
        'available': ea.get_available(),
        'default': ea.get_default(),
        'disabled': ea.get_disabled()
    }

    return parameters


def get_distribution_title(used_table_type: str) -> str:
    """Gets appropriate title for value distribution plots.
    
    Args:
        used_table_type (str): Type of table being plotted
        
    Returns:
        str: Plot title indicating value type and transformation
    """
    if used_table_type == 'intensity':
        title: str = 'Log2 transformed value distribution'
    else:
        title = 'Value distribution'
    return title


def read_dia_nn(data_table: pd.DataFrame) -> List[Union[pd.DataFrame, Dict[str, int]]]:
    """Reads DIA-NN report file into an intensity matrix.
    
    Args:
        data_table (pd.DataFrame): Raw DIA-NN data table
        
    Returns:
        list: Contains:
            - pd.DataFrame: Processed intensity matrix
            - pd.DataFrame: Empty placeholder table
            - dict: Protein length information if available
            
    Notes:
        - Handles both report and matrix formats
        - Extracts protein length information
        - Replaces zeros with NaN values
        - Pivots data if in report format
    """
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


def read_fragpipe(data_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, int]]]:
    """Reads FragPipe report into spectral count and intensity tables.
    
    Args:
        data_table (pd.DataFrame): Raw FragPipe data table
        
    Returns:
        tuple: Contains:
            - pd.DataFrame: Intensity table
            - pd.DataFrame: Spectral count table
            - dict: Protein length information if available
            
    Notes:
        - Identifies intensity and spectral count columns
        - Handles unique peptide counts
        - Supports MaxLFQ intensity values
        - Replaces zeros with NaN values
    """
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


def read_matrix(data_table: pd.DataFrame, is_spc_table: bool = False, 
                max_spc_ever: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[Dict[str, int]]]:
    """Reads a generic matrix into a data table.
    
    Args:
        data_table (pd.DataFrame): Input data matrix
        is_spc_table (bool, optional): Whether matrix contains spectral counts. 
            Defaults to False
        max_spc_ever (int, optional): Maximum expected spectral count value. 
            Defaults to 0
            
    Returns:
        tuple: Contains:
            - pd.DataFrame: Intensity table
            - pd.DataFrame: Spectral count table
            - dict: Protein length information if available
            
    Notes:
        - Automatically detects spectral count tables
        - Handles protein length information
        - Removes non-numeric columns
        - Replaces zeros with NaN values
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
    drop_cols: list = []
    # Remove non-numeric columns and convert numeric-looking columns to numeric
    for column in table.columns:
        isnumber: bool = np.issubdtype(table[column].dtype, np.number)
        if not isnumber:
            try:
                table[column] = pd.to_numeric(table[column])
            except ValueError:
                drop_cols.append(column)
                continue
    if table.select_dtypes(include=[np.number]).max().max() <= max_spc_ever:
        is_spc_table = True
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    table.drop(columns=drop_cols, inplace=True)
    spc_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
    intensity_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
    if is_spc_table:
        spc_table = table
    else:
        intensity_table = table
    return (intensity_table, spc_table, protein_lengths)


def read_df_from_content(content: str, filename: str, lowercase_columns: bool = False) -> pd.DataFrame:
    """Reads a dataframe from uploaded file content.
    
    Args:
        content: Base64 encoded file content
        filename (str): Original filename with extension
        lowercase_columns (bool, optional): Whether to convert column names to 
            lowercase. Defaults to False
            
    Returns:
        pd.DataFrame: Parsed data frame
        
    Notes:
        - Supports CSV, TSV, TXT, XLS, and XLSX formats
        - Handles UTF-8 encoded text files
        - Uses appropriate pandas reader based on file extension
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


def read_data_from_content(file_contents: str, filename: str, maxpsm: int) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Determines and applies the appropriate read function for the given data file.
    
    Args:
        file_contents: The contents of the uploaded file
        filename (str): Name of the uploaded file
        maxpsm (int): Maximum theoretical PSM value for spectral counting
        
    Returns:
        tuple: Contains:
            - dict: Table dictionary with:
                - spc (str): Spectral count table in JSON split format
                - int (str): Intensity table in JSON split format
            - dict: Info dictionary with:
                - protein lengths (dict): Protein length information
                - Data type (str): DDA or DIA
                - Data source guess (str): Software source of the data
                
    Notes:
        - Automatically detects file format based on column headers
        - Supports DIA-NN, FragPipe and generic matrix formats
        - Removes duplicate protein groups
        - Returns tables in JSON split format for serialization
    """
    table: pd.DataFrame = read_df_from_content(file_contents, filename)
    table.columns = [text_handling.replace_accent_and_special_characters(str(x), allow_space=True, make_lowercase=False) for x in table.columns]
    # Validation: initialize containers
    warnings: list[str] = []
    validation: dict[str, Any] = {}
    # Pre-parse sanity metrics on initial table
    try:
        validation.update({
            'rows_initial': int(table.shape[0]),
            'cols_initial': int(table.shape[1]),
            'numeric_cols_initial': int(table.select_dtypes(include=[np.number]).shape[1]),
        })
        if validation['rows_initial'] == 0:
            warnings.append('Empty file: 0 rows')
        if validation['cols_initial'] < 2:
            warnings.append('Suspiciously few columns (<2)')
        if validation['numeric_cols_initial'] == 0:
            warnings.append('No numeric columns detected')
    except Exception:
        # Be conservative: do not fail parsing due to validation
        pass

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
    # Post-reader validation metrics for intensity and spc tables
    try:
        for name, df in [('intensity', intensity_table), ('spc', spc_table)]:
            is_placeholder: bool = (list(df.columns) == ['No data']) and (df.shape == (1, 1))
            nrows: int = int(df.shape[0])
            ncols: int = int(df.shape[1])
            num_df: pd.DataFrame = df.select_dtypes(include=[np.number])
            nnum: int = int(num_df.shape[1])
            non_nan: int = int(num_df.count().sum()) if nnum else 0
            all_zero: bool = bool(nnum and (num_df.sum().sum() == 0))
            all_nan: bool = bool(nnum and num_df.isna().all().all())
            validation.update({
                f'{name}_rows': nrows,
                f'{name}_cols': ncols,
                f'{name}_numeric_cols': nnum,
                f'{name}_non_nan_values': non_nan,
            })
            if is_placeholder:
                warnings.append(f'{name} table missing or placeholder')
            if (nrows <= 1) or (ncols <= 1):
                warnings.append(f'{name} table very small (rows<=1 or cols<=1)')
            if nnum == 0:
                warnings.append(f'{name} has no numeric columns')
            if all_nan:
                warnings.append(f'{name} numeric data all NA')
            if all_zero:
                warnings.append(f'{name} numeric data sums to 0')
        # Combined checks
        if (validation.get('intensity_rows', 0) <= 1) and (validation.get('spc_rows', 0) <= 1):
            warnings.append('Both intensity and SPC tables are missing or tiny')
        if (validation.get('intensity_numeric_cols', 0) == 0) and (validation.get('spc_numeric_cols', 0) == 0):
            warnings.append('No numeric data available in intensity nor SPC')
    except Exception:
        # Do not interrupt main flow due to validation
        pass

    info_dict: dict = {
        'protein lengths': protein_length_dict,
        'Data type': data_type[0],
        'Data source guess': data_type[1],
        'validation': validation,
        'warnings': warnings,
    }
    table_dict: dict = {
        'spc': spc_table.to_json(orient='split'),
        'int': intensity_table.to_json(orient='split'),
    }
    return table_dict, info_dict


def guess_controls(sample_groups: Dict[str, List[str]], ctrl_indicators: List[str]) -> Tuple[List[str], List[List[str]]]:
    """Guesses control samples from sample group names based on indicator terms.
    
    Args:
        sample_groups (dict): Dictionary mapping group names to sample lists
        ctrl_indicators (list): List of strings that indicate control samples
        
    Returns:
        tuple: Contains:
            - list: Control group names
            - list: Lists of samples in each control group
            
    Notes:
        - Case-insensitive matching of control indicators
        - Returns empty lists if no controls are found
        - Each control group's samples are kept together
    """
    control_groups: list = []
    control_samples: list = []
    for group_name, samples in sample_groups['norm'].items():
        might_be_control: bool = False
        for ctrl_ind in ctrl_indicators:
            if ctrl_ind in group_name.lower():
                might_be_control = True
                break
        if might_be_control:
            control_groups.append(group_name)
            control_samples.append(samples)
    return (control_groups, control_samples)

def parse_comparisons(control_group: Optional[str], comparison_data: Optional[List[List[str]]], 
                     sgroups: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    """Parses control group and comparison data into pairwise comparisons.
    
    Args:
        control_group (str): Name of the main control group
        comparison_data (list): List of explicit [sample, control] comparisons
        sgroups (dict): Dictionary of all sample groups
        
    Returns:
        list: List of [sample, control] pairs representing comparisons
            
    Notes:
        - If control_group is specified, creates comparisons against all other groups
        - Appends any explicit comparisons from comparison_data
        - Skips invalid group names
        - Returns empty list if no valid comparisons found
    """
    comparisons: list = []
    if (control_group is not None) and (control_group != ''):
        comparisons.extend([(sample, control_group)
                            for sample in sgroups.keys()if sample != control_group])
    if comparison_data is not None:
        if len(comparison_data) > 0:
            comparisons.extend(comparison_data)
    return comparisons


def remove_duplicate_protein_groups(data_table: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate protein groups by aggregating their values.
    
    Args:
        data_table (pd.DataFrame): Input data table with protein groups as index
        
    Returns:
        pd.DataFrame: Table with unique protein groups and aggregated values
            
    Notes:
        - Numeric columns are summed
        - Non-numeric columns take first value
        - Zero values are replaced with NaN
        - Groups by index values (protein groups)
        - Preserves column order and names
    """
    # If no columns remain (e.g., non-numeric columns were dropped earlier),
    # there is nothing to aggregate. Return as-is to avoid pandas concat error.
    if data_table.shape[1] == 0:
        return data_table
    aggfuncs: dict = {}
    numerical_columns: set = set(
        data_table.select_dtypes(include=np.number).columns)
    for column in data_table.columns:
        if column in numerical_columns:
            aggfuncs[column] = 'sum'
        else:
            aggfuncs[column] = 'first'
    return data_table.groupby(data_table.index).agg(aggfuncs).replace(0, np.nan)


def parse_data_file(data_file_contents: str, data_file_name: str, 
                   data_file_modified_data: int, new_upload_style: Dict[str, str], 
                   parameters: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, str], list[str]]:
    """Parses a data file and validates its contents.

    Args:
        data_file_contents: The contents of the uploaded file
        data_file_name (str): Name of the uploaded file
        data_file_modified_data: Last modified timestamp of the file
        new_upload_style (dict): Style dictionary for UI feedback
        parameters (dict): Processing parameters including max PSM threshold

    Returns:
        tuple: Contains:
            - dict: Updated upload style with background color indicating status
            - dict: File info including metadata and data type
            - dict: Tables dictionary with intensity and spectral count data in split JSON format
            - list: List of warnings
    Notes:
        - Validates file has sufficient numeric columns (>=3)
        - Sets background-color to 'green' if valid, 'red' if invalid
        - Tables are stored in split JSON format for serialization
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


    warnings: list[str] = []
    dt_info = more_info['validation']
    if dt_info['spc_numeric_cols'] == 0 and dt_info['intensity_numeric_cols'] == 0:
        warnings.append(f'- Data table: Neither intensity nor spectral count columns were able to be identified in input.')
    if dt_info['spc_rows'] <= 1 and dt_info['intensity_rows'] <= 1:
        warnings.append(f'- Data table: Neither intensity nor spectral count data was able to be identified in input.')

    for key, table_data in tables.items():
        if isinstance(table_data, str):
            if table_data.count('No data') != 2:
                data_table: pd.DataFrame = pd.read_json(
                    io.StringIO(table_data), orient='split')
                numeric_columns: set = set(
                    data_table.select_dtypes(include=np.number).columns)
                if len(numeric_columns) >= 1:
                    has_data = True
    new_upload_style['background-color'] = 'green'
    if not has_data:
        new_upload_style['background-color'] = 'red'
    return (new_upload_style, info, tables, warnings)


def check_sample_table_column(column: str, accepted_values: List[str]) -> Optional[str]:
    """Checks if a column name matches any accepted values.

    Args:
        column (str): Column name to check
        accepted_values (list): List of valid column name variations

    Returns:
        str: Original column name if match found, None otherwise

    Notes:
        - Case-insensitive matching
        - Returns exact original column name if match found
    """
    for candidate in accepted_values:
        if candidate == column.lower():
            return column
    return None


def check_required_columns(columns: List[str]) -> Tuple[Dict[str, str], Set[str]]:
    """Validates presence of required columns in sample table.

    Args:
        columns (list): List of column names to check

    Returns:
        tuple: Contains:
            - dict: Mapping of standardized names to actual column names
            - set: Set of required column types that were found

    Notes:
        - Required columns: sample name, sample group
        - Optional columns: bait uniprot/id
        - Case-insensitive matching of column names
    """
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


def parse_sample_table(data_file_contents: str, data_file_name: str,
                      data_file_modified_data: int, 
                      new_upload_style: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, Any], str]:
    """Parses and validates a sample metadata table.

    Args:
        data_file_contents: Contents of the uploaded sample table file
        data_file_name (str): Name of the uploaded file
        data_file_modified_data: Last modified timestamp of the file
        new_upload_style (dict): Style dictionary for UI feedback

    Returns:
        tuple: Contains:
            - dict: Updated upload style with background color indicating status
            - dict: File info including metadata and column mappings
            - str: Table data in JSON split format

    Notes:
        - Validates table dimensions and required columns
        - Sets background-color to:
            - 'green': Valid table with required columns
            - 'blue': Valid table with bait information
            - 'red': Invalid table or missing required columns
        - Required columns: sample name, sample group
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
    info['required columns found'] = sorted(list(reqs_found))
    for k, v in additional_info.items():
        info[k] = v
    if len(reqs_found) < 2:
        indicator_color = 'red'
    elif 'bait uniprot' in info:
        indicator_color = 'blue'
    new_upload_style['background-color'] = indicator_color
    if indicator_color != 'red':
        for c in decoded_table.columns:
            decoded_table[c] = [text_handling.replace_accent_and_special_characters(str(x), allow_space=True, make_lowercase=False) for x in decoded_table[c]]
    return (new_upload_style, info, decoded_table.to_json(orient='split'))


def check_bait(bait_entry: Optional[str]) -> str:
    """Checks if a string contains a valid bait name.
    
    Args:
        bait_entry (str): The bait entry to validate
        
    Returns:
        str: A string representation of the bait. Returns 'No bait uniprot' if the entry is 
            empty, None, or 'nan'
            
    Examples:
        >>> check_bait('P12345')
        'P12345'
        >>> check_bait(None) 
        'No bait uniprot'
        >>> check_bait('nan')
        'No bait uniprot'
    """
    bval: str = ''
    if bait_entry is not None:
        bval = str(bait_entry)
    if (len(bval) == 0) or (bval == 'nan'):
        bval = 'No bait uniprot'
    return bval


def format_data(session_uid: str, data_tables: Dict[str, str], 
                data_info: Dict[str, Any], expdes_table: Dict[str, str],
                expdes_info: Dict[str, Any], contaminants_to_remove: List[str],
                replace_replicate_names: bool, use_unique_only: bool,
                control_indicators: List[str], 
                bait_id_column_names: List[str]) -> Dict[str, Any]:
    """Formats experimental data into a standardized dictionary structure for analysis.
    
    Args:
        session_uid (str): Unique identifier for the analysis session
        data_tables (dict): Dictionary containing intensity and spectral count tables in JSON format
        data_info (dict): Metadata about the data tables including file info and data type
        expdes_table (dict): Experimental design table in JSON format
        expdes_info (dict): Metadata about the experimental design table
        contaminants_to_remove (list): List of contaminant proteins to filter out
        replace_replicate_names (bool): Whether to replace sample names with standardized replicate names
        use_unique_only (bool): Whether to use only unique peptides/proteins
        control_indicators (list): List of terms that indicate control samples
        bait_id_column_names (list): List of possible column names for bait identifiers
        
    Returns:
        dict: A structured dictionary containing:
            - sample_groups: Sample grouping information
            - data_tables: Processed data tables (intensity, spectral counts, etc.)
            - info: Processing metadata and experiment type
            - file_info: Source file information
            - other: Additional data including protein lengths and bait information
            
    Notes:
        - Intensity values are log2 transformed if present
        - Zero values are replaced with NaN
        - Tables are stored in JSON split format
        - Experiment type is determined based on presence of bait information
        - Control samples are guessed based on provided indicators
    """
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
            'data type': data_info['Data type'],
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
            'experimental design all info': expdes_info,
            'data table all info': data_info,
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


def remove_from_table(table_name: str, table: pd.DataFrame, 
                     discard_samples: List[str]) -> pd.DataFrame:
    """Removes specified samples from a data table based on table type.

    Args:
        table_name (str): Name of the table being processed
        table (pd.DataFrame): Data table to remove samples from
        discard_samples (list): List of sample names to remove

    Returns:
        pd.DataFrame: Table with specified samples removed

    Notes:
        - For experimental design tables, removes rows where Sample name matches discard list
        - For other tables, removes columns matching discard list
    """
    if table_name == 'experimental design':
        table_without_discarded_samples = table[
            ~table['Sample name'].isin(discard_samples)
        ]
    else:
        table_without_discarded_samples = table[
            [c for c in table.columns if c not in discard_samples]
        ]
    return table_without_discarded_samples


def delete_samples(discard_samples: List[str], 
                  data_dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Removes specified samples from all tables in the data dictionary.

    Args:
        discard_samples (list): List of sample names to remove
        data_dictionary (dict): Dictionary containing all experimental data tables and metadata

    Returns:
        dict: Updated data dictionary with samples removed and sample groups adjusted

    Notes:
        - Processes all tables including intensity, spectral counts, and experimental design
        - Updates sample group mappings to reflect removed samples
        - Adds list of discarded samples to dictionary
        - Handles both regular and contaminant-containing tables
        - Removes empty sample groups after sample deletion
    """
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

def clean_sample_names(expdesign: pd.DataFrame, 
                      bait_id_column_names: List[str]) -> pd.DataFrame:
    """Clean and validate the experimental design dataframe.
    
    Args:
        expdesign (pd.DataFrame): Input experimental design dataframe containing at minimum
            'Sample group' and 'Sample name' columns
        bait_id_column_names (list): List of possible column names that could contain
            bait identifiers (e.g., ['bait id', 'bait uniprot'])
            
    Returns:
        pd.DataFrame: Cleaned experimental design dataframe with:
            - Rows containing missing required values removed
            - All values converted to strings
            - Sample names cleaned of file paths and special characters
            - Standardized bait column name if present
            
    Notes:
        - Required columns are 'Sample group' and 'Sample name'
        - Rows with NA values in required columns are dropped
        - File paths in sample names are removed (handles both Windows and Unix paths)
        - Special characters in sample names are replaced with underscores
        - If a bait identifier column exists, it is renamed to 'Bait uniprot'
        - All modifications are done on a copy of the input dataframe
        
    Example:
        >>> expd = pd.DataFrame({
        ...     'Sample name': ['path/to/sample1.raw', 'sample2'],
        ...     'Sample group': ['group1', 'group2'],
        ...     'bait id': ['P12345', 'P67890']
        ... })
        >>> cleaned = clean_sample_names(expd, ['bait id', 'bait uniprot'])
        >>> cleaned['Sample name']
        0    sample1
        1    sample2
        Name: Sample name, dtype: object
    """
    # Remove rows with missing required values
    expd_columns = ['Sample name','Sample group']
    init_rename = {}
    for c in expd_columns:
        for col in expdesign.columns:
            if c.lower().strip().replace(' ','') == col.lower().strip().replace(' ',''):
                init_rename[col] = c
    expdesign = expdesign.rename(columns=init_rename)

    expdesign = expdesign[~(expdesign[expd_columns].isna().sum(axis=1)>0)].copy()
    expd_columns.extend([c for c in expdesign.columns if c not in expd_columns])
    # Convert all values to strings
    for col in expd_columns:
        expdesign[col] = expdesign[col].apply(_to_str)
    # Remove file paths from sample names (handles both Windows and Unix paths)
    expdesign.loc[:, 'Sample name'] = expdesign['Sample name'].apply(
        lambda x: text_handling.replace_special_characters(
            clean_column_name(x),
            replacewith='_',make_lowercase=False
        )
    )
    expdesign.loc[:, 'Sample group'] = expdesign['Sample group'].apply(
        lambda x: text_handling.replace_special_characters(
            clean_column_name(x),
            replacewith='_',make_lowercase=False
        )
    )
    # Standardize bait column name if it exists
    for bid in bait_id_column_names:
        matching_cols = [c for c in expdesign.columns if c.lower().strip() == bid]
        if matching_cols:
            expdesign.rename(columns={matching_cols[0]: 'Bait uniprot'}, inplace=True)
            break
    return expdesign

def clean_column_name(col_name: str) -> str:
    """Removes file paths and extensions from column names.

    Args:
        col_name (str): Original column name potentially containing path and extensions

    Returns:
        str: Cleaned column name with paths and extensions removed

    Notes:
        - Handles both Windows and Unix style paths
        - Removes _SPC suffix
        - Removes .d extension
        - Processes path components from right to left
    """
    col = col_name.rsplit('\\', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1].rsplit('_SPC', maxsplit=1)[0].rsplit('.d', maxsplit=1)[0]
    return col

def format_sample_group_name(sample_group: Union[str, int, float]) -> Optional[str]:
    """Format sample group names, handling numeric cases.
    
    Args:
        sample_group: The sample group identifier to format. Can be numeric or string.
        
    Returns:
        str: Formatted sample group name. Returns None if input is NaN.
            For numeric inputs, returns "SampleGroup_<number>".
            For string inputs, returns the string value.
            
    Examples:
        >>> format_sample_group_name(1)
        'SampleGroup_1'
        >>> format_sample_group_name("Control")
        'Control'
        >>> format_sample_group_name(np.nan)
        None
    """
    if pd.isna(sample_group):
        return None
    
    try_num = check_numeric(sample_group)
    if try_num['success']:
        return f'SampleGroup_{try_num["value"]}'
    return str(sample_group)

def generate_replicate_name(group_name: str, sample_name: str, 
                          existing_names: Set[str], replace_names: bool) -> str:
    """Generate unique replicate names for samples within groups.
    
    Args:
        group_name (str): Name of the sample group
        sample_name (str): Original name of the sample
        existing_names (set): Set of already assigned replicate names
        replace_names (bool): If True, generates names like "Group_Rep_1". 
            If False, preserves original sample names with numeric suffixes if needed.
            
    Returns:
        str: A unique replicate name that doesn't exist in existing_names.
        
    Notes:
        When replace_names is True:
            - Names follow pattern: "{group_name}_Rep_{i}"
            - i increments until a unique name is found
            
        When replace_names is False:
            - Uses cleaned original sample name as base
            - Adds "_i" suffix only if needed for uniqueness
            - i starts at 0 and increments until unique
            
    Examples:
        >>> generate_replicate_name("Control", "sample1", {"Control_Rep_1"}, True)
        'Control_Rep_2'
        >>> generate_replicate_name("Control", "sample1", {"sample1"}, False)
        'sample1_0'
    """
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
    
def rename_columns_and_update_expdesign(expdesign: pd.DataFrame,
                                      tables: List[pd.DataFrame],
                                      bait_id_column_names: List[str],
                                      replace_names: bool = True) -> Tuple[Dict[str, Dict[str, List[str]]], 
                                                                        List[str], 
                                                                        List[Dict[str, str]], 
                                                                        pd.DataFrame]:
    """Rename columns and update experimental design table to standardize sample names.
    
    Args:
        expdesign (pd.DataFrame): Experimental design DataFrame containing at minimum
            'Sample group' and 'Sample name' columns
        tables (list): List of pandas DataFrames containing the data tables to process
        bait_id_column_names (list): List of possible column names that could contain
            bait identifiers
        replace_names (bool, optional): Whether to replace sample names with standardized
            replicate names. Defaults to True.
            
    Returns:
        tuple: Contains:
            - dict: Sample groups mapping with 'norm' (group->samples) and 'rev' (sample->group)
            - list: Columns that were discarded during processing
            - list: Used columns
            - pd.DataFrame: Updated experimental design table with only used samples
            
    Notes:
        The function performs several steps:
        1. Cleans sample names in experimental design table
        2. Maps original column names to cleaned names and group assignments
        3. Generates unique replicate names for samples
        4. Updates table column names and builds sample group mappings
        5. Removes unused samples from experimental design table
        
    The process handles:
        - Multiple input tables
        - Special characters in sample names
        - Numeric sample group names
        - Duplicate sample names
        - Bait information for interactomics experiments
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
            clean_col = col
            # Attempt cleaning up the column name if not found as is.
            if clean_col not in expdesign['Sample name'].values:
                clean_col = clean_column_name(col)
            if clean_col not in expdesign['Sample name'].values:
                clean_col = text_handling.replace_special_characters(clean_col,replacewith='_',make_lowercase=False)
            # Skip if column not in experimental design
            if clean_col not in expdesign['Sample name'].values:
                discarded_columns.append(clean_col)
                discarded_columns.append(col)
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
        tables[table_idx].rename(columns=final_names, inplace=True)
    
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

def check_comparison_file(file_contents: str, file_name: str,
                         sgroups: Dict[str, List[str]],
                         new_upload_style: Dict[str, str]) -> Tuple[Dict[str, str], List[List[str]]]:
    """Validates and parses a comparison file containing sample-control pairs.
    
    Args:
        file_contents: Base64 encoded contents of the uploaded comparison file
        file_name (str): Name of the uploaded file
        sgroups (dict): Dictionary of valid sample groups
        new_upload_style (dict): Style dictionary for UI feedback
        
    Returns:
        tuple: Contains:
            - dict: Updated upload style with background color indicating status:
                - 'green': All comparisons valid
                - 'yellow': Some comparisons invalid
                - 'red': No valid comparisons
                - 'grey': File content error
            - list: Valid comparison pairs as [sample, control]
            
    Notes:
        - Expects a table with 'sample' and 'control' columns (or first two columns)
        - Handles numeric group names by converting to 'SampleGroup_X' format
        - Skips comparisons where sample or control group not found in sgroups
        - Attempts to standardize group names to match sgroups format
    """
    indicator: str = 'green'
    try:
        comparisons: list = []
        dataframe: pd.DataFrame = read_df_from_content(
            file_contents, file_name, lowercase_columns=True)
        scol: str = 'sample'
        ccol: str = 'control'
        if ('sample' not in dataframe.columns) or ('control' not in dataframe.columns):
            scol, ccol = dataframe.columns[:2]
        for col in [scol,ccol]:
            dataframe[col] = [text_handling.replace_accent_and_special_characters(str(x), allow_space=True, make_lowercase=False) for x in dataframe[col]]
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


def validate_basic_inputs(*args: Any, fail_on_None: bool = True) -> bool:
    """Validates basic inputs for Proteogyver analysis.
    
    Args:
        *args: Variable number of input arguments to validate. The last two arguments
            should be style dictionaries containing 'background-color' keys.
        fail_on_None (bool, optional): Whether to fail validation if any input is None.
            Defaults to True.
            
    Returns:
        bool: True if validation fails (inputs invalid), False if validation passes
            
    Notes:
        - Checks for None values in all arguments if fail_on_None is True
        - Checks background-color of last two arguments (style dictionaries)
        - Validation fails if:
            - Any input is None (when fail_on_None=True)
            - Either style dictionary has background-color='red'
    """
    not_valid: bool = False
    if fail_on_None:
        for arg in args:
            if arg is None:
                not_valid = True
    for style_arg in args[-2:]:
        if style_arg['background-color'] == 'red':
            not_valid = True
    return not_valid
