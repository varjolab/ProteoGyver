from http.client import UnknownTransferEncoding
import qnorm
import pandas as pd
import io
import numpy as np
import base64
from typing import Tuple
from utilitykit import dftools

def read_df_from_content(content, filename) -> pd.DataFrame:
    _, content_string = content.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    f_end: str = filename.rsplit('.', maxsplit=1)[-1]
    data = None
    if f_end == 'csv':
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), index_col=False)
    elif f_end in ['tsv', 'tab', 'txt']:
        data: pd.DataFrame = pd.read_csv(io.StringIO(
            decoded_content.decode('utf-8')), sep='\t', index_col=False)
    elif f_end in ['xlsx', 'xls']:
        data: pd.DataFrame = pd.read_excel(io.StringIO(decoded_content))
    return data


def median_normalize(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Median-normalizes a dataframe by dividing each column by its median.

    Args:
        df (pandas.DataFrame): The dataframe to median-normalize.
        Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The median-normalized dataframe.
    """
    # Calculating the medians prior to looping is about 2-3 times more efficient,
    # than calculating the median of each column inside of the loop.
    medians: pd.Series = data_frame.median(axis=0)
    mean_of_medians: float = medians.mean()
    for col in data_frame.columns:
        data_frame[col] = (data_frame[col] / medians[col]) * mean_of_medians
    return data_frame


def quantile_normalize(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalizes a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to quantile-normalize.
        Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The quantile-normalized dataframe.
    """
    return qnorm.quantile_normalize(dataframe)


def read_dia_nn(data_table: pd.DataFrame) -> pd.DataFrame:
    table: pd.DataFrame = pd.pivot_table(
        data=data_table, index='Protein.Group', columns='Run', values='PG.MaxLFQ')
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    return [table, pd.DataFrame({'No data': ['No data']})]


def count_per_sample(data_table: pd.DataFrame, rev_sample_groups: dict) -> pd.Series:
    index: list = list(rev_sample_groups.keys())
    retser: pd.Series = pd.Series(
        index=index,
        data=[data_table[i].notna().sum() for i in index]
    ).to_json()
    return retser

def normalize(data_table: pd.DataFrame, normalization_method: str) -> pd.DataFrame:
    if normalization_method == 'Median':
        data_table: pd.DataFrame = median_normalize(data_table)
    elif normalization_method == 'Quantile':
        data_table = quantile_normalize(data_table)
    return data_table

    
def filter_missing(data_table: pd.DataFrame, sample_groups: dict, threshold: int = 60) -> pd.DataFrame:
    threshold: int = threshold/100
    keeps: list = []
    for _, row in data_table.iterrows():
        keep: bool = False
        for _, sample_columns in sample_groups.items():
            keep = keep | (row[sample_columns].notna().sum()
                           >= (threshold*len(sample_columns)))
            if keep:
                break
        keeps.append(keep)
    return data_table[keeps]


def impute(data_table:pd.DataFrame, method:str='QRILC', tempdir:str='.') -> pd.DataFrame:
    ret: pd.DataFrame = data_table
    if method == 'minProb':
        ret = dftools.impute_minprob_df(data_table)
    elif method == 'minValue':
        ret = dftools.impute_minval(data_table)
    elif method == 'QRILC':
        ret = dftools.impute_qrilc(data_table, tempdir = tempdir)
    return ret


def read_fragpipe(data_table: pd.DataFrame) -> pd.DataFrame:
    intensity_cols: list = []
    spc_cols: list = []
    for column in data_table.columns:
        if 'Total' in column:
            continue
        if 'Unique' in column:
            continue
        if 'Combined' in column:
            continue
        if 'Intensity' in column:
            intensity_cols.append(column)
        elif 'Spectral Count' in column:
            spc_cols.append(column)
    protein_col: str = 'Protein ID'
    table: pd.DataFrame = data_table
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    table.index = table[protein_col]
    intensity_table: pd.DataFrame = table[intensity_cols]
    spc_table: pd.DataFrame = table[spc_cols].rename(
        columns={ic: ic.replace('Spectral Count', '').strip()
                 for ic in spc_cols}
    )
    if intensity_table[intensity_cols[0:2]].sum().sum() == 0:
        intensity_table = pd.DataFrame({'No data': ['No data']})
    else:
        intensity_table.rename(
        columns={ic: ic.replace('Intensity', '').strip()
                 for ic in intensity_cols},
        inplace=True
        )        
    return (intensity_table, spc_table)

# TODO: for generic format, ask user whether it's SPC or intensity


def read_matrix(data_table: pd.DataFrame, is_spc_table=False) -> pd.DataFrame:
    protein_id_column: str = 'Protein.Group'
    table: pd.DataFrame = data_table
    table.index = table[protein_id_column]
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    table.drop(columns=[protein_id_column],inplace=True)
    spc_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
    intensity_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
    if is_spc_table:
        spc_table = table
    else:
        intensity_table = table
    return (intensity_table, spc_table)


def rename_columns_and_update_expdesign(expdesign, tables) -> Tuple[dict, dict]:
    expdesign['Sample name'] = [
        oldvalue.rsplit('\\', maxsplit=1)[-1]\
            .rsplit('/', maxsplit=1)[-1]\
            .rstrip('.d') for oldvalue in expdesign['Sample name'].values
    ]
    sample_groups: dict = {}
    for table in tables:
        if len(table.columns) < 2:
            continue

        rename_columns: dict = {}
        for column_name in table.columns:
            # Discard samples that are not named
            col: str = column_name
            if col not in expdesign['Sample name'].values:
                # Try to see if the column without possible file path is in the expdesign:
                col = col.rsplit(
                    '\\', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1]
                if col not in expdesign['Sample name'].values:
                    # Discard column if not found
                    continue
            sample_group: str = expdesign[expdesign['Sample name']
                                          == col].iloc[0]['Sample group']
            # If no value is available for sample in the expdesign
            # (but sample column name is there for some reason), discard column
            if pd.isna(sample_group):
                continue
            newname: str = str(sample_group)
            # We expect replicates to not be specifically named; they will be named here.
            if newname[0].isdigit():
                newname = f'Sample_{newname}'
            if newname not in sample_groups:
                newname_to_use = f'{newname}_Rep_1'
                sample_groups[newname] = [newname_to_use]
                rename_columns[newname_to_use] = col
            else:
                i: int = 2
                while f'{newname}_Rep_{i}' in rename_columns:
                    i += 1
                newname_to_use: str = f'{newname}_Rep_{i}'
                sample_groups[newname].append(newname_to_use)
                rename_columns[newname_to_use] = col
        # Reverse the rename dict to be able to use it for renaming the dataframe columns
        rename_columns = {v: k for k, v in rename_columns.items()}
        # Discard unneeded columns
        table.drop(
            columns= [c for c in table.columns if c not in rename_columns],
            inplace=True
            )
        table.rename(columns=rename_columns, inplace=True)

    rev_sample_groups: dict = {}
    for group, samples in sample_groups.items():
        for sample in samples:
            rev_sample_groups[sample] = group
    return (sample_groups, rev_sample_groups)


def parse_data(data_content, data_name, expdes_content, expdes_name) -> list:
    table: pd.DataFrame = read_df_from_content(data_content, data_name)
    expdesign: pd.DataFrame = read_df_from_content(expdes_content, expdes_name)
    read_funcs: dict[tuple[str, str]] = {
        ('DIA', 'DIA-NN'): read_dia_nn,
        ('DDA', 'FragPipe'): read_fragpipe,
        ('DDA', 'Unknown'): read_matrix
    }

    data_type: tuple = None
    if 'Fragment.Quant.Raw' in table.columns:
        if 'Decoy.CScore' in table.columns:
            data_type = ('DIA', 'DIA-NN')
    elif 'Top Peptide Probability' in table.columns:
        if 'Protein Existence' in table.columns:
            data_type = ('DDA', 'FragPipe')
    if data_type is None:
        data_type = ('DDA', 'Unknown')

    intensity_table: pd.DataFrame
    spc_table: pd.DataFrame
    intensity_table, spc_table = read_funcs[data_type](table)
    sample_groups: dict
    rev_sample_groups: dict
    sample_groups, rev_sample_groups = rename_columns_and_update_expdesign(
        expdesign,
        [intensity_table, spc_table]
    )
    if len(intensity_table.columns) > 1:
        untransformed_intensity_table: pd.DataFrame = intensity_table
        intensity_table = intensity_table.apply(np.log2)
    return [
        intensity_table,
        sample_groups,
        rev_sample_groups,
        spc_table,
        data_type,
        untransformed_intensity_table
    ]


def get_count_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.\
        notna().sum().\
        to_frame(name='Protein count')
    data.index.name = 'Sample name'
    return data


def get_sum_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.sum().\
        to_frame(name='Value sum')
    data.index.name = 'Sample name'
    return data


def get_avg_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.mean().\
        to_frame(name='Value mean')
    data.index.name = 'Sample name'
    return data


def get_na_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = ((data_table.
                           isna().sum() / data_table.shape[0]) * 100).\
        to_frame(name='Missing value %')
    data.index.name = 'Sample name'
    return data
