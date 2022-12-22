import qnorm
import pandas as pd
import io
import numpy as np
import base64

def read_df_from_content(content, filename) -> pd.DataFrame:
    _, content_string = content.split(',')
    decoded_content: bytes = base64.b64decode(content_string)
    f_end:str = filename.rsplit('.',maxsplit=1)[-1]
    data = None
    if f_end=='csv':
        data:pd.DataFrame = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')))
    elif f_end in ['tsv','tab','txt']:
        data:pd.DataFrame = pd.read_csv(io.StringIO(decoded_content.decode('utf-8')),sep='\t')
    elif f_end in ['xlsx','xls']:
        data:pd.DataFrame = pd.read_excel(io.StringIO(decoded_content))
    return data

def median_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median-normalizes a dataframe by dividing each column by its median.

    Args:
        df (pandas.DataFrame): The dataframe to median-normalize. Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The median-normalized dataframe.
    """
    # Calculating the medians prior to looping is about 2-3 times more efficient, than calculating the median of each column inside of the loop.
    medians: pd.Series = df.median(axis=0)
    mean_of_medians: float = medians.mean()
    for col in df.columns:
        df[col] = (df[col] / medians[col]) * mean_of_medians
    return df

def quantile_normalize(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalizes a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to quantile-normalize. Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The quantile-normalized dataframe.
    """
    return qnorm.quantile_normalize(dataframe)

def parse_data(data_content, data_name, expdes_content, expdes_name,log2_transform=False) -> list:
    table: pd.DataFrame = read_df_from_content(data_content,data_name)
    expdesign: pd.DataFrame = read_df_from_content(expdes_content,expdes_name)

    table = table.replace(0,np.nan)
    column_map: dict = {}
    sample_groups: dict = {}
    rev_sample_groups:dict = {}
    rename_columns: dict = {oldname: oldname.rsplit('\\',maxsplit=1)[-1].rsplit('/')[-1] for oldname in table.columns}
    table = table.rename(columns=rename_columns)
    expdesign['Sample name'] = [
            oldvalue.rsplit('\\',maxsplit=1)[-1].rsplit('/')[-1] for oldvalue in expdesign['Sample name'].values
        ]
    protein_id_column: str = 'Protein.Group'
    keep_columns: set = set()
    for col in table.columns:
        if col not in expdesign['Sample name'].values:
            continue
        sample_group = expdesign[expdesign['Sample name']==col].iloc[0]['Sample group']
        # If no value is available for sample in the expdesign (but sample column name is there for some reason), discard column
        if pd.isna(sample_group):
            continue
        newname: str = str(sample_group)
        # We expect replicates to not be specifically named; they will be named here.
        i: int = 1
        if newname[0].isdigit():
            newname = f'Sample_{newname}'
        while f'{newname}_Rep_{i}' in column_map:
            i+=1
        newname_to_use: str = f'{newname}_Rep_{i}'
        if newname not in sample_groups:
            sample_groups[newname] = []
        sample_groups[newname].append(newname_to_use)
        rev_sample_groups[newname_to_use] = newname
        column_map[newname_to_use] = col
        keep_columns.add(newname_to_use)
    table: pd.DataFrame = table.rename(columns={v:k for k,v in column_map.items()})
    table.index = table[protein_id_column]
    table = table[[c for c in table.columns if c in keep_columns]]
    table = table.apply(np.log2)

    return [table, column_map, sample_groups, rev_sample_groups]
    
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
    data: pd.DataFrame = ((data_table.\
    isna().sum() / data_table.shape[0]) * 100).\
    to_frame(name='Missing value %')
    data.index.name = 'Sample name'
    return data

    