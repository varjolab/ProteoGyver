import os
import qnorm
import pandas as pd
import io
import time
import numpy as np
import base64
from typing import Tuple
import platform
import uuid
import subprocess
import textwrap

def read_df_from_content(content, filename) -> pd.DataFrame:
    """Reads a dataframe from uploaded content.
    
    Filenames ending with ".csv" are read as comma separated, filenames ending with ".tsv", ".tab" or
    ".txt" are read as tab-delimed files, and ".xlsx" and ".xls" are read as excel files.
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
    """Reads dia-nn report file into an intensity matrix"""
    table: pd.DataFrame = pd.pivot_table(
        data=data_table, index='Protein.Group', columns='Run', values='PG.MaxLFQ')
    # Replace zeroes with missing values
    table.replace(0, np.nan, inplace=True)
    return [table, pd.DataFrame({'No data': ['No data']}), None]


def count_per_sample(data_table: pd.DataFrame, rev_sample_groups: dict) -> pd.Series:
    """Counts non-zero values per sample (sample names from rev_sample_groups.keys()) and returns a series with sample names in index and counts as values."""
    index: list = list(rev_sample_groups.keys())
    retser: pd.Series = pd.Series(
        index=index,
        data=[data_table[i].notna().sum() for i in index]
    )
    return retser

def normalize(data_table: pd.DataFrame, normalization_method: str) -> pd.DataFrame:
    """Normalizes a given dataframe with the wanted method."""
    if normalization_method == 'Median':
        data_table: pd.DataFrame = median_normalize(data_table)
    elif normalization_method == 'Quantile':
        data_table = quantile_normalize(data_table)
    return data_table

    
def filter_missing(data_table: pd.DataFrame, sample_groups: dict, threshold: int = 60) -> pd.DataFrame:
    """Discards rows with more than threshold percent of missing values in all sample groups"""
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
    """Imputes missing values into the dataframe with the specified method"""
    ret: pd.DataFrame = data_table
    if method == 'minProb':
        ret = impute_minprob_df(data_table)
    elif method == 'minValue':
        ret = impute_minval(data_table)
    elif method == 'QRILC':
        ret = impute_qrilc(data_table, tempdir = tempdir)
    elif method == 'gaussian':
        ret = impute_gaussian(data_table)
    return ret

def impute_qrilc(dataframe: pd.DataFrame, rpath:str='C:\\Program Files\\R\\newest-r\\bin', tempdir:str=None) -> pd.DataFrame:
    """Impute missing values in dataframe using QRILC method

    Calls an R function to qrilc-impute missing values into the input dataframe.
    Input dataframe should only have numerical data with missing values.

    Parameters:
    df: pandas dataframe with the missing values. Should not have any text columns
    rpath: path to Rscript.exe
    """
    if not tempdir: 
        tempdir: str = '.'
    tempname: uuid.UUID = uuid.uuid4()
    temp_r_file: str = os.path.join(tempdir,f'fromimpute_{tempname}.R')
    tempdffile: str = os.path.join(tempdir,f'fromimpute_{tempname}_df.tsv')
    tempdffile_dest: str = os.path.join(tempdir,f'fromimpute_{tempname}_dest_df.tsv')
    dataframe.to_csv(tempdffile, sep='\t')

    with open(temp_r_file, 'w', encoding='utf-8') as fil:
        fil.write(textwrap.dedent(f"""
                    library("imputeLCMD")
                    df <- read.csv("{tempdffile}",sep="\\t",row.names=1)
                    df2 = impute.QRILC(df, tune.sigma = 1)
                    imputed = df2[1]
                    df3 = data.frame(imputed)
                    write.table(imputed,file="{tempdffile_dest}",sep="\\t")
                    """))

    process: subprocess.CompletedProcess = run_r_script(temp_r_file, rpath=rpath)
    df2: pd.DataFrame = pd.DataFrame()
    try:
        df2 = pd.read_csv(tempdffile_dest, index_col=0, sep='\t')
    except FileNotFoundError:
        errormsg: str = '\n'.join([
            'R script FAILED for QRILC imputation.',
            'Some likely causes include: ',
            '- Columns or rows with nothing but missing values in input',
            '- Rscript.exe not in given path',
            '-----------------------------------------------------------',
            'Detailed error information:',
            'Subprocess exit code:',
            f'{process.returncode}',
            '-----',
            'Subprocess stderr:',
            f'{process.stderr.decode()}',
            '-----',
            'Subprocess stdout:',
            f'{process.stdout.decode()}'
        ])

        raise RuntimeError(errormsg) from None
    finally:
        for tempfile in [temp_r_file, tempdffile, tempdffile_dest]:
            try:
                os.remove(tempfile)
            except PermissionError:
                time.sleep(2)
                os.remove(tempfile)
            except FileNotFoundError:
                continue

    df2.index.name = dataframe.index.name

    column_first_letter: list = list({x[0] for x in df2.columns})
    if len(column_first_letter) == 1:
        column_first_letter = column_first_letter[0]
        if column_first_letter in ('X', 'Y'):
            rename_dict: dict = {c: c[1:] for c in df2.columns}
            df2.rename(columns=rename_dict, inplace=True)
    df2.columns = dataframe.columns
    return df2


def get_newest_r(rpath) -> str:
    """Returns rpath, where the string "newest-r" has been replaced by the newest R version found\
        in the preceding directory"""
    sort_list: list = []
    rfolds: dict = {}
    rbase: str
    rend: str
    rbase, rend = rpath.split(os.sep+'newest-r'+os.sep)
    for filename in os.listdir(rbase):
        if not os.path.isdir(os.path.join(rbase, filename)):
            continue
        folder_tuple: tuple = tuple(int(i) for i in filename.split('-')[1].split('.'))
        sort_list.append(folder_tuple)
        rfolds[folder_tuple] = filename
    sort_list = sorted(sort_list, reverse=True)
    return os.path.join(rbase, rfolds[sort_list[0]], rend)

def run_r_script(scriptfilepath: str, rpath: str = 'C:\\Program Files\\R\\newest-r\\bin',
                 output_file: str = None) -> subprocess.CompletedProcess:
    """Runs an R script

    Parameters:
    script_file: path to script file.
    rpath: Path to bin directory of R, where Rscript.exe is located. If you use 'newest-r' instead\
        of the actual R directory (e.g. 'R-4.2.1'), program will default to newest R version\
            identified.
    output_file: filename where to save output, if desired
    """
    if platform.system() == 'Linux': ## Linux
        rpath: str = 'Rscript'
    elif 'newest-r' in rpath: ## Windows probably
        rpath: str = get_newest_r(rpath)
        rpath = os.path.join(rpath, 'Rscript.exe')
    cmd: list = [rpath, scriptfilepath]
    process: subprocess.CompletedProcess = subprocess.run(cmd, capture_output=True, check=False)
    if output_file:
        output: list = process.args
        output.extend([
            '-------',
            'STDOUT:',
            process.stdout.decode("utf-8"),
            '-------',
            'STDERR:',
            process.stderr.decode("utf-8"),
        ])
        with open(output_file, 'w', encoding='utf-8') as fil:
            fil.write('\n'.join(output))
    return process

def impute_minval(dataframe: pd.DataFrame, impute_zero:bool=False) -> pd.DataFrame:
    """Impute missing values in dataframe using minval method

    Input dataframe should only have numerical data with missing values.
    Missing values will be replaced by the minimum value of each column.

    Parameters:
    df: pandas dataframe with the missing values. Should not have any text columns
    impute_zero: True, if zero should be considered a missing value
    """
    newdf: pd.DataFrame = pd.DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        newcol: pd.Series = dataframe[column]
        if impute_zero:
            newcol = newcol.replace(0,np.nan)
        newcol = newcol.fillna(newcol.min())
        newdf.loc[:, column] = newcol
    return newdf

def impute_gaussian(data_table: pd.DataFrame, dist_width: float = 0.3, dist_down_shift: float = 1.8) -> pd.DataFrame:
    """Impute missing values in dataframe using values from random numbers from normal distribution.

    Based on the method used by Perseus (http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian)

    Parameters:
    data_table: pandas dataframe with the missing values. Should not have any text columns
    dist_width: Gaussian distribution relative to stdev of each column. 
        Value of 0.5 means the width of the distribution is half the standard deviation of the sample column values.
    dist_down_shift: How far downwards the distribution is shifted. By default, 1.8 standard deviations down.
    """
    newdf: pd.DataFrame = pd.DataFrame(index=data_table.index)
    for column in data_table.columns:
        newcol: pd.Series = data_table[column]
        stdev: float = newcol.std()
        distribution: np.ndarray = np.random.normal(
                loc = newcol.mean - (dist_down_shift*stdev),
                scale = dist_width*stdev,
                size = column.shape[0]*100
            )
        replace_values: pd.Series = pd.Series(
            index = data_table.index,
            data = np.random.choice(a=distribution,size=column.shape[0],replace=False)
        )
        newcol = newcol.fillna(replace_values)
        newdf.loc[:, column] = newcol
    return newdf

def impute_minprob(series_to_impute: pd.Series, scale: float = 1.0,
                   tune_sigma: float = 0.01, impute_zero=True) -> pd.Series:
    """Imputes missing values with randomly selected entries from a distribution \
        centered around the lowest non-NA values of the series.

    Arguments:
    series_to_impute: pandas series with possible missing values

    Keyword arguments:
    scale: passed to numpy.random.normal
    tune_sigma: fraction of values from the lowest end of the series to use for \
        generating the distribution
    impute_zero: treat 0 values as missing values and impute new values for them
    """

    ser: pd.Series = series_to_impute.sort_values(ascending=True)
    ser = ser[ser > 0].dropna()
    ser = ser[:int(len(ser)*tune_sigma)]

    # implement q value
    distribution: np.ndarray = np.random.normal(
        loc=ser.median(), scale=scale, size=len(series_to_impute*100))

    output_series: pd.Series = series_to_impute.copy()
    for index, value in output_series.items():
        impute_value: bool = False
        if pd.isna(value):
            impute_value = True
        elif (value == 0) and impute_zero:
            impute_value = True
        if impute_value:
            output_series[index] = np.random.choice(distribution)
    return output_series

def impute_minprob_df(dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """imputes whole dataframe with minprob imputation. Dataframe should only have numerical columns

    Parameters:
    df: dataframe to impute
    kwargs: keyword args to pass on to impute_minprob
    """
    newdf: pd.DataFrame = pd.DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        newdf.loc[:, column] = impute_minprob(dataframe[column], **kwargs)
    return newdf

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
    spc_table: pd.DataFrame = table[spc_cols].rename(
        columns={ic: ic.replace('Spectral Count', '').strip()
                 for ic in spc_cols}
    )
    if intensity_table[intensity_cols[0:2]].sum().sum() == 0:
        print('no intensities')
        intensity_table = pd.DataFrame({'No data': ['No data']})
    else:
        intensity_table.rename
        columns={ic: ic.replace('Intensity', '').strip()
                 for ic in intensity_cols},
        inplace=True
        )        
    return (intensity_table, spc_table, protein_lengths)

def read_matrix(data_table: pd.DataFrame, is_spc_table:bool=False, max_spc_ever:int=0) -> pd.DataFrame:
    """Reads a generic matrix into a data table. Either the returned SPC or intensity table is an empty dataframe.
    
    Matrix is assumed to be SPC matrix, if the maximum value is smaller than max_spc_ever.
    """
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
        if table.select_dtypes(include=[np.number]).max().max() <= max_spc_ever:
            spc_table = table
        else:
            intensity_table = table
    return (intensity_table, spc_table, None)

def run_saint(
    data_table: pd.DataFrame,
    rev_sample_groups: dict,
    protein_lengths: dict,
    saint_command: str,
    control_table: pd.DataFrame = None,
    control_groups:set = None,
    output_directory:str = None) -> Tuple[pd.DataFrame,set]:
    """This function will run SAINTexpress analysis on the given data table and control sets."""

    if control_groups is None:
        control_groups: set = set()
    baitfile: list = []
    for column in data_table:
        groupname: str = rev_sample_groups[column]
        if groupname in control_groups:
            baitfile.append(f'{column}\t{groupname}\tC\n')
        else:
            baitfile.append(f'{column}\t{groupname}\tT\n')
    if control_table is not None:
        for col in control_table.columns:
            baitfile.append(f'{col}\tChosen_control\tC\n')
            rev_sample_groups[col] = 'Chosen_control'
    
    preyfile: list = []
    discarded: set = set()
    all_proteins: set = set(data_table.index.values)
    if control_table is not None:
        all_proteins |= set(control_table.index.values)
    for prey_protein in all_proteins:
        if prey_protein in protein_lengths:
            preyfile.append(f'{prey_protein}\t{protein_lengths[prey_protein]}\t{prey_protein}\n')
        else:
            discarded.add(prey_protein)
    
    intfile: pd.DataFrame
    intfile = data_table.reset_index().melt(id_vars='index').dropna()
    if control_table is not None:
        intfile = pd.concat([intfile, control_table.reset_index().melt(id_vars='index').dropna()])
    intfile.rename(columns={'variable': 'Bait', 'index': 'Prey', 'value': 'SPC'},inplace=True)
    intfile = intfile[~intfile['Prey'].isin(discarded)]
    intfile.loc[:,'BaitGroup'] = intfile.apply(lambda x: rev_sample_groups[x['Bait']],axis=1)

    saint_dir:str
    saint_cmd:str
    saint_dir,saint_cmd = saint_command
    temp_dir: str = os.path.join(saint_dir, f'{uuid.uuid4()}')
    if not os.path.isdir(temp_dir): 
        os.makedirs(temp_dir)

    saint_cmd = os.path.join('..',saint_cmd)

    intfile_name: str = os.path.join(temp_dir,'int.txt')
    baitfile_name: str = os.path.join(temp_dir,'bait.txt')
    preyfile_name: str = os.path.join(temp_dir, 'prey.txt')
    saint_output_file:str = os.path.join(temp_dir,'list.txt')

    intfile[['Bait','BaitGroup','Prey','SPC']].to_csv(intfile_name,index=False,header=False,sep='\t')
    with open(baitfile_name,'w', encoding='utf-8') as fil:
        fil.write(''.join(baitfile))
    with open(preyfile_name,'w', encoding='utf-8') as fil:
        fil.write(''.join(preyfile))
    cmd: list = [saint_cmd,'int.txt','prey.txt','bait.txt']

    process: subprocess.CompletedProcess = subprocess.run(cmd, capture_output=True, check=False,cwd=temp_dir)

    output_dataframe: pd.DataFrame = pd.read_csv(saint_output_file,sep='\t')
    if output_directory is not None:
        output_report_file:str = os.path.join(output_directory, 'SAINT_output.txt')
        with open(output_report_file,'w',encoding='utf-8') as fil:
            fil.write(f'Subprocess exit code: {process.returncode}\n')
            fil.write(f'----------\nSubprocess output:\n{process.stdout.decode()}\n')
            fil.write(f'----------\nSubprocess errors:\n{process.stderr.decode()}\n')
        os.rename(intfile_name, os.path.join(output_directory,'interaction.txt'))
        os.rename(baitfile_name, os.path.join(output_directory,'bait.txt'))
        os.rename(preyfile_name, os.path.join(output_directory,'prey.txt'))
        os.rename(saint_output_file, os.path.join(output_directory,'saint_output_list.txt'))

    return (output_dataframe, sorted(list(discarded)))

def rename_columns_and_update_expdesign(expdesign, tables) -> Tuple[dict, dict]:
    # Get rid of file paths and timstof .d -file extension, if present:
    expdesign['Sample name'] = [
        oldvalue.rsplit('\\', maxsplit=1)[-1]\
            .rsplit('/', maxsplit=1)[-1]\
            .rstrip('.d') for oldvalue in expdesign['Sample name'].values
    ]
    discarded_columns:list = []
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
                    discarded_columns.append(col)
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
    return (sample_groups, rev_sample_groups, discarded_columns)

def remove_duplicate_protein_groups(data_table: pd.DataFrame) -> pd.DataFrame:
    aggfuncs: dict = {}
    numerical_columns: set = set(data_table.select_dtypes(include=np.number).columns)
    for column in data_table.columns:
        if column in numerical_columns:
            aggfuncs[column] = sum
        else:
            aggfuncs[column] = 'first'
    return data_table.groupby(data_table.index).agg(aggfuncs).replace(0,np.nan)

def parse_data(data_content, data_name, expdes_content, expdes_name, max_theoretical_spc: int=0) -> list:
    table: pd.DataFrame = read_df_from_content(data_content, data_name)
    expdesign: pd.DataFrame = read_df_from_content(expdes_content, expdes_name)
    read_funcs: dict[tuple[str, str]] = {
        ('DIA', 'DIA-NN'): read_dia_nn,
        ('DDA', 'FragPipe'): read_fragpipe,
        ('DDA', 'Unknown'): read_matrix
    }

    data_type: tuple = None
    keyword_args: dict = {}
    if 'Fragment.Quant.Raw' in table.columns:
        if 'Decoy.CScore' in table.columns:
            data_type = ('DIA', 'DIA-NN')
    elif 'Top Peptide Probability' in table.columns:
        if 'Protein Existence' in table.columns:
            data_type = ('DDA', 'FragPipe')
    if data_type is None:
        data_type = ('DDA', 'Unknown')
        keyword_args['max_spc_ever'] = max_theoretical_spc

    intensity_table: pd.DataFrame
    spc_table: pd.DataFrame
    protein_length_dict: dict
    intensity_table, spc_table, protein_length_dict = read_funcs[data_type](table, **keyword_args)
    intensity_table = remove_duplicate_protein_groups(intensity_table)
    spc_table = remove_duplicate_protein_groups(spc_table)

    sample_groups: dict
    rev_sample_groups: dict
    discarded_columns: list
    sample_groups, rev_sample_groups, discarded_columns = rename_columns_and_update_expdesign(
        expdesign,
        [intensity_table, spc_table]
    )
    if len(intensity_table.columns) > 1:
        untransformed_intensity_table: pd.DataFrame = intensity_table
        intensity_table = intensity_table.apply(np.log2)
    else:
        untransformed_intensity_table = intensity_table
    return [
        intensity_table,
        sample_groups,
        rev_sample_groups,
        spc_table,
        data_type,
        untransformed_intensity_table,
        protein_length_dict,
        discarded_columns
    ]
def guess_controls(sample_groups: dict) -> Tuple[list,list]:
    rl: list = []
    rl_samples: list = []
    for group_name, samples in sample_groups.items():
        if 'gfp' in group_name.lower():
            rl.append(group_name)
            rl_samples.append(samples)
        
    return (rl, rl_samples)

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
