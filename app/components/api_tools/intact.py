"""
IntAct database interaction module for protein-protein interaction data.

This module provides functionality to download, parse, and manage protein interaction data
from the IntAct database (https://www.ebi.ac.uk/intact/). It handles:
- Automated updates from IntAct's FTP server
- Parsing of PSI-MITAB formatted files
- Conversion to pandas DataFrames with standardized column names
- Version tracking and data freshness checks
- Methods text generation for citations

The main entry points are:
- update(): Check for and download new IntAct releases
- get_latest(): Retrieve the most recent downloaded data as a DataFrame
- methods_text(): Generate citation text for the data source
"""

import sys
import os
import zipfile
from datetime import datetime
import pandas as pd
import ftplib
import time
from typing import Optional
from . import apitools

def get_ids(df, col1, col2, uniprots_to_get: set|None) -> list[str]:
    nvals = []
    for row in df.itertuples(index=False):
        ids_a = [getattr(row, col1)] + getattr(row, col2).split('|')
        ids_a = [i.split(':')[1] for i in ids_a if 'uniprotkb' in i]
        if uniprots_to_get:
            ids_a = [i for i in ids_a if i.split('-')[0] in uniprots_to_get]
        nvals.append(';'.join(ids_a).strip(';') if ids_a else '|DROP|')
    return nvals

def get_iso(ser: pd.Series) -> list[str]:
    return [';'.join(s.split('-')[0] for s in sv.split(';')).strip(';') for sv in ser.values]

def split_and_save_by_prefix(df: pd.DataFrame, column: str, num_chars: int, output_dir: str, index: bool = False, sep: str = '\t') -> None:
    os.makedirs(output_dir, exist_ok=True)
    for (prefix, result) in df.groupby(df[column].str[:num_chars]):
        filename = os.path.join(output_dir, f"{prefix}.tsv")
        write_header = not os.path.exists(filename)
        result.to_csv(filename, mode='a', header=write_header, index=index, sep=sep)

def get_org(ser: pd.Series) -> list:
    return [
        ','.join(
            sorted(
                list(
                    set(
                        [
                            r.split(':')[-1].split('(')[0] for r in val.split('|')
                        ]
                    )
                )
            )
        ) for val in ser
    ]
    
def filter_chunk(chunk: pd.DataFrame, uniprots_to_get: set|None, organisms: set|None) -> pd.DataFrame:
    mask = chunk['id1'].str.contains('uniprotkb', na=False) | \
            chunk['id1a'].str.contains('uniprotkb', na=False) | \
            chunk['id2'].str.contains('uniprotkb', na=False) | \
            chunk['id2a'].str.contains('uniprotkb', na=False)
    chunk = chunk.loc[mask].copy()
    chunk = chunk.loc[chunk['Negative']==False]
    if chunk.shape[0] > 0:
        chunk['idsa'] = get_ids(chunk, 'id1', 'id1a', uniprots_to_get)
        chunk['idsb'] = get_ids(chunk, 'id2', 'id2a', uniprots_to_get)
        chunk = chunk.loc[~chunk['idsa'].str.contains('|DROP|', regex=False)]
        chunk = chunk.loc[~chunk['idsb'].str.contains('|DROP|', regex=False)]
    if chunk.shape[0] > 0:
        chunk['noisoa'] = get_iso(chunk['idsa'])
        chunk['noisob'] = get_iso(chunk['idsb'])
        chunk['organism_interactor_a'] = get_org(chunk['Taxid interactor A'])
        chunk['organism_interactor_b'] = get_org(chunk['Taxid interactor B'])
        if organisms:
            organisms = {str(x) for x in organisms}
            chunk = chunk[
                (chunk['organism_interactor_a'].isin(organisms)) |
                (chunk['organism_interactor_b'].isin(organisms))
            ]
        chunk.sort_values(by='idsa',inplace=True)
    return chunk

def handle_and_split_save(df: pd.DataFrame, temp_dir: str, sep: str = '\t') -> None:
    normcols = [
        'Interaction type(s)',
        'Interaction detection method(s)',
        'Publication Identifier(s)',
        'Confidence value(s)',
        'intact_creation_date',
        'intact_update_date'
    ]
    swcols = [
        'Biological role(s) interactor A',
        'Annotation(s) interactor A',
        'organism_interactor_a',
        'Experimental role(s) interactor A',
        'Biological role(s) interactor B',
        'Annotation(s) interactor B',
        'organism_interactor_b',
        'Experimental role(s) interactor B'
    ]

    renames = {c: c.lower().replace('(s)','').replace(' ','_') for c in normcols+swcols}
    normcols = [renames[c] for c in normcols]
    swcols = [renames[c] for c in swcols]
    df = df.rename(columns=renames)
    df['idsa_split'] = df['idsa'].str.split(';')
    df['idsb_split'] = df['idsb'].str.split(';')
    df['noisoa_split'] = df['noisoa'].str.split(';')
    df['noisob_split'] = df['noisob'].str.split(';')
    
    # Pre-process normcols data using vectorized operations
    for c in normcols:
        df[f'{c}_processed'] = (
            df[c].str.split('|')
            .apply(lambda x: [v for val in x for v in val.split('__')])  # flatten
            .apply(lambda vals: [
                v.strip().lower().replace('-', '').replace(';', '').replace('_', '')
                for v in vals
                if v.strip().lower() not in ('', '0', 'nan', 'none')
            ])
            .str.join(';')
        )

    dfcols = [
        'uniprot_id_a',
        'uniprot_id_b',
        'uniprot_id_a_noiso',
        'uniprot_id_b_noiso',
        'isoform_a',
        'isoform_b',
    ]  + normcols + swcols + ['source_database', 'notes', 'update_time', 'interaction']
    datarows = {}
    # Use list comprehension instead of multiple append operations
    for row in df.itertuples(index=False):
        new_rows = [
            [
                id1, # uniprot_id_a
                id2,  # uniprot_id_b
                noiso1[i],  # uniprot_id_a_noiso
                noiso2[j],  # uniprot_id_b_noiso
                id1,  # isoform_a
                id2,  # isoform_b
            ] + 
            [getattr(row, f'{c}_processed') for c in normcols] +
            [getattr(row, f'{c[:-1]}{n1}') for c in swcols[:4]] +
            [getattr(row, f'{c[:-1]}{n2}') for c in swcols[:4]] +
            ['IntAct', '', f'IntAct:{str(datetime.today()).split()[0]}', f'{id1.split("-")[0]}_-_{id2.split("-")[0]}']
            for n1, n2 in [('a', 'b'), ('b', 'a')]
            for i, id1 in enumerate(getattr(row, f'ids{n1}_split'))
            for j, id2 in enumerate(getattr(row, f'ids{n2}_split'))
            for noiso1, noiso2 in [(getattr(row, f'noiso{n1}_split'), getattr(row, f'noiso{n2}_split'))]
        ]
        for n in new_rows:
            index = n[-1]
            keys = dfcols[:-1]
            datarows.setdefault(index, {v: set() for v in keys})
            for i, k in enumerate(keys):
                datarows[index][k]|= set(n[i].split(';'))
    # Create DataFrame in one go
    findf = pd.DataFrame.from_dict({ind: {key: ';'.join(sorted(list(val))).strip(';') for key, val in ind_dict.items()} for ind, ind_dict in datarows.items()},orient='index')
    findf['publication_identifier'] = findf['publication_identifier'].str.lower()
    findf.index.name = 'interaction'
    # Split save by prefix (3 first letters of interaction ID) to make deduplication later possible, and to conserver RAM.
    split_and_save_by_prefix(findf.reset_index(), 'interaction', 3, temp_dir, index=False, sep=sep)

def only_latest_date(ser: pd.Series, time_format: str = '%Y/%m/%d') -> pd.Series:
    vals = []
    for v in ser.values:
        if not ';' in v:
            vals.append(v)
        else:
            dates = pd.to_datetime(v.strip(';').split(';'))
            vals.append(dates.max().strftime(time_format))
    return pd.Series(vals, index=ser.index)

def get_final_df(df_filename) -> pd.DataFrame:
    non_str_cols = {}
    with open(df_filename, 'r') as file:
        header = file.readline().strip().split('\t')
    dtype_dict = {col: str for col in header if col not in non_str_cols} | non_str_cols
    
    chunk_df = pd.read_csv(df_filename,sep='\t', index_col='interaction',dtype=dtype_dict)
    datarows = {}
    for intname, row in chunk_df.iterrows():
        datarows.setdefault(intname, {v: set() for v in chunk_df.columns})
        for c in chunk_df.columns:
            datarows[intname][c]|= set(str(row[c]).split(';'))
    findf = pd.DataFrame.from_dict({ind: {key: ';'.join(sorted(list(val))).strip(';') for key, val in ind_dict.items()} for ind, ind_dict in datarows.items()},orient='index')
    for timecol in ['intact_creation_date', 'intact_update_date']:
        findf[timecol] = only_latest_date(findf[timecol])
    findf.index.name = 'interaction'
    return findf

# super inefficient, but run rarely, so good enough.
def generate_pandas(file_path:str, output_name:str, uniprots_to_get:set|None, organisms: set|None = None) -> None:
    """
    Inefficiently generates a pandas dataframe from a given intact zip file (downloaded  by update()) and writes it to a .tsv file with the same name as input file path.

    :param file_path: path to the downloaded zip file
    :param output_name: path for the output file
    :param uniprots_to_get: set of which uniprots should be included in the written .tsv file. If None, all uniprots will be included.
    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    """
    dropcols = [
        'Checksum(s) interactor A',
        'Checksum(s) interactor B',
        'Expansion method(s)',
        'Feature(s) interactor A',
        'Feature(s) interactor B',
        'Host organism(s)',
        'Identification method participant A',
        'Identification method participant B',
        'Interaction Checksum(s)',
        'Interaction Xref(s)',
        'Interaction annotation(s)',
        'Interaction identifier(s)',
        'Interaction parameter(s)',
        'Negative',
        'Publication 1st author(s)',
        'Stoichiometry(s) interactor A',
        'Stoichiometry(s) interactor B',
        'Taxid interactor A',
        'Taxid interactor B',
        'Type(s) interactor A',
        'Type(s) interactor B',
        'Xref(s) interactor A',
        'Xref(s) interactor B',
        'Alias(es) interactor A',
        'Alias(es) interactor B',
        'Source database(s)'
    ]
    renames = {
        '#ID(s) interactor A': 'id1','Alt. ID(s) interactor A': 'id1a',
        'ID(s) interactor B': 'id2','Alt. ID(s) interactor B': 'id2a',
        'Creation date': 'intact_creation_date',
        'Update date': 'intact_update_date'
    }
    folder_path: str = file_path.replace('.zip','')
   # output_name = folder_path+'.tsv'
    unzipped_path: str = os.path.join(folder_path,'intact.txt')
    if not os.path.exists(unzipped_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(file_path.replace('.zip',''))
    
    #temp_dir = os.path.join(folder_path,'parser_tmp')
    # Handle in chunks to conserver RAM
    for i, chunk in enumerate(pd.read_csv(unzipped_path,sep='\t', chunksize=10000)):
        print(f'Processing IntAct chunk: {i}')
        chunk.rename(columns=renames,inplace=True)
        chunk = filter_chunk(chunk, uniprots_to_get, organisms)
        chunk.drop(columns=dropcols, inplace=True)
        chunk.drop_duplicates(inplace=True)
        if chunk.shape[0] > 0:
            handle_and_split_save(chunk, folder_path)
    # Final deduplication of the 3-letter prefix split files
    for fname in os.listdir(folder_path):
        if fname.endswith('.tsv'):
            findf: pd.DataFrame = get_final_df(os.path.join(folder_path, fname))
            findf.to_csv(os.path.join(folder_path, fname), index=True, sep='\t')

    os.remove(file_path)
    os.remove(unzipped_path)
    os.remove(unzipped_path.replace('.txt','_negative.txt'))

def download_intact_ftp(save_file: str, max_retries: int = 10, retry_delay: int = 30) -> Optional[str]:
    print('Downloading IntAct')
    ftpurl = 'ftp.ebi.ac.uk'
    ftpdir = '/pub/databases/intact/current/psimitab'
    ftpfilename = 'intact.zip'

    for attempt in range(1, max_retries + 1):
        try:

            ftp = ftplib.FTP(ftpurl, timeout=30)
            ftp.login()
            ftp.cwd(ftpdir)

            with open(save_file, 'wb') as fil:
                ftp.retrbinary(f'RETR {ftpfilename}', fil.write)

            ftp.quit()
            print(f'IntAct FTP download completed successfully: {save_file}')
            return save_file

        except (ftplib.error_temp, ftplib.error_perm, ConnectionResetError, TimeoutError, OSError) as e:
            if attempt < max_retries:
                print(f'IntAct FTP download failed: {e}\n Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)
            else:
                print(f'IntAct FTP download failed after all retries: {e}')
                return None


def do_update(save_file, uniprots_to_get: set|None, organisms: set|None) -> None:
    """
    Handles practicalities of updating the intact tsv file on disk

    :param save_dir: path to the .tsv file where data should be saved
    :param uniprots_to_get: a set of which uniprots should be retained. If None, all uniprots will be retained.
    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    """
    if not os.path.exists(save_file):
        download_intact_ftp(save_file)
    generate_pandas(save_file, save_file.replace('.zip','.tsv'),uniprots_to_get, organisms)

def update(uniprots_to_get: set|None = None, organisms: set|None = None) -> None:
    """
    Identifies whether update should be done or not, and does an update if needed.
    
    :param uniprots_to_get: which uniprots should be retained in the output file. If None, all uniprots will be retained.
    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    """
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/intact/current/'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    latest = datetime.strptime(ftp.pwd().rsplit('/',maxsplit=1)[1], '%Y-%m-%d').date()
    ftp.quit()
    current_version: str = apitools.get_newest_file(apitools.get_save_location('IntAct'))   
    should_update: bool = False
    if os.path.exists(os.path.join(apitools.get_save_location('IntAct'), current_version)):
        should_update = latest > apitools.parse_timestamp_from_str(current_version.split('_')[0])
    else:
        should_update = True
    if should_update:
        print('Updating IntAct')
        do_update(os.path.join(apitools.get_save_location('IntAct'),f'{apitools.get_timestamp()}_intact.zip'), uniprots_to_get, organisms)

def read_file_chunks(filepath: str, organisms: set|None = None, subset_letter: str|None = None, since_date: datetime|None = None) -> pd.DataFrame:
    iter_csv = pd.read_csv(filepath, iterator=True, sep='\t', chunksize=1000)
    df_parts = []
    for chunk in iter_csv:
        if subset_letter:
            chunk = chunk.loc[chunk['interaction'].str.startswith(subset_letter)]
        if organisms:
            chunk = chunk.loc[chunk['organism_interactor_a'].isin(organisms) | chunk['organism_interactor_b'].isin(organisms)]
        if since_date:
            chunk['intact_update_date'] = pd.to_datetime(chunk['intact_update_date'])
            chunk['intact_creation_date'] = pd.to_datetime(chunk['intact_creation_date'])
            chunk = chunk.loc[(chunk['intact_update_date'] >= since_date) | (chunk['intact_creation_date'] >= since_date)]
        df_parts.append(chunk)
    return pd.concat(df_parts)

def read_folder_chunks(folderpath: str, organisms: set|None = None, subset_letter: str|None = None, since_date: datetime|None = None) -> pd.DataFrame:
    df_parts = []
    for file in os.listdir(folderpath):
        if subset_letter and not file.startswith(subset_letter):
            continue
        df_parts.append(read_file_chunks(os.path.join(folderpath, file), organisms, None, since_date))
    if len(df_parts) > 0:
        return pd.concat(df_parts)
    else:
        return pd.DataFrame()

def get_latest(organisms: set|None = None, subset_letter: str|None = None, name_only: bool = False, since_date: datetime|None = None) -> pd.DataFrame|str:
    """
    Fetches the latest data from disk

    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    :returns: Pandas dataframe of the latest IntACT data.
    """
    current_version: str = apitools.get_newest_file(apitools.get_save_location('IntAct'))
    filepath: str = os.path.join(apitools.get_save_location('IntAct'), current_version)
    if name_only:
        return filepath
    if not os.path.exists(filepath):
        return pd.DataFrame()
    else:
        return read_folder_chunks(filepath, organisms, subset_letter=subset_letter, since_date=since_date)
    
def get_available() -> list[str]:
    filepath: str = get_latest(name_only=True) # type: ignore
    return [f.split('.')[0] for f in os.listdir(filepath) if f.endswith('.tsv')]

def get_version_info() -> str:
    """
    Parses version info from the newest downloaded IntACT file
    """
    nfile: str = apitools.get_newest_file(apitools.get_save_location('IntAct'))
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text() -> str:
    """
    Generates a methods text for used intact data

    :returns: a tuple of (readable reference information (str), PMID (str), intact description (str))
    """
    short,long,pmid = apitools.get_pub_ref('IntAct')
    return '\n'.join([
        'IntAct',
        f'Interactions were mapped with IntAct (https://www.ebi.ac.uk/intact) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])