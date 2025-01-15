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
import numpy as np
import shutil

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

# super inefficient, but run rarely, so good enough.
def generate_pandas(file_path:str, output_name:str, uniprots_to_get:set) -> None:
    """
    Inefficiently generates a pandas dataframe from a given intact zip file (downloaded  by update()) and writes it to a .tsv file with the same name as input file path.

    :param file_path: path to the downloaded zip file
    :param output_name: path for the output file
    :param uniprots_to_get: set of which uniprots should be included in the written .tsv file
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(file_path.replace('.zip',''))
    df = pd.read_csv(os.path.join(file_path.replace('.zip',''),'intact.txt'),sep='\t')
    shutil.rmtree(file_path.replace('.zip',''))
    for c in df.columns:
        df[c] = [str(v).replace(';',',') for v in df[c].values]
    ndata = {}
    cols = {
        'Biological role(s) interactor A',
        'Experimental role(s) interactor A',
        'Annotation(s) interactor A',
        'Biological role(s) interactor B',
        'Experimental role(s) interactor B',
        'Annotation(s) interactor B',
        'Interaction type(s)',
        'Interaction detection method(s)',
        'Publication Identifier(s)',
        'Confidence value(s)'
    }
    for _, row in df.iterrows():
        ids_a = [row['#ID(s) interactor A']] + row['Alt. ID(s) interactor A'].split('|')
        ids_a = [i.split(':')[1] for i in ids_a if 'uniprotkb' in i]
        ids_b = [row['ID(s) interactor B']] + row['Alt. ID(s) interactor B'].split('|')
        ids_b = [i.split(':')[1] for i in ids_b if 'uniprotkb' in i]
        ids_a = [i for i in ids_a if i in uniprots_to_get]
        ids_b = [i for i in ids_b if i in uniprots_to_get]
        if len(ids_a) == 0:continue
        if len(ids_b) == 0:continue
        for i in ids_a:
            if i not in ndata:
                ndata[i] = {}
            for j in ids_b:
                if j not in ndata[i]:
                    ndata[i][j] = {c: [] for c in cols}
                for c in cols:
                    ndata[i][j][c].append(row[c])
        for i in ids_b:
            if i not in ndata:
                ndata[i] = {}
            for j in ids_a:
                if j not in ndata[i]:
                    ndata[i][j] = {c: [] for c in cols}
                for c in cols:
                    flipped_c = c.replace(' A',' B')
                    if c.endswith(' B'):
                        flipped_c = c.replace(' B',' A')
                    ndata[i][j][flipped_c].append(row[c])
    dodata = []
    for id1, idi in ndata.items():
        if '-' in id1:
            iso1: str = id1
            id1: str = id1.split('-')[0]
        else:
            iso1 = ''
        for id2, cdic in idi.items():
            if '-' in id2:
                iso2: str = id2
                id2: str = id2.split('-')[0]
            else:
                iso2 = ''
            dodata.append([id1,id2,iso1,iso2])
            for c in cols:
                cval = []
                for val in cdic[c]:
                    try:
                        cval.extend(val.split('|'))
                    except AttributeError:
                        cval.append(val)
                try:
                    dodata[-1].append(';'.join(cval))
                except TypeError:
                    vs = set(cval)
                    dodata[-1].append(list(vs)[0])
    docols = ['uniprot_id_a', 'uniprot_id_b', 'isoform_a','isoform_b'] + [c.lower().replace('(s)','').replace(' ','_') for c in cols]
    findf = pd.DataFrame(data=dodata, columns=docols)
    findf['source_database'] = 'IntAct'
    findf['notes'] = ''
    findf['update_time'] = 'IntAct:'+str(datetime.today()).split()[0]
    id1c = []
    id2c = []
    iso1c = []
    iso2c = []
    for _,row in findf.iterrows():
        id1 = row['uniprot_id_a']
        id2 = row['uniprot_id_b']
        if '-' in id1:
            iso1: str = id1
            id1: str = id1.split('-')[0]
        else:
            iso1 = ''
        if '-' in id2:
            iso2: str = id2
            id2: str = id2.split('-')[0]
        else:
            iso2 = ''
        id1c.append(id1)
        id2c.append(id2)
        iso1c.append(iso1)
        iso2c.append(iso2)
    findf['uniprot_id_a_noiso'] = id1c
    findf['uniprot_id_b_noiso'] = id2c
    findf['isoform_a'] = iso1c
    findf['isoform_b'] = iso2c
    findf['interaction'] = findf['uniprot_id_a'] + '_-_' + findf['uniprot_id_b']
    findf = findf[[
        'interaction','uniprot_id_a', 'uniprot_id_b', 'uniprot_id_a_noiso', 'uniprot_id_b_noiso',
        'isoform_a', 'isoform_b', 'publication_identifier', 
        'interaction_detection_method', 'interaction_type', 'confidence_value',
        'source_database', 'experimental_role_interactor_a','experimental_role_interactor_b',
        'biological_role_interactor_a', 'biological_role_interactor_b','annotation_interactor_a',
        'annotation_interactor_b', 'notes', 'update_time'
    ]]
    for c in findf.columns:
        for repchar in ['|','__']:
            tmp = [v for v in findf[c].values if repchar in str(v)]
            if len(tmp)>0:
                findf[c] = [str(v).replace(repchar,';') for v in findf[c].values]
    for c in findf.columns:
        findf[c] = [str(v).replace('nan','').replace('None','').replace('-|-','|') for v in findf[c].values]
    for c in findf.columns:
        nvals = findf[c].values
        for nullval in  ['-',';','_','|','',' ','0']:
            nvals = [str(v).strip(nullval).strip() for v in nvals]
        if sum([len(v)==0 for v in nvals]) > 0:
            findf[c] = nvals
    findf = findf.replace('',np.nan)
    findf['publication_identifier'] = findf['publication_identifier'].str.lower()
    findf.to_csv(output_name, sep='\t', index=False)

def do_update(save_file, uniprots_to_get: set) -> None:
    """
    Handles practicalities of updating the intact tsv file on disk

    :param save_dir: path to the .tsv file where data should be saved
    :param uniprots_to_get: a set of which uniprots should be retained.
    """
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/intact/current/psimitab'
    ftpfilename: str = 'intact.zip'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    with open(save_file,'wb') as fil:
        ftp.retrbinary(f'RETR {ftpfilename}',fil.write)
    ftp.quit()
    generate_pandas(save_file, save_file.replace('.zip','.tsv'),uniprots_to_get)

def update(uniprots_to_get: set) -> None:
    """
    Identifies whether update should be done or not, and does an update if needed.
    
    :param uniprots_to_get: which uniprots should be retained in the output file.
    """
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/intact/current/'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    latest = datetime.strptime(ftp.pwd().rsplit('/',maxsplit=1)[1], '%Y-%m-%d').date()
    current_version: str = apitools.get_newest_file(apitools.get_save_location('IntAct')).split('_')[0]
    should_update: bool = False
    try:
        should_update = latest > apitools.parse_timestamp_from_str(current_version)
    except ValueError:
        should_update = True
    if should_update:
        do_update(os.path.join(apitools.get_save_location('IntAct'),f'{apitools.get_timestamp()}_intact.zip'), uniprots_to_get)

def get_latest() -> pd.DataFrame:
    """
    Fetches the latest data from disk

    :returns: Pandas dataframe of the latest IntACT data.
    """
    current_version: str = apitools.get_newest_file(apitools.get_save_location('IntAct'),namefilter='.tsv')
    return pd.read_csv(
        os.path.join(apitools.get_save_location('IntAct'), current_version),
        index_col = 'interaction',
        sep = '\t',
        low_memory=False
    )

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