import sys
import os
import zipfile
from datetime import datetime
import pandas as pd
import ftplib

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

# super inefficient, but run rarely, so good enough.
def generate_pandas(file_path:str, output_name:str) -> None:
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall('intact_temp')
    df = pd.read_csv(os.path.join('intact_temp','intact.xt'))
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
                    dodata[-1].append('|'.join(cval))
                except TypeError:
                    vs = set(cval)
                    dodata[-1].append(list(vs)[0])
    docols = ['uniprot_id_a', 'uniprot_id_b', 'isoform_a','isoform_b'] + [c.lower().replace('(s)','').replace(' ','_') for c in cols]
    findf = pd.DataFrame(data=dodata, columns=docols)
    findf['source_database'] = 'IntAct'
    findf['notes'] = ''
    findf['update_time'] = str(datetime.today()).split()[0]
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
    findf.to_csv(output_name, sep='\t')

def do_update(save_file) -> None:
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/intact/current/psimitab'
    ftpfilename: str = 'intact.zip'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    with open(save_file,'wb',encoding = 'utf-8') as fil:
        ftp.retrbinary(f'RETR {ftpfilename}',fil.write)
    ftp.quit()
    generate_pandas(save_file, save_file.replace('.zip','.tsv'))

def update() -> None:
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/intact/current/'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    latest = datetime.strptime(ftp.pwd().rsplit('/',maxsplit=1)[1], '%Y-%m-%d').date()
    current_version: str = apitools.get_newest_file(apitools.get_save_location('IntAct')).split('_')[0]
    current_version = apitools.parse_timestamp_from_str(current_version)
    if latest > current_version:
        do_update(os.path.join(apitools.get_save_location('IntAct'),f'{apitools.get_timestamp()}_intact.zip'))
        
def get_version_info() -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('IntAct'))
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text() -> str:
    short,long,pmid = apitools.get_pub_ref('IntAct')
    return '\n'.join([
        'IntAct',
        f'Interactions were mapped with IntAct (https://www.ebi.ac.uk/intact) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])