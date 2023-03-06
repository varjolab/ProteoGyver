import sys
import os
import shutil
import urllib.request
import gzip
from datetime import datetime
import pandas as pd
import ftplib

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

def parse(xmlfile) -> list:
    with open(xmlfile,encoding='utf-8') as fil:
       doc: dict = xmltodict.parse(fil.read())
    interactions: list = []
    heads: list = ['Source database','Reference DB','Reference ID','interaction detection name',
        'interaction detection reference DB','interaction detection reference ID',]
    for i in [1,2]:
        heads.extend([
            f'Protein {i} confidence unit',
            f'Protein {i} confidence value',
            f'Protein {i} is overexpressed',
            f'Protein {i} is tagged',
            f'Protein {i} fullName',
            f'Protein {i} shortName',
            f'Protein {i} organism',
            f'Protein {i} xref DB',
            f'Protein {i} xref ID',
        ])
    for entry in doc['entrySet']['entry']['interactionList']['interaction']:
            interactions.append([
                'MIPS',
                entry['experimentList']['experimentDescription']['bibref']['xref']['primaryRef']['@db']
            ])
            try:
                interactions[-1].append(entry['experimentList']['experimentDescription']['bibref']
                                            ['xref']['primaryRef']['@id'])
            except KeyError:
                interactions[-1].append('')
            interactions[-1].extend([
                entry['experimentList']['experimentDescription']['interactionDetection']['names']
                                        ['shortLabel'],
                entry['experimentList']['experimentDescription']['interactionDetection']['xref']
                                        ['primaryRef']['@db'],
                entry['experimentList']['experimentDescription']['interactionDetection']['xref']
                                        ['primaryRef']['@id']
            ])
            for pdic in entry['participantList']['proteinParticipant']:
                interactions[-1].extend([
                    pdic['confidence']['@unit'],
                    pdic['confidence']['@value'],
                    pdic['isOverexpressedProtein'],
                    pdic['isTaggedProtein'],
                    pdic['proteinInteractor']['names']['fullName'],
                    pdic['proteinInteractor']['names']['shortLabel'],
                    pdic['proteinInteractor']['organism']['@ncbiTaxId'],
                    pdic['proteinInteractor']['xref']['primaryRef']['@db']
                ])
                try:
                    interactions[-1].append(pdic['proteinInteractor']['xref']['primaryRef']['@id'])
                except KeyError:
                    interactions[-1].append(None)
    return interactions
    
def get_interactions() -> dict:
    df: pd.DataFrame = pd.read_csv(apitools.get_newest_file('IntAct', namefilter = '_interactions.tsv'),sep='\t')
    interactions: dict = {}
    for _,row in df[(df['Protein 1 xref ID']!='-') & (df['Protein 2 xref ID']!='-')].iterrows():
        int1, int2 = row[['Protein 1 xref ID', 'Protein 2 xref ID']]
        for i in [int1,int2]:
            if i not in interactions:
                interactions[i] = {}
        for i1, i2 in [[int1, int2], [int2, int1]]:
            if i1 not in interactions[i2]:
                interactions[i2][i1] = {'references': set()}
            interactions[i2][i1]['references'].add((f'{row["Reference DB"]}:{row["Reference ID"]}', row['interaction detection name']))
    return interactions



## TODO: split the file into organism-specific files, and implement a retrieval method for getting one or more organism-specific interactomes
def do_update(save_file) -> None:
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/intact/latest/psimitab'
    ftpfilename: str = 'intact.txt'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    with open(save_file,'wb') as fil:
        ftp.retrbinary(f'RETR {ftpfilename}',fil.write)
    ftp.quit()
    split_file(save_file)

#Very inefficient, but it's done very rarely (depending on IntAct version schedule), so does not matter much for now.
def split_file(file_path:str) -> None:
    df: pd.DataFrame = pd.read_csv(file_path,sep='\t')
    filepath: list = file_path.split(os.sep)
    filename:str = filepath[-1]
    filepath = os.path.join(*filepath[:-1])
    organism_indexes: dict = {'single': {},'multi': {}}
    org_names = {}

    for index, row in df.iterrows():
        for organism_column in ['Taxid interactor A', 'Taxid interactor B']:
            if row[organism_column] == '-':
                continue
            orgs: list = row[organism_column].split('|')
            orgs = list(set([o.split(':')[1] for o in orgs]))
            oids: set = set()
            for o in orgs:
                oid, oname = o.split('(', maxsplit=1)
                oids.add(oid)
                oname = oname.strip(')')
                if oid not in org_names:
                    org_names[oid] = set()
                org_names[oid].add(oname)
            if len(oids)>1:
                target: str = 'multi'
            else:
                target = 'single'
            for o in oids:
                if o not in organism_indexes[target]:
                    organism_indexes[target][o] = set()
                organism_indexes[target][o].add(index)

    for org_type, org_dict in organism_indexes.items():
        for organism, organism_indexes in org_dict.items():
            org_df = df.loc[list(organism_indexes)]
            new_filename = os.path.join(
                filepath,
                filename.replace('.txt','') + 
                f'-{org_type}-{organism}.tsv'
            )
            org_df.to_csv(new_filename,sep='\t')
    with open(os.path.join(filepath, filename.replace('.txt','_organism-names.tsv')),'w', encoding='utf-8') as fil:
        fil.write('Organism ID\tOrganism names\n')
        for key, values in org_names.items():
            fil.write(f'{key}\t{";".join(sorted(list(values)))}')

def get_latest_ftp_dirname() -> str:
    ftpurl: str = 'ftp.ebi.ac.uk'
    ftpdir: str = '/pub/databases/current/'
    ftp: ftplib.FTP = ftplib.FTP(ftpurl)
    ftp.login()
    ftp.cwd(ftpdir)
    filelist: list = ftp.nlst()
    ftp.quit()
    latest: datetime = datetime.strptime('0001-01-01','%Y-%m-%d').date()
    for file in filelist:
        filedate: datetime = datetime.strptime(file, '%Y-%m-%d').date()
        if filedate > latest: 
            latest = filedate
    return filedate

def update() -> None:
    latest: str = get_latest_ftp_dirname()
    current_version: str = apitools.get_newest_file(apitools.get_save_location('IntAct')).split('_')[0]
    current_version = apitools.parse_timestamp_from_str(current_version)
    if latest > current_version:
        do_update(os.path.join(apitools.get_save_location('IntAct'),f'{apitools.get_timestamp()}_intact.txt'))
        
def get_version_info() -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('IntAct'))
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text() -> str:
    short,long,pmid = apitools.get_pub_ref('IntAct')
    return '\n'.join([
        f'Interactions were mapped with IntAct (https://www.ebi.ac.uk/intact) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])