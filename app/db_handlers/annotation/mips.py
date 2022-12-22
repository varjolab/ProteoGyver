import sys
import os
import shutil
import urllib.request
import gzip
import xmltodict
import pandas as pd

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

def download_and_parse(outfile) -> None:
    """Will download MIPS interaction database from https://mips.helmholtz-muenchen.de/proj/ppi/ \
        and parse it into a file
    """
    xmlfile: str = outfile.replace('gz','xml')
    if not os.path.isfile(xmlfile):
        url: str = "http://mips.helmholtz-muenchen.de/proj/ppi/data/mppi.gz"
        if not os.path.isfile(outfile):
            urllib.request.urlretrieve(url, outfile)

        with gzip.open(outfile, 'rb') as f_in:
            with open(xmlfile, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    for _ in range(0,5):
        interactions = parse(xmlfile)
        df = pd.DataFrame(data=interactions,columns=heads)
        if df.shape[0]>0:
            break
    df.to_csv(outfile.replace('.gz','_interactions.tsv'),sep='\t',index=False)

def update() -> None:
    today: str = apitools.get_timestamp()
    outdir: str = apitools.get_save_location('MIPS')
    if len([f for f in os.listdir(outdir) if 'gz' in f])<1:
    #if len(apitools.get_files_newer_than(outdir, today, 30, namefilter='mppi.gz'))<1:
        outfile: str = os.path.join(outdir, f'{today}_mppi.gz')
        download_and_parse(outfile)
    
def get_interactions(filter = None) -> dict:
    df: pd.DataFrame = pd.read_csv(apitools.get_newest_file(namefilter = '_interactions.tsv'),sep='\t')
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

def get_version_info() -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('MIPS'), namefilter='gz')
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text() -> str:
    short,long,pmid = apitools.get_pub_ref('MIPS')
    return '\n'.join([
        f'Interactions were mapped with MIPS (https://mips.helmholtz-muenchen.de/proj/ppi/) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])

