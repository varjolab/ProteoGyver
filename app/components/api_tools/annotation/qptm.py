import sys
import os
import zipfile
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

def parse() -> None:
    today: str = apitools.get_timestamp()
    outdir: str = apitools.get_save_location('qPTM')
    # No need to check updates, the data is only downloadable via manual request from qPTM.
    if len([f for f in os.listdir(outdir) if 'tsv' in f])<1:
    # no need to check dates - PTMint does not have updates.
        datafile: str = os.path.join(outdir, f'{today}_qPTM.tsv')
        with zipfile.ZipFile('qPTM_all_data.zip', 'r') as zip_ref:
            zip_ref.extractall(outdir)
        pd.read_csv('qPTM_all_data.txt',sep='\t').to_csv(datafile,sep='\t',encoding = 'utf-8')

def get_version_info() -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('qPTM'), namefilter='.tsv')
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text() -> str:
    short,long,pmid = apitools.get_pub_ref('qPTM')
    return '\n'.join([
        f'PTM sites were mapped from qPTM (http://qptm.omicsbio.info/) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])

