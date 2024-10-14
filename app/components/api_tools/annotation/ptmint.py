import sys
import os
import urllib.request
import pandas as pd

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

def download_and_parse(outfile) -> None:
    """
    Will download PTMint database from https://ptmint.sjtu.edu.cn/ and parse it into a file

    :param outfile: path to the output file.
    """
    csvfile: str = outfile
    if not os.path.isfile(csvfile):
        url: str = 'https://ptmint.sjtu.edu.cn/data/PTM%20experimental%20evidence.csv'
        if not os.path.isfile(outfile):
            urllib.request.urlretrieve(url, outfile)
    pd.read_csv(outfile).to_csv(outfile.replace('.csv','.tsv'),sep='\t',index=False,encoding = 'utf-8')

def update() -> None:
    """
    Updates the database, though only required if rebuilding. PTMint does not have update.
    """
    today: str = apitools.get_timestamp()
    outdir: str = apitools.get_save_location('PTMint')
    if len([f for f in os.listdir(outdir) if 'csv' in f])<1:
    # no need to check dates - PTMint does not have updates.
        outfile: str = os.path.join(outdir, f'{today}_PTMint.csv')
        download_and_parse(outfile)

def get_version_info() -> str:
    """
    Returns version info for the newest (and only) available PTMint version.
    """
    nfile: str = apitools.get_newest_file(apitools.get_save_location('PTMint'), namefilter='.tsv')
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text() -> str:
    """
    Generates a methods text for used PTMint data
    
    :returns: a tuple of (readable reference information (str), PMID (str), PTMint description (str))
    """
    short,long,pmid = apitools.get_pub_ref('PTMint')
    return '\n'.join([
        f'Phosphosites were mapped from PTMint (ptmint.sjtu.edu.cn) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])

