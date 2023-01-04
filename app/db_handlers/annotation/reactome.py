import os
import pandas as pd
import requests
import json
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

# TODO: date the files, retrieve only, when necessary, e.g. every month? or two months?
def newer_reactome_available(current_database_file) -> bool:
    current_version: int = int(current_database_file.split('_',maxsplit=1)[0])
    newest_db_version: int = int(current_reactome_version())
    return (newest_db_version > current_version)


def retrieve_reactome_data(reactome_ids: list) -> tuple:
    """Retrieves more detailed data for given reactome identifiers from reactome

    Returns:
    tuple of (pd.DataFrame, list).
    pandas dataframe will contain the most often used information, and the list contains \
        full dict objects containing all the available data.
    """
    reactome_url = 'https://reactome.org/ContentService/data/query/ids'
    headers = {'accept': 'application/json', 'content-type': 'text/plain'}
    data = ','.join(reactome_ids)
    request = requests.post(url=reactome_url, data=data, headers=headers, timeout=20)
    json_dict = json.loads(request.text)
    columns = ['stIdVersion', 'displayName', 'speciesName', 'literatureReference',
               'goBiologicalProcess', 'summation', 'className', 'schemaClass']
    datarows = []
    index = []
    for iddata in json_dict:
        index.append(iddata['stId'])
        newrow = []
        for column_name in columns:
            identifier_string = None
            if column_name not in iddata:
                newrow.append(identifier_string)
                continue
            if column_name == 'literatureReference':
                pubmed_identifiers = [str(x['pubMedIdentifier'])
                                      for x in iddata[column_name]]
                identifier_string = ';'.join(pubmed_identifiers)
            elif column_name == 'goBiologicalProcess':
                identifier_string = (
                    f'GO:{iddata[column_name]["accession"]}'
                    f'({iddata[column_name]["displayName"]})'
                )
            elif column_name == 'summation':
                identifier_string = iddata[column_name][0]['text']
            else:
                identifier_string = iddata[column_name]
            newrow.append(identifier_string)
        datarows.append(newrow)
    return pd.DataFrame(data=datarows, columns=columns, index=pd.Series(data=index, name='stId'))

def get_tab_delimed_file(reactome_url: str, output_filename, output_fileheaders: list = None):
    """Retrieves a single tab delimed (e.g. reactome) file and saves it to file according to \
        specified output filename and output file headers. Output will be tab delimed.
    """
    response = requests.get(reactome_url, timeout=10)
    filelines = response.text.split('\n')
    filelines[0] = filelines[0].strip('#').strip()
    filelines = [fl.split('\t') for fl in filelines]
    if not output_fileheaders:
        output_fileheaders = filelines[0]
        filelines = filelines[1:]
    pd.DataFrame(data=filelines, columns=output_fileheaders).to_csv(output_filename,
                                                                    sep='\t', index=False)

# TODO: move to an external file


def get_default_reactome_dict():
    """Returns default dict of reactome urls and their column names.
    """
    return {
        'uniprot2lowestlevel':
        (
            'https://reactome.org/download/current/UniProt2Reactome.txt',
            ['UniprotID', 'ReactomeID', 'URL', 'Name', 'Evidence code', 'Species']
        ),
        'uniprot2allLvl':
        (
            'https://reactome.org/download/current/UniProt2Reactome_All_Levels.txt',
            ['UniprotID', 'ReactomeID', 'URL', 'Name', 'Evidence code', 'Species']
        ),
        'uniprot2allReactions':
        (
            'https://reactome.org/download/current/UniProt2ReactomeReactions.txt',
            ['UniprotID', 'ReactomeID', 'URL', 'Name', 'Evidence code', 'Species']
        ),
        'PathwayList':
        (
            'https://reactome.org/download/current/ReactomePathways.txt',
            ['ReactomeID', 'Name', 'Species']
        ),
        'Complexes2Pathways':
        (
            'https://reactome.org/download/current/Complex_2_Pathway_human.txt',
            None
        ),
        'PathwayHierarchy':
        (
            'https://reactome.org/download/current/ReactomePathwaysRelation.txt',
            ['Parent ReactomeID', 'Child ReactomeID']
        ),
        'allInteractionsFromPathways':
        (
            'https://reactome.org/download/current/interactors/\
                    reactome.all_species.interactions.tab-delimited.txt',
            None
        )
    }

# TODO: date the files, retrieve only, when necessary, e.g. every month? or two months?


def retrieve_full_reactome(reactome_folder: str = 'Reactome_Data', reactome_dict: dict = None) -> None:
    """Retrieves full mapping and pathway information files from Reactome to a specified folder

    Parameters:
    reactome_folder: folder, where data should be saved. If it doesn't exist, it will be
        created.
    reactome_dict: A dictionary of {output_fileName: reactome_url} for Reactome files.
        See method get_default_reactome_dict for reference.
    """
    if not reactome_dict:
        reactome_dict: dict = get_default_reactome_dict()
    current_version: str = current_reactome_version()
    current_date: str = apitools.get_timestamp()
    if not os.path.isdir(reactome_folder):
        os.makedirs(reactome_folder)
    for out_file, (r_url, headers) in reactome_dict.items():
        out_file = f'{current_version}_{current_date}_{out_file}'
        out_file:str = os.path.join(reactome_folder, out_file)
        get_tab_delimed_file(r_url, out_file, headers)

def current_reactome_version() -> str:
    for _ in range(0, 20):
        r: requests.Response = requests.get('https://reactome.org/ContentService/data/database/version')
        if r.status_code == 200: 
            return r.text
##TODO: Below code is from uniprot.py. Convert it to reactome-specific format. 
            
def update(organism = 9606,progress=False) -> None:
    outdir: str = apitools.get_save_location('Uniprot')
    if is_newer_available(apitools.get_newest_file(outdir, namefilter=str(organism))):
        today: str = apitools.get_timestamp()
        df: pd.DataFrame = download_full_uniprot_for_organism(organism=organism,progress=progress,overall_progress=progress)
        outfile: str = os.path.join(outdir, f'{today}_Uniprot_{organism}.tsv')
        df.to_csv(outfile)

def is_newer_available(newest_file: str, organism: int = 9606) -> bool:
    uniprot_url: str = f"https://rest.uniprot.org/uniprotkb/search?\
        query=organism_id:{organism}&format=fasta"
    uniprot_response: requests.Response = requests.get(uniprot_url)
    newest_version:str = uniprot_response.headers['X-UniProt-Release'].replace('_','-')
    vals: list =  newest_version.split('-')
    newest_y: int = int(vals[0])
    newest_m: int = int(vals[1])
    vals =  newest_file.split('_',maxsplit=1)[0].split('-')
    if len(vals) < 2:
        vals = [-1,-1]
    current_y: int = vals[0]
    current_m: int = vals[1]
    ret: bool = False
    if current_y < newest_y:
        ret = True
    elif current_y == newest_y:
        if current_m < newest_m:
            ret = True
    return ret

def get_version_info(organism:int=9606) -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('Uniprot'), namefilter=str(organism))
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text(organism=9606) -> str:
    short,long,pmid = apitools.get_pub_ref('uniprot')
    return '\n'.join([
        f'Protein annotations were mapped from UniProt (https://uniprot.org) {short}',
        f'{get_version_info(organism)}',
        pmid,
        long
    ])

