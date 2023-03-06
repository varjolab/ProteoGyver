import os
import pandas as pd
import requests
import json
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

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

def save_tab_delimed_file(reactome_url: str, output_filename, output_fileheaders: list = None):
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
    pd.DataFrame(data=filelines, columns=output_fileheaders).to_csv(output_filename + '.tsv',
                                                                    sep='\t', index=False)

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
            'https://reactome.org/download/current/interactors/reactome.all_species.interactions.tab-delimited.txt',
            None
        )
    }

# TODO: date the files, retrieve only, when necessary, e.g. every month? or two months?


def retrieve_reactome(reactome_folder: str = 'Reactome_Data', reactome_dict: dict = None) -> None:
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
        save_tab_delimed_file(r_url, out_file, headers)

def current_reactome_version() -> str:
    for _ in range(0, 20):
        r: requests.Response = requests.get('https://reactome.org/ContentService/data/database/version')
        if r.status_code == 200: 
            return r.text
          
def newer_reactome_available() -> bool:
    data_directory:str = apitools.get_save_location('Reactome')
    current_database_file:str = apitools.get_newest_file(data_directory)
    try:
        current_version: int = int(current_database_file.split('_',maxsplit=1)[0])
    except ValueError:
        return True
    newest_db_version: int = int(current_reactome_version())
    return (newest_db_version > current_version)

  
def update() -> None:
    reactome_files: dict = get_default_reactome_dict()
    latest: str = current_reactome_version()
    data_directory:str = apitools.get_save_location('Reactome')
    to_be_downloaded: dict = {}
    for reactome_filename, file_details in reactome_files.items():
        have: str = apitools.get_newest_file(data_directory,namefilter = reactome_filename).split('_')[0]
        if have == '':
            to_be_downloaded[reactome_filename] = file_details
        elif int(have) < int(latest):
            to_be_downloaded[reactome_filename] = file_details
    retrieve_reactome(reactome_folder = data_directory, reactome_dict = to_be_downloaded)

def get_version_info() -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('Reactome'))
    version:str = nfile.split('_')[0]
    downdate:str = nfile.split('_')[1]
    return f'Version {version}, downloaded ({downdate}).'

def methods_text() -> str:
    short,long,pmid = apitools.get_pub_ref('reactome')
    return '\n'.join([
        f'Reactome pathways were mapped from Reactome (https://reactome.org) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])

