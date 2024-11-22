import os
import pandas as pd
import requests
import json
import numpy as np
from io import StringIO
from urllib.parse import quote
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


class handler():

    _defaults: list = [
        'Panther Reactome pathways',
        'Panther GOBP slim',
        'Panther GOMF slim',
        'Panther GOCC slim'
    ]
    _available: list = []
    _names: dict = {
        'Panther GO molecular function': 'GO:0003674',
        'Panther GO biological process': 'GO:0008150',
        'Panther GO cellular component': 'GO:0005575',
        'Panther GOMF slim': 'ANNOT_TYPE_ID_PANTHER_GO_SLIM_MF',
        'Panther GOBP slim': 'ANNOT_TYPE_ID_PANTHER_GO_SLIM_BP',
        'Panther GOCC slim': 'ANNOT_TYPE_ID_PANTHER_GO_SLIM_CC',
        'Panther protein class': 'ANNOT_TYPE_ID_PANTHER_PC',
        'Panther pathway': 'ANNOT_TYPE_ID_PANTHER_PATHWAY',
        'Panther Reactome pathways': 'ANNOT_TYPE_ID_REACTOME_PATHWAY'
    }

    _datasets: dict = {}
    _nice_name: str = 'PantherDB'
    _names_rev: dict

    @property
    def nice_name(self) -> str:
        return self._nice_name

    def __init__(self, get_datasets: bool = False) -> None:
        self._available = sorted(list(self._names.keys()))
        self._names_rev = {v: k for k, v in self._names.items()}
        if get_datasets:
            datasets = self.get_pantherdb_datasets()
        else:
            datasets = {
                "GO:0003674": [
                    "molecular_function",
                    "Gene Ontology Molecular Function annotations including both manually curated and electronic annotations."
                ],
                "GO:0008150": [
                    "biological_process",
                    "Gene Ontology Biological Process annotations including both manually curated and electronic annotations."
                ],
                "GO:0005575": [
                    "cellular_component",
                    "Gene Ontology Cellular Component annotations including both manually curated and electronic annotations."
                ],
                "ANNOT_TYPE_ID_PANTHER_GO_SLIM_MF": [
                    "PANTHER GO Slim Molecular Function",
                    "A molecular process that can be carried out by the action of a single macromolecular machine, usually via direct physical interactions with other molecular entities. Function in this sense denotes an action, or activity, that a gene product (or a complex) performs. These actions are described from two distinct but related perspectives: (1) biochemical activity, and (2) role as a component in a larger system/process."
                ],
                "ANNOT_TYPE_ID_PANTHER_GO_SLIM_BP": [
                    "PANTHER GO Slim Biological Process",
                    "A biological process represents a specific objective that the organism is genetically programmed to achieve. Biological processes are often described by their outcome or ending state, e.g., the biological process of cell division results in the creation of two daughter cells (a divided cell) from a single parent cell. A biological process is accomplished by a particular set of molecular functions carried out by specific gene products (or macromolecular complexes), often in a highly regulated manner and in a particular temporal sequence."
                ],
                "ANNOT_TYPE_ID_PANTHER_GO_SLIM_CC": [
                    "PANTHER GO Slim Cellular Location",
                    "A location, relative to cellular compartments and structures, occupied by a macromolecular machine when it carries out a molecular function. There are two ways in which the gene ontology describes locations of gene products: (1) relative to cellular structures (e.g., cytoplasmic side of plasma membrane) or compartments (e.g., mitochondrion), and (2) the stable macromolecular complexes of which they are parts (e.g., the ribosome)."
                ],
                "ANNOT_TYPE_ID_PANTHER_PC": [
                    "protein class",
                    ""
                ],
                "ANNOT_TYPE_ID_PANTHER_PATHWAY": [
                    "ANNOT_TYPE_PANTHER_PATHWAY",
                    "Panther Pathways"
                ],
                "ANNOT_TYPE_ID_REACTOME_PATHWAY": [
                    "ANNOT_TYPE_REACTOME_PATHWAY",
                    "Reactome Pathways"
                ]
            }
        for annotation, (name, description) in datasets.items():
            realname: str = self._names_rev[annotation]
            self._datasets[realname] = [annotation, name, description]
            
    def get_available(self) -> dict:
        return self._available

    def get_pantherdb_datasets(self, ) -> list:
        """Retrieves all available pantherDB datasets and returns them in a list of [annotation, \
            annotation_name, annotation_description]"""
        success: bool = False
        for i in range(20, 100, 20):
            try:
                request: requests.Response = requests.get(
                    'http://pantherdb.org/services/oai/pantherdb/supportedannotdatasets',
                    timeout=i
                )
                types: dict = json.loads(request.text)
                success = True
                break
            except requests.exceptions.ReadTimeout:
                continue
            except requests.exceptions.ConnectionError:
                continue
        if not success:
            return {}
        datasets: dict = {}
        for entry in types['search']['annotation_data_sets']['annotation_data_type']:
            annotation: str = entry['id']
            name: str = entry['label']
            description: str = entry['description']
            datasets[annotation] = (name, description)
        return datasets

    def __get_species_from_panther_datafiles(self, request: str, species_list: list) -> list:
        """Parses out wanted species datafiles from panther request
        """
        datafilelist = [a.split('href')[-1].split('<')[0].split('>')[-1].strip() for a in
                        request.text.split('\n')]
        datafilelist = [f for f in datafilelist if 'PTHR' in f]
        if not species_list:
            species_list: list = ['human']
        if species_list[0] != 'all':
            ndat: list = []
            for datafile in datafilelist:
                add: bool = False
                for spec in species_list:
                    if spec in datafile:
                        add = True
                if add:
                    ndat.append(datafile)
            datafilelist: list = ndat
        return datafilelist

    def retrieve_pantherdb_gene_classification(self, species: list = None,
                                               savepath: str = 'PANTHER datafiles',
                                               progress: bool = False) -> None:
        """Downloads PANTHER gene classification files for desired organisms.

        Will not download, if files with the same name already exist in the save directory.

        Parameters:
        species: list of species to download. If None, will download human only.\
            If 'all', will download all species files.
        savepath: directory in which to save the files.
        """
        pantherpath: str = 'http://data.pantherdb.org/ftp/sequence_classifications/current_release/\
            PANTHER_Sequence_Classification_files/'
        request: requests.Response = requests.get(pantherpath, timeout=10)
        datafilelist: list = self.__get_species_from_panther_datafiles(
            request, species)
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        already_have: set = set(os.listdir(savepath))
        panther_headers: list = 'FASTA header, UniProt, gene name, PTHR, Protein family, \
            protein relation smh, Panther GOMF_slim, Panther GOBP_slim, Panther GOCC_slim, \
                pathways, pathways2'.split(', ')
        for i, line in enumerate(datafilelist):
            if f'{line}.tsv' not in already_have:
                filepath: str = f'http://data.pantherdb.org/ftp/sequence_classifications/\
                    current_release/PANTHER_Sequence_Classification_files/{line}'
                request_2: requests.Response = requests.get(
                    filepath, timeout=10)
                dataframe: pd.DataFrame = pd.read_csv(
                    StringIO(request_2.text), sep='\t', names=panther_headers)
                dataframe.to_csv(os.path.join(
                    savepath, f'{line}.tsv'), sep='\t', index=False)
                # with open(os.path.join(savepath,f'{line}.tsv'),'w') as fil:
                #   fil.write(r2.text)
            if progress:
                print(f'{line} done, {len(datafilelist)-(i+1)} left')

    def get_default_panel(self) -> list:
        return self._defaults

    def enrich(self, parameters,data_lists: list, options: str) -> list:
        """
        """
        if options == 'defaults':
            datasets: list = self.get_default_panel()
        else:
            datasets = options.split(';')
        datasets = [self._datasets[d] for d in datasets]
        results: dict = {}
        legends: dict = {}
        logger.warning(f'Enrich: {len(datasets)}')
        for bait, preylist in data_lists:
            logger.warning(f'Enrich for {bait}: {len(preylist)}')
            for data_type_key, result in self.run_panther_overrepresentation_analysis(datasets, preylist, bait).items():
                results_df: pd.DataFrame = result['Results']
                results_df.insert(1, 'Bait', bait)
                if data_type_key not in results:
                    results[data_type_key] = []
                    legends[data_type_key] = []
                results[data_type_key].append(results_df)
                legends[data_type_key].append(result['Reference information'])
            logger.warning(f'Enrich for {bait}: done')
        result_names: list = []
        result_dataframes: list = []
        result_legends: list = []
        for annokey, result_dfs in results.items():
            result_names.append(annokey)
            result_dataframes.append(
                ('fold_enrichment', 'fdr', 'label', pd.concat(result_dfs)))
            result_legends.append(
                (annokey, '\n\n'.join(list(set(legends[annokey])))))
        logger.warning(f'Enrich done')
        return (result_names, result_dataframes, result_legends)

    def run_panther_overrepresentation_analysis(self, datasets: list, protein_list: list, data_set_name: str,
                                                background_list: list = None,
                                                organism: int = 9606,
                                                test_type: str = 'FISHER',
                                                correction_type: str = 'FDR') -> dict:
        """Runs statistical overrepresentation analysis on PANTHER server (pantherdb.org), and \
            returns the results as a dictionary.

        The output will contain dictionary with following keys:
            Name: name of the enrichment database
            Description: description of the database
            Reference information: information about tool, database, and analysis. E.g. versions
            Results: pandas dataframe with the full enrichment results

        Parameters:
        datasets: pantherDB datasets to run overrepresentation analysis against, see \
            get_pantherdb_datasets method
        protein_list: list of identified proteins
        background_list: list of background proteins. if None, entire annotation database \
            will be used
        organism: numerical ID of the organism, e.g. human is 9606
        test_type: statistical test to apply, see PANTHER documentation for options: \
            http://pantherdb.org/services/openAPISpec.jsp
        correction_type: correction to apply to p-values
        """
        baseurl: str = 'http://pantherdb.org/services/oai/pantherdb/enrich/overrep?'
        ret: dict = {}
        for annotation, name, description in datasets:
            data: dict = {
                'organism': organism,
                'refOrganism': organism,
                'annotDataSet': quote(annotation),
                'enrichmentTestType': test_type,
                'correction': correction_type,
                'geneInputList': ','.join(protein_list)
            }
            if background_list:
                data.update({'refInputList': ','.join(background_list)})

            final_url: str = baseurl
            for key, value in data.items():
                final_url += f'{key}={value}&'
            final_url = final_url.strip('&')
            reference_string: str = f'PANTHER overrepresentation analysis for {data_set_name} with {name}\n----------\n'
            success: bool = False
            logger.warning(f'Run enrichment: {data_set_name} {name}')
            for i in range(20, 100, 20):
                try:
                    request: requests.Response = requests.post(
                        final_url, timeout=i)
                    req_json: dict = json.loads(request.text)
                    success = True
                    logger.warning(f'Run enrichment: success')
                    break
                except requests.exceptions.ReadTimeout as e:
                    logger.warning(f'Run enrichment: fail-ReadTimeOut - {e}')
                    continue
                except requests.exceptions.ConnectionError as e:
                    logger.warning(f'Run enrichment: fail-ConnectionError - {e}')
                    continue
            if not success:
                ret[self._names_rev[annotation]] = {'Name': name, 'Description': description, 'Results': pd.DataFrame(),
                                                    'Reference information': 'PANTHER failed.'}
                continue
            try:
                reference_string += f'PANTHERDB reference information:\nTool release date: \
                    {req_json["results"]["tool_release_date"]}\nAnalysis run: {datetime.now()}\n'
            except KeyError as exc:
                logger.warning(f'Run enrichment: fail-KeyError - {exc}')
                raise exc
            reference_string += (
                f'Enrichment test type: '
                f'{req_json["results"]["enrichment_test_type"]}\n'
            )
            reference_string += f'Correction: {req_json["results"]["correction"]}\n'
            reference_string += f'Annotation: {req_json["results"]["annotDataSet"]}\n'
            reference_string += (
                f'Annotation version release date: '
                f'{req_json["results"]["annot_version_release_date"]}\n')
            reference_string += f'Database: {name}\nDescription: {description}\n'
            reference_string += '-----\nSearch:\n'
            for key, value in req_json['results']['search'].items():
                reference_string += f'{key}: {value}\n'
            reference_string += '-----\nReference:\n'
            for key, value in req_json['results']['reference'].items():
                reference_string += f'{key}: {value}\n'
            reference_string += '-----\nInput:'
            for key, value in req_json['results']['input_list'].items():
                if key not in {'mapped_id', 'unmapped_id'}:
                    reference_string += f'{key}: {value}\n'
            reference_string += '-----\n'
            if 'unmapped_id' in req_json['results']['input_list']:
                reference_string += (
                    f'Unmapped IDs: '
                    f'{", ".join(req_json["results"]["input_list"]["unmapped_id"])}\n'
                )
            reference_string += '-----\n'
            reference_string += (
                f'Mapped IDs: '
                f'{", ".join(req_json["results"]["input_list"]["mapped_id"])}\n'
            )
            reference_string += '-----\n'
            results: pd.DataFrame = pd.DataFrame(
                req_json['results']['result'])  # .keys()
            results = results.join(pd.DataFrame(list(results['term'].values))).\
                drop(columns=['term'])
            results.loc[:, 'DB'] = self._names_rev[annotation]
            order: list = ['DB', 'id', 'label']
            with np.errstate(divide='ignore'):
                results.loc[:, 'log2_fold_enrichment'] = np.log2(
                    results['fold_enrichment'])
            order.extend([c for c in results.columns if c not in order])
            results = results[order]

            ret[self._names_rev[annotation]] = {'Name': name, 'Description': description, 'Results': results,
                                                'Reference information': reference_string}
        return ret
