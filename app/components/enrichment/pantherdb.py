import os
import pandas as pd
import requests
import json
import numpy as np
from io import StringIO
from urllib.parse import quote
from datetime import datetime


class handler():

    _defaults: list = [
            'Panther Reactome pathways',
            'Panther GOBP slim',
            'Panther GOMF slim',
            'Panther GOCC slim'
        ]
    _available:list = []
    _names:dict =  {
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

    def __init__(self, datasetfile:str = 'panther_datasets.json') -> None:
        self._available = sorted(list(self._names.keys()))
        self._names_rev = {v: k for k,v in self._names.items()}
        if datasetfile:
            if os.path.isfile(datasetfile):
                with open(datasetfile) as fil:
                    datasets: dict = json.load(fil)
            else:
                datasets = self.get_pantherdb_datasets()
                with open(datasetfile,'w',encoding='utf-8') as fil:
                    json.dump(datasets,fil)
        for annotation, (name, description) in datasets.items():
            realname: str = self._names_rev[annotation]
            self._datasets[realname] = [annotation, name, description]

    @property
    def handler_types(self) -> list:
        return list(self._available.keys())

    def get_available(self) -> dict:
        return self._available

    def get_pantherdb_datasets(self, ) -> list:
        """Retrieves all available pantherDB datasets and returns them in a list of [annotation, \
            annotation_name, annotation_description]"""
        success:bool = False
        for i in range(20, 100, 20):
            try:
                request:requests.Response = requests.get(
                    'http://pantherdb.org/services/oai/pantherdb/supportedannotdatasets',
                    timeout=i
                )
                types: dict = json.loads(request.text)
                success=True
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
        datafilelist: list = self.__get_species_from_panther_datafiles(request, species)
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
                request_2: requests.Response = requests.get(filepath, timeout=10)
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
    
    def enrich(self, data_lists:list, options: str) -> list:
        if options == 'defaults':
            datasets: list = self.get_default_panel()
        else:
            datasets = options.split(';')
        datasets = [self._datasets[d] for d in datasets]
        results: dict = {}
        legends: dict = {}
        for bait, preylist in data_lists:
            for data_type_key, result in self.run_panther_overrepresentation_analysis(datasets, preylist, bait).items():
                results_df: pd.DataFrame = result['Results']
                results_df.insert(1, 'Bait', bait)
                if data_type_key not in results:
                    results[data_type_key] = []
                    legends[data_type_key] = []
                results[data_type_key].append(results_df)
                legends[data_type_key].append(result['Reference information'])
        result_names: list = []
        result_dataframes: list = []
        result_legends: list = []
        for annokey, result_dfs in results.items():
            result_names.append(annokey)
            result_dataframes.append(('fold_enrichment', 'fdr','label', pd.concat(result_dfs)))
            result_legends.append((annokey, '\n\n'.join(list(set(legends[annokey])))))
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
            for i in range(20, 100, 20):
                try:
                    request:requests.Response = requests.post(final_url, timeout=i)
                    req_json: dict = json.loads(request.text)
                    success = True
                    break
                except requests.exceptions.ReadTimeout:
                    continue
                except requests.exceptions.ConnectionError:
                    continue
            try:
                reference_string += f'PANTHERDB reference information:\nTool release date: \
                    {req_json["results"]["tool_release_date"]}\nAnalysis run: {datetime.now()}\n'
            except KeyError as exc:
                print(final_url)
                print(annotation)
                print(name)
                print(description)
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
            if not success:
                ret[self._names_rev[annotation]] = {'Name': name, 'Description': description, 'Results': pd.DataFrame(),
                            'Reference information': reference_string}
                continue
            results: pd.DataFrame = pd.DataFrame(req_json['results']['result'])  # .keys()
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