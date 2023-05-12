"""Quick module for database things. Should be replaced by a real database.

"""
import json
from typing import Union, Tuple
import os
import pandas as pd


class DbEngine:
    """dbengine class"""

    # pylint: disable=too-many-instance-attributes
    # Temporary class until moving to real DB, not the time to worry about too many attributes
    # TODO: Move to PostgreSQL

    _parameters: dict = {}
    _data: pd.DataFrame = pd.DataFrame()
    _last_id: int = -1
    _upload_table_data_columns: list = []
    _upload_table_data_dropdowns: dict = {}
    _protein_lengths: dict = {}
    _control_table: pd.DataFrame
    _controls: dict
    _crapome_table: pd.DataFrame
    _crapome_proteins: set
    _crapome: dict
    _override_keys: dict
    _contaminant_table: pd.DataFrame
    _contaminant_list: list
    _figure_names_and_legends: dict = {}
    _protein_names: dict

    @property
    def parameters(self) -> dict:
        """Get the current parameter dictionary."""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: Union[dict, str]) -> None:
        """Set the current parameter dictionary

        Args:
            parameters: A dictionary of the new parameters, or a filename pointing to a json file
                of parameters. 
        """
        if isinstance(parameters, str):
            parameters: dict = self.parse_parameters(parameters)
        for key, value in self._override_keys.items():
            parameters[key] = value
        self._parameters: dict = parameters
        self.data = self._parameters['Run data']
        self.temp_dir = self._parameters['Temporary data directory']
        self.protein_lengths = os.path.join(
            *self.parameters['files']['data']['protein lengths file'])
        self.protein_names = os.path.join(
            *self.parameters['files']['data']['protein names file'])
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.isdir(self.temp_dir):
            os.makedirs(self.temp_dir)

        with open(os.path.join(*parameters['files']['data']['control sets']), encoding='utf-8') as fil:
            self._controls = json.load(fil)
        self._control_table = pd.read_csv(
            os.path.join(*parameters['files']['data']['control table']),
            sep='\t', index_col='PROTID')
        self._control_table.index.name = 'index'

        with open(os.path.join(*parameters['files']['data']['crapome']), encoding='utf-8') as fil:
            self._crapome = json.load(fil)
        self._crapome_table = pd.read_csv(
            os.path.join(*parameters['files']['data']['crapome table']),
            sep='\t', index_col='PROTID')
        self._crapome_proteins = set(self._crapome_table.index.values)
        self.contaminant_table = os.path.join(
            *parameters['files']['data']['Contaminant list'])
        parameters['Figure legend dir'] = os.path.join(*parameters['Figure legend dir'])
        if os.path.isdir(parameters['Figure legend dir']):
            for root, _, files in os.walk(parameters['Figure legend dir']):
                for file in files:
                    with open(os.path.join(root,file),encoding='utf-8') as fil:
                        self._figure_names_and_legends[file] = fil.read()

    @property
    def controlsets(self) -> dict:
        return self._controls['sets']['all']
    
    
    @property
    def figure_data(self) -> dict:
        return self._figure_names_and_legends

    @property
    def default_controlsets(self) -> dict:
        return self._controls['sets']['default']

    @property
    def disabled_controlsets(self) -> int:
        return self._controls['sets']['disabled']

    @property
    def full_control_table(self) -> pd.DataFrame:
        return self._control_table

    def controls(self, control_list) -> pd.DataFrame:
        these_columns: list = []
        for control_group in control_list:
            these_columns.extend(self._controls[control_group])
        return self.full_control_table[these_columns].dropna(how='all')


# TODO: split crapome to multiple files, read requested files while saint is running?

    @property
    def crapomesets(self) -> dict:
        return self._crapome['sets']['all']

    @property
    def default_crapomesets(self) -> dict:
        return self._crapome['sets']['default']

    @property
    def disabled_crapomesets(self) -> int:
        return self._crapome['sets']['disabled']

    @property
    def full_crapome_table(self) -> pd.DataFrame:
        return self._crapome_table

    def crapome(self, crapome_list, proteins) -> Tuple[pd.DataFrame, list]:
        these_columns: list = []
        column_groups: list = []
        for crapome_group in crapome_list:
            these_columns.append(f'{crapome_group} AvgSpc')
            these_columns.append(f'{crapome_group} Frequency')
            column_groups.append([these_columns[-2], these_columns[-1]])
        ret_table: pd.DataFrame = self.full_crapome_table.loc[list(
            set(proteins) & self._crapome_proteins)]
        ret_table = ret_table[these_columns].dropna(how='all')
        return [
            ret_table,
            column_groups
        ]

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    @data.setter
    def data(self, data_frame: Union[pd.DataFrame, str]) -> None:
        if isinstance(data_frame, list):
            data_frame = os.sep.join(data_frame)
            data_frame: pd.DataFrame = pd.read_csv(data_frame, sep='\t')
        self._data: pd.DataFrame = data_frame
        self.last_id = self._data['id'].max()+1
        if len(self._upload_table_data_columns) == 0:
            self.set_upload_columns()

    def get_known(self, which_proteins) -> tuple:
        knowns: dict = {}
        for bait in which_proteins:
            kdict: dict = {}
            try:
                with open(os.path.join('data','known interactions','per bait',f'{bait}.json'), encoding = 'utf-8') as fil:
                    kdict= json.load(fil)
            except FileNotFoundError:
                pass
            knowns[bait] = kdict
        known_cols: set = set()
        for _,interactor_dict in knowns.items():
            for _, col_value_dict in interactor_dict.items():
                known_cols |= set(col_value_dict.keys())
        known_cols: list = sorted(list(known_cols))
        return (knowns, known_cols)
    
    def protein_lengths_from_fasta(self, fasta_file_name, uniprot=True) -> None:
        lens: dict = {}
        seqs: list = []
        protein_id: str
        with open(fasta_file_name, encoding='utf-8') as fil:
            for line in fil:
                if line.startswith('>'):
                    protein_id = line.strip().strip('>')
                    if uniprot:
                        protein_id = protein_id.split('|')[1]
                    lens[protein_id] = ''
                    seqs.append([protein_id, ''])
                else:
                    lens[protein_id] += line.strip()
                    seqs[-1][1] += line.strip()
        self.protein_lengths = lens

        with open(self.protein_seq_file, 'a', encoding='utf-8') as fil:
            for line in seqs:
                fil.write('\t'.join(line)+'\n')

    @property
    def protein_names(self) -> dict:
        return self._protein_names
    
    @protein_names.setter
    def protein_names(self, filename) -> None:
        with open(filename, encoding='utf-8') as fil:
            self._protein_names = json.load(fil)

    @property
    def protein_lengths(self) -> dict:
        return self._protein_lengths

    @protein_lengths.setter
    def protein_lengths(self, filename) -> None:
        if isinstance(filename, dict):
            with open(self._protein_lentgh_file, 'a', encoding='utf-8') as fil:
                current: int
                for key, value in filename.items():
                    try:
                        current = self._protein_lengths[key]
                    except KeyError:
                        current = -1
                    if current != value:
                        if key not in self._protein_lengths:
                            self._protein_lengths[key] = value
                            fil.write(f'{key}\t{value}\n')
        else:
            with open(filename, encoding='utf-8') as fil:
                next(fil)
                for line in fil:
                    line: list = line.strip().split('\t')
                    if len(line) < 2:
                        continue
                    self._protein_lengths[line[0]] = int(line[1])

    @property
    def known_types(self) -> dict:
        return self.parameters['Known types']

    @property
    def max_theoretical_spc(self) -> int:
        return self.parameters['Maximum psm ever theoretically encountered']

    @property
    def last_id(self) -> int:
        return self._last_id

    @last_id.setter
    def last_id(self, last_id: int) -> None:
        self._last_id: int = last_id

    @property
    def protein_lentgh_file(self) -> str:
        return self._protein_lentgh_file

    @protein_lentgh_file.setter
    def protein_lentgh_file(self, protein_lentgh_file_name: str) -> None:
        self._protein_lentgh_file: str = protein_lentgh_file_name

    @property
    def default_workflow(self) -> str:
        return self.implemented_workflows[self.parameters['Default workflow']]

    @property
    def implemented_workflows(self) -> list:
        return self.parameters['Implemented workflows']

    @property
    def imputation_options(self) -> dict:
        return self.parameters['Imputation methods']

    @property
    def default_imputation_method(self) -> dict:
        return self.parameters['Default imputation method']

    @property
    def normalization_options(self) -> dict:
        return self.parameters['Normalization methods']

    @property
    def default_normalization_method(self) -> dict:
        return self.parameters['Default normalization method']

    @property
    def files(self) -> dict:
        return self.parameters['files']

    @property
    def protein_seq_file(self) -> dict:
        return os.path.join(*self.parameters['files']['data']['Protein sequence file'])

    @property
    def upload_table_data_columns(self) -> list:
        return self._upload_table_data_columns

    @upload_table_data_columns.setter
    def upload_table_data_columns(self, upload_table_data_columns: list) -> None:
        self._upload_table_data_columns: list = upload_table_data_columns

    @property
    def contaminant_table(self) -> list:
        return self._contaminant_table

    @contaminant_table.setter
    def contaminant_table(self, contaminant_file: str) -> None:
        self._contaminant_table: pd.DataFrame = pd.read_csv(
            contaminant_file, sep='\t')
        self._contaminant_list = list(
            self._contaminant_table['Uniprot ID'].values)

    @property
    def contaminant_list(self) -> list:
        return self._contaminant_list

    @property
    def temp_dir(self) -> list:
        return self._temp_dir

    @temp_dir.setter
    def temp_dir(self, temp_dir: list) -> None:
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        self._temp_dir: list = temp_dir

    @property
    def upload_table_data_dropdowns(self) -> dict:
        return self._upload_table_data_dropdowns

    @upload_table_data_dropdowns.setter
    def upload_table_data_dropdowns(self, upload_table_data_dropdowns: dict) -> None:
        self._upload_table_data_dropdowns: list = upload_table_data_dropdowns

    def delete_empty_directories(self, basedir) -> None:
        for path, _, _ in os.walk(basedir, topdown=False):
            if len(os.listdir(path)) == 0:
                os.rmdir(path)

    @property
    def cache_dir(self) -> str:
        return self.parameters['cache dir']

    def get_temp_file(self, filename) -> str:
        return os.path.join(self.temp_dir, filename)

    def clear_session_data(self, uid) -> None:
        for root, _, files in os.walk(self.get_cache_dir(uid)):
            for filename in files:
                os.remove(os.path.join(root, filename))
        self.delete_empty_directories(self.get_cache_dir(uid))

    ## TODO: this does not work with current cache layout. Need to recursively delete directories.
    def clear_cache_fully(self) -> None:
        for root, _, files in os.walk(self.cache_dir):
            for filename in files:
                os.remove(os.path.join(root, filename))
        self.delete_empty_directories(self.cache_dir)

    def get_cache_file(self, session_folder_name, filename) -> str:
        cache_dir_for_session: str = self.get_cache_dir(session_folder_name)
        return os.path.join(cache_dir_for_session, filename)
    
    def get_cache_dir(self, session_folder_name) -> str:
        cache_dir_for_session: str = os.path.join(
            self.cache_dir, session_folder_name)
        if not os.path.isdir(cache_dir_for_session):
            os.makedirs(cache_dir_for_session)
        return cache_dir_for_session

    def scripts(self, scriptname) -> Tuple[str, str]:
        return (
            os.path.join(*self.parameters['script_dirs'][scriptname]),
            self.parameters['scripts'][scriptname]
        )

    def set_upload_columns(self) -> None:
        data: pd.DataFrame = self.data
        clist: list = []
        cdic: dict = {}
        for bname in data.columns:
            b_column: dict = {'id': f'{bname}_upload', 'name': bname}
            if bname in self.known_types:
                b_column['presentation'] = 'dropdown'
                cdic[f'{bname}_upload'] = {
                    'options': [{'label': known_value, 'value': known_value} for
                                known_value in self.known_types[bname]]
                }
            if bname != 'id':
                #b_column['hideable'] = True
                clist.append(b_column)
        self.upload_table_data_columns = clist
        self.upload_table_data_dropdowns = cdic

    def parse_parameters(self, parameter_file) -> dict:
        """Parses a json file into parameters dict"""
        with open(parameter_file, encoding='utf-8') as fil:
            return json.load(fil)

    def add_to_data(self, data_frame) -> None:
        pass  # self.data

    def get_protein_lengths(self, protein_ids) -> dict:
        plendic: dict = self.protein_lengths
        protein_ids: list = [p for p in protein_ids if p in plendic]
        return {protein_id: plendic[protein_id] for protein_id in protein_ids}
    def get_name(self, protein_id) -> str:
        try:
            return self.protein_names[protein_id]
        except KeyError:
            return protein_id

    def request_file(self, location, filename) -> str:
        file_location: list = self.files[location][filename]
        return os.path.join(*file_location)

    def __init__(self, override_keys=None) -> None:
        parameter_file: str = 'parameters.json'
        if override_keys is None:
            override_keys = {}
        self._override_keys = override_keys
        self.parameters = parameter_file
