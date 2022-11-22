"""Quick module for database things

"""
import json
from typing import Union
import pandas as pd

class DbEngine:
    """dbengine class"""

    # pylint: disable=too-many-instance-attributes
    # Temporary class until moving to real DB, not the time to worry about too many attributes

    _parameters = {}
    _data = pd.DataFrame()
    _last_id = -1
    _upload_table_data_columns = []
    _upload_table_data_dropdowns = {}


    @property
    def parameters(self) -> dict:
        return self._parameters
    @parameters.setter
    def parameters(self, parameters:Union[dict,str]) -> None:
        if isinstance(parameters,str):
            parameters: dict = self.parse_parameters(parameters)
        self._parameters: dict = parameters
        self.data = self._parameters['Run data']

    @property
    def data(self) -> pd.DataFrame:
        return self._data
    @data.setter
    def data(self, data_frame:Union[pd.DataFrame,str]) -> None:
        if isinstance(data_frame,str):
            data_frame: pd.DataFrame = pd.read_csv(data_frame,sep='\t')
        self._data: pd.DataFrame = data_frame
        self.last_id = self._data['id'].max()+1
        if len(self._upload_table_data_columns) == 0:
            self.set_upload_columns()

    @property
    def known_types(self) -> dict:
        return self._parameters['Known types']

    @property
    def last_id(self) -> int:
        return self._last_id
    @last_id.setter
    def last_id(self, last_id:int) -> None:
        self._last_id:int = last_id

    @property
    def upload_table_data_columns(self) -> list:
        return self._upload_table_data_columns
    @upload_table_data_columns.setter
    def upload_table_data_columns(self, upload_table_data_columns:list) -> None:
        self._upload_table_data_columns:list = upload_table_data_columns

    @property
    def upload_table_data_dropdowns(self) -> dict:
        return self._upload_table_data_dropdowns
    @upload_table_data_dropdowns.setter
    def upload_table_data_dropdowns(self, upload_table_data_dropdowns:dict) -> None:
        self._upload_table_data_dropdowns:list = upload_table_data_dropdowns

    def set_upload_columns(self) -> None:
        data: pd.DataFrame = self.data
        clist: list = []
        cdic: dict = {}
        for bname in data.columns:
            b_column: dict = {'id': f'{bname}_upload', 'name': bname}
            if bname in self.known_types:
                b_column['presentation'] = 'dropdown'
                cdic[f'{bname}_upload'] = {
                    'options': [{'label': known_value, 'value': known_value} for \
                        known_value in self.known_types[bname]]
                }
            if bname != 'id':
                #b_column['hideable'] = True
                clist.append(b_column)
        self.upload_table_data_columns = clist
        self.upload_table_data_dropdowns = cdic

    def parse_parameters(self, parameter_file) -> dict:
        with open(parameter_file, encoding='utf-8') as fil:
            return json.load(fil)

    def add_to_data(self,data_frame) -> None:
        pass #self.data


    def __init__(self) -> None:
        parameter_file: str = 'parameters.json'
        self.parameters = parameter_file