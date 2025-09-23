from importlib import util as import_util
import os
import pandas as pd
from components import parsing
from components.tools import utils


class EnrichmentAdmin:
    def __init__(self, parameters_file: str) -> None:
        parameters: dict = utils.read_toml(parameters_file)

        self._enrichment_handlers: dict = {}
        self._enrichments: dict = {}
        self._defaults: dict = {}
        self._disabled: list = []
        self._handler_basedir: str = os.path.join(*parameters['Module paths']['Enrichers'])
        
        for _module_filename in os.listdir(self._handler_basedir):
            if not _module_filename.endswith('.py'):
                continue
            if _module_filename.startswith('__'):
                continue
            _filepath: str = os.path.join(self._handler_basedir, _module_filename)
            _module_name: str = _module_filename.rsplit('.',maxsplit=1)[0]
            _spec = import_util.spec_from_file_location(
                'module.name', _filepath)
            _api_module = import_util.module_from_spec(_spec)
            _spec.loader.exec_module(_api_module)
            _handler = _api_module.handler()
            self._enrichment_handlers[_module_name] = {
                'handler': _handler,
                'available': _handler.get_available(),
                'name': _handler.nice_name,
                'defaults': _handler.get_default_panel()
            }
            for a in self._enrichment_handlers[_module_name]['available']:
                self._enrichments[a] = _module_name
            for a in self._enrichment_handlers[_module_name]['defaults']:
                show = True
                for ban_str in parameters['file loading']['Do not show in enrichment default']:
                    if ban_str in a.lower():
                        show = False
                if show:
                    self._defaults[a] = _module_name

        self.import_handlers()
    def get_available(self) -> list:
        return sorted(list(self._enrichments.keys()))
    def get_default(self) -> list:
        return sorted(list(self._defaults.keys()))
    def get_disabled(self) -> list:
        return sorted(self._disabled)
    def import_handlers(self) -> dict:
        ret_dict: dict = {}
        for module_filename in os.listdir(self._handler_basedir):
            if module_filename.endswith('.py'):
                filepath: str = os.path.join(self._handler_basedir, module_filename)
                module_name: str = module_filename.rsplit('.',maxsplit=1)[0]
                spec = import_util.spec_from_file_location('module.name', filepath)
                api_module = import_util.module_from_spec(spec)
                spec.loader.exec_module(api_module)
                ret_dict[module_name] = api_module.handler()
        self._imported_handlers: dict = ret_dict

    def enrich_all(self, data_table: pd.DataFrame,enrichment_strings: list, id_column: str = None, id_list: list = None, split_by_column: str = None, split_name: str = None) -> list:
        assert ((id_column is not None) or (id_list is not None)), 'Supply either id_column or id_list'
        if split_by_column is not None:
            if split_name is None:
                split_name = 'Sample group'
        else:
            assert id_list is None, 'Can not supply id_list with split_by_column!'
        
        enrichments_to_do: dict = {}
        for e_str in enrichment_strings:
            apiname = self._enrichments[e_str]
            if apiname not in enrichments_to_do:
                enrichments_to_do[apiname] = []
            enrichments_to_do[apiname].append(e_str)
        enrichment_results: list = []
        enrichment_names: list = []
        done_info: list = []
        for api, enrichmentlist in enrichments_to_do.items():
            try:
                enrichment_options: str = ';'.join(enrichmentlist)
                enrichment_input = []
                if split_by_column:
                    for b in data_table[split_by_column].unique():
                        df = data_table[data_table[split_by_column]==b]
                        enrichment_input.append([b, list(df[id_column].values)])
                else:
                    if id_list:
                        enrichment_input.append(['All',id_list])
                    else:
                        enrichment_input.append(['All',df[id_column]])
                result_names: list
                return_dataframes: list
                done_information: list
                handler = self._imported_handlers[api]
                result_names, return_dataframes, done_information = handler.enrich(enrichment_input, enrichment_options)
            except Exception as e:
                #TODO move to logging module
                print(f'Error in enrichment {api}: {e}')
                result_names = ['Error']
                return_dataframes = ['','','',pd.DataFrame()]
                done_information = ['Enrichment failed.']
                continue
            enrichment_results.extend(return_dataframes)
            enrichment_names.extend(result_names)
            done_info.extend(done_information)
        return (enrichment_names, enrichment_results, done_info)
