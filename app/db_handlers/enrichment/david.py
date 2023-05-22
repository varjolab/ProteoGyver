import pandas as pd
import requests
import json
import numpy as np

from urllib.parse import quote
from datetime import datetime
from suds.client import Client
from datetime import datetime
from typing import Any

class handler():

    @property
    def nice_name(self) -> str:
        return self._nice_name
    
    @property
    def legends(self) -> dict:
        return self._legends

    def __init__(self) -> None:
        self._defaults: list = [
                'GOBP Direct',
                'GOCC Direct',
                'GOMF Direct',
                'KEGG',
                'Reactome',
                'Wikipathway',
            ]
        self._names:dict =  {
            'GOBP Direct': 'GOTERM_BP_DIRECT',
            'GOCC Direct': 'GOTERM_CC_DIRECT',
            'GOMF Direct': 'GOTERM_MF_DIRECT',
            'GOBP All': 'GOTERM_BP_ALL',
            'GOCC All': 'GOTERM_CC_ALL',
            'GOMF All': 'GOTERM_MF_ALL',
            'KEGG': 'KEGG_PATHWAY',
            'Reactome': 'REACTOME_PATHWAY',
            'Wikipathway': 'WIKIPATHWAY',
            'Interpro Domains': 'INTERPRO',
            'Pfam domains': 'PFAM',
            'Disgenet diseases': 'DISGENET',
            'GAD diseases': 'GAD_DISEASE',
        }
        self._nice_name: str = 'DAVID'
        self._available: dict = {}
        self._available['enrichment'] = sorted(list(self._names.keys()))
        self._names_rev: dict = {v: k for k,v in self._names.items()}
        self._legends: dict = self._names_rev

    @property
    def handler_types(self) -> list:
        return list(self._available.keys())

    def get_available(self) -> dict:
        return self._available

    def get_default_panel(self) -> list:
        return self._defaults
    
    def enrich(self, data_lists:list, options: str) -> list:
        if options == 'defaults':
            datasets: list = self.get_default_panel()
        else:
            datasets = options.split(';')
        results: dict = {}
        for bait, preylist in data_lists:
            for data_type_key, result in self.run_david_overrepresentation_analysis(datasets, preylist).items():
                if data_type_key not in results:
                    results[data_type_key] = []
                results_df: pd.DataFrame = result['Results']
                results_df.insert(1, 'Bait', bait)
                results[data_type_key].append(results_df)
        result_names: list = []
        result_dataframes: list = []
        result_legends: list = []
        for annokey, result_dfs in results.items():
            result_names.append(annokey)
            result_dataframes.append(pd.concat(result_dfs))
            result_legends.append(self.legends[annokey])
        return (result_names, result_dataframes, result_legends)


    def suds_to_dict(self, suds_item: Any) -> dict:
        """Transcodes suds objects to a dictionary.
        Can handle nested suds objects as well, unlike the suds Client.dict(sucs_object) -method.
        :param suds_item: suds.sudsobject.simpleChartRecord object
        :return:  dictionary representation of the input object
        """
        # Thanks to radtek on stackoverflow for most of this function: https://stackoverflow.com/questions/17581731/parsing-suds-soap-complex-data-type-into-python-dict
        # if object doesn't have __keylist__, we can't iterate it and should just return it as is.
        if not hasattr(suds_item, '__keylist__'):
            return suds_item
        output_dictionary: dict = {}
        for key in suds_item.__keylist__:
            value = getattr(suds_item, key)
            if isinstance(value,list):
                output_dictionary[key] = [
                    self.suds_to_dict(list_item) for list_item in value
                ]
            else:
                output_dictionary[key] = self.suds_to_dict(value)

        return output_dictionary
    
    def get_sig_cols(self)-> list:
        """Returns a list of significance columns available in DAVID output"""
        return [
            'fisher',
            'ease',
            'bonferroni',
            'EASEBonferroni',
            'benjamini',
            'afdr',
            'rfdr'
        ]

    def run_david_overrepresentation_analysis(self, david_categories: list, uniprot_idlist: list, bg_uniprot_list: list = None, sig_threshold: float = 0.05, sig_col:str = 'afdr', fold_enrichment_threshold: float = 2, count_threshold: int = 2) -> tuple:
        """Runs statistical overrepresentation analysis on DAVID server (david.ncifcrf.gov), and \
            returns the results as a dictionary.

        The output will contain 

        :param david_categories: DAVID categories, see get_available method.
        :param uniprot_idlist: list of identified proteins
        :param bg_uniprot_list: list of background proteins, defaults to DAVID default background
        :param sig_threshold: significance threshold for the final output
        :param sig_col: column to use for significance testing, see get_sig_cols method.
        :param fold_enrichment_threshold: threshold to filter results based on fold change.
        :param count_threshold: threshold to filter results based on annotated protein count

        :returns: dictionary with {categoryName: resultDataFrame}
        """

        input_list_name: str = 'InputList'
        bg_list_name: str = 'Background'
        url = 'https://david.ncifcrf.gov/webservice/services/DAVIDWebService?wsdl'
        client: Client = Client(url)
        client.wsdl.services[0].setlocation('https://david.ncifcrf.gov/webservice/services/DAVIDWebService.DAVIDWebServiceHttpSoap11Endpoint/')
        client.service.authenticate('kari.salokas@helsinki.fi')
        list_type: int = 0 # 0 for sample, 1 for background
        id_type: str = 'UNIPROT_ACCESSION'
        input_ids: str = ','.join(uniprot_idlist)

        # These proportions are unused for now.
        proportion_of_input_mapped: float =  client.service.addList(input_ids, id_type, input_list_name, list_type)
        proportion_of_background_mapped: float = -1
        if bg_uniprot_list:
            input_bg_ids: str = ','.join(bg_uniprot_list)
            list_type = 1
            proportion_of_background_mapped = client.service.addList(input_bg_ids, id_type, bg_list_name, list_type)

        category_string: str = ','.join(david_categories)
        client.service.setCategories(category_string)
        chart_report: list = client.service.getChartReport(1,fold_enrichment_threshold)
        result_dataframe:pd.DataFrame = pd.DataFrame(data=[self.suds_to_dict(s) for s in chart_report])

        # Filter results
        result_dataframe = result_dataframe[result_dataframe[sig_col]<sig_threshold]
        result_dataframe = result_dataframe[result_dataframe['listHits']>count_threshold]
        # Reorganize columns
        result_dataframe = result_dataframe[[
                            'categoryName',
                            'termName',
                            'id',
                            'geneIds',
                            'listHits',
                            'listTotals',
                            'popHits',
                            'popTotals',
                            'percent',
                            'foldEnrichment',
                            'fisher',
                            'ease',
                            'bonferroni',
                            'EASEBonferroni',
                            'benjamini',
                            'afdr',
                            'rfdr'
                        ]]
        result_dict: dict= {}
        for category in result_dataframe['categoryName'].unique():
            result_dict[category] = result_dataframe[result_dataframe['categoryName']==category]
        return result_dict
