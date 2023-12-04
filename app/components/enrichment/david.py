import numpy as np
import pandas as pd
from suds.client import Client
from typing import Any
class handler():

    @property
    def nice_name(self) -> str:
        return self._nice_name

    def __init__(self) -> None:
        self._defaults: list = [
                'DAVID GOBP Direct',
                'DAVID Reactome',
                'DAVID Wikipathway',
            ]
        self._names:dict =  {
            'DAVID GOBP Direct': 'GOTERM_BP_DIRECT',
            'DAVID GOCC Direct': 'GOTERM_CC_DIRECT',
            'DAVID GOMF Direct': 'GOTERM_MF_DIRECT',
      #      'DAVID GOBP All': 'GOTERM_BP_ALL',
       #     'DAVID GOCC All': 'GOTERM_CC_ALL',
        #    'DAVID GOMF All': 'GOTERM_MF_ALL',
            'DAVID KEGG': 'KEGG_PATHWAY',
            'DAVID Reactome': 'REACTOME_PATHWAY',
            'DAVID Wikipathway': 'WIKIPATHWAY',
       #     'DAVID Interpro Domains': 'INTERPRO',
            'DAVID Pfam domains': 'PFAM',
       #     'DAVID Disgenet diseases': 'DISGENET',
       #     'DAVID GAD diseases': 'GAD_DISEASE',
        }
        self._nice_name: str = 'DAVID'
        self._available: list = []
        self._available = sorted(list(self._names.keys()))
        self._names_rev: dict = {v: k for k,v in self._names.items()}

    @property
    def handler_types(self) -> list:
        return list(self._available.keys())

    def get_available(self) -> dict:
        return self._available

    def get_default_panel(self) -> list:
        return self._defaults
    
    def enrich(self, data_lists:list, options: str) -> tuple:
        """Main enrich method.
        
        :returns: a tuple of (result_names, result_data, result_legends). Result names is a list of the names of different enrichments. Result data is a list of tuples of (fold change column name, dataframe), and result legends is a list of more elaborate information about each enrichment, if available.
        """
        if options == 'defaults':
            datasets: list = self.get_default_panel()
        else:
            datasets = options.split(';')
        datasets = [self._names[dataname] for dataname in datasets]
        results: dict = {}
        legends: dict = {}
        for bait, preylist in data_lists:
            for data_type_key, results_df in self.run_david_overrepresentation_analysis(datasets, preylist).items():
                if data_type_key not in results:
                    results[data_type_key] = [(pd.DataFrame(), bait, preylist)]
                    legends[data_type_key] = []
                results_df.insert(1, 'Bait', bait)
                results[data_type_key].append((results_df, bait, preylist))
                legends[data_type_key].append((bait,preylist))
        result_names: list = []
        result_dataframes: list = []
        result_legends: list = []
        
        for annokey, res_list in results.items():
            result_dfs = []
            legend = []
            for (results_df, bait, preylist)  in res_list:
                results_df['Bait'] = bait
                result_dfs.append(results_df)
                legend.append(f'DAVID enrichment analysis for {bait} with {annokey}\nAnalysis completed through DAVID API.\nUsed input list: {",".join(preylist)}\n-----\n')
            result_names.append(annokey)
            result_df: pd.DataFrame = pd.concat(result_dfs)
            with np.errstate(divide='ignore'):
                result_df.loc[:, 'log2_foldEnrichment'] = np.log2(
                    result_df['foldEnrichment'])
            result_dataframes.append(('foldEnrichment', 'afdr', 'termName', result_df))
            result_legends.append((annokey, '\n\n'.join(legend)))
        
        return (result_names, result_dataframes, result_legends)

    def suds_to_dict(self, suds_item: Any) -> dict:
        """Transcodes suds objects to a dictionary.
        Can handle nested suds objects as well, unlike the suds Client.dict(sucs_object) -method.
        :param suds_item: suds.sudsobject.simpleChartRecord object
        :returns:  dictionary representation of the input object
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
        ]

    def run_david_overrepresentation_analysis(self, david_categories: list, uniprot_idlist: list, bg_uniprot_list: list = None, sig_threshold: float = 0.05, sig_col:str = 'benjamini', fold_enrichment_threshold: float = 2, count_threshold: int = 2) -> tuple:
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
        if bg_uniprot_list is not None:
            input_bg_ids: str = ','.join(bg_uniprot_list)
            list_type = 1
            proportion_of_background_mapped = client.service.addList(input_bg_ids, id_type, bg_list_name, list_type)

        category_string: str = ','.join(david_categories)
        client.service.setCategories(category_string)
        chart_report: list = client.service.getChartReport(1,fold_enrichment_threshold)
        result_dataframe:pd.DataFrame = pd.DataFrame(data=[self.suds_to_dict(s) for s in chart_report])
        # Filter results
        if len(result_dataframe.columns) == 0:
            return {}
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
        new_category_names: list = []
        discarded: set = set()
        for cat_value in result_dataframe['categoryName'].values:
            if cat_value not in self._names_rev:
                discarded.add(cat_value)
                new_category_names.append('discard')
            else:
                new_category_names.append(self._names_rev[cat_value])
        result_dataframe['categoryName'] = new_category_names
        result_dataframe = result_dataframe[result_dataframe['categoryName']!='discard']
        result_dict: dict= {}
        for category in result_dataframe['categoryName'].unique():
            result_dict[category] = result_dataframe[result_dataframe['categoryName']==category]
        return result_dict