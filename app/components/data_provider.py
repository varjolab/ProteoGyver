from components import db_functions
from components.tools import utils
import os

parameters = utils.read_toml('parameters.toml')
db_file = os.path.join(*parameters['Data paths']['Database file'])

def map_protein_info(uniprot_ids: list, info: list | str = None, placeholder: list | str = None):
    """Map information from the protein table.
    :param uniprot_ids: IDs to map. If ID is not found, placeholder value will be used
    :param info: if str, returned list will have only the mapped values from this column. If type is list, will return a list of lists. By default, will return gene_name column data.
    :param placeholder: Value to use if ID is not found. By default, value from the uniprot_ids list will be used. If type list, the placeholders should be in the same order as info list. 
    """
    ret_info = []
    if info is None:
        info = 'gene_name'
    if isinstance(info, str):
        info = [info]
    if placeholder is None:
        placeholder = 'PLACEHOLDER_IS_INPUT_UPID'
    if isinstance(placeholder, str):
        placeholder = [placeholder for _ in info]
    return_mapping = {}
    for _, row in db_functions.get_from_table_by_list_criteria(
            db_functions.create_connection(db_file),
            'proteins',
            'uniprot_id',
            uniprot_ids,
        ).iterrows():
        return_mapping[row['uniprot_id']] = [row[ic] for ic in info]
    for uniprot_id in (set(uniprot_ids)-set(return_mapping.keys())):
        return_mapping[uniprot_id] = [
            uniprot_id if placeholder[i]=='PLACEHOLDER_IS_INPUT_UPID' else placeholder[i] 
            for i in range(len(info))
        ]
    retlist = [
        return_mapping[upid] 
        for upid in uniprot_ids
    ]
    if len(retlist[0]) == 1:
        retlist = [r[0] for r in retlist]
    return retlist