"""QC analysis functions for ProteoGyver"""

import pandas as pd
import numpy as np
from pyrsistent import discard

def check_bait(bait_entry: str) -> str:
    """Checks if a string contains a valid bait name.
    
    :returns: a string representation of the bait. 
    """
    bval: str = ''
    if bait_entry is not None:
        bval = str(bait_entry)
    if (len(bval) == 0) or (bval == 'nan'):
        bval = 'No bait uniprot'
    return bval

def guess_controls(sample_groups: dict) -> tuple:
    """Guesses controls from sample groups.
    
    Any samples with GFP in the name are assumed to be controls.
    :returns: tuple of (list of control sample groups, list of control samples)
    """
    control_groups: list = []
    control_samples: list = []
    for group_name, samples in sample_groups.items():
        if 'gfp' in group_name.lower():
            control_groups.append(group_name)
            control_samples.append(samples)
    return (control_groups, control_samples)
def format_data(session_uid: str, data_tables: dict, data_info: dict, expdes_table: dict, expdes_info: dict, contaminants_to_remove: list) -> dict:
    """Formats data formats into usable form and produces a data dictionary for later use"""

    intensity_table: pd.DataFrame = pd.read_json(data_tables['int'],orient='split')
    spc_table: pd.DataFrame = pd.read_json(data_tables['spc'],orient='split')
    expdesign: pd.DataFrame = pd.read_json(expdes_table, orient='split')

    sample_groups: dict
    discarded_columns: list
    used_columns: list
    sample_groups, discarded_columns, used_columns = rename_columns_and_update_expdesign(
        expdesign,
        [intensity_table, spc_table]
    )
    spc_table = spc_table[sorted(list(spc_table.columns))]

    if len(intensity_table.columns) > 1:
        intensity_table = intensity_table[sorted(list(intensity_table.columns))]
        untransformed_intensity_table: pd.DataFrame = intensity_table
        intensity_table = intensity_table.apply(np.log2)
    else:
        untransformed_intensity_table = intensity_table
    
    wcont_spc_table: pd.DataFrame = spc_table
    wcont_untransformed_intensity_table: pd.DataFrame = untransformed_intensity_table
    wcont_intensity_table: pd.DataFrame = intensity_table
    if len(contaminants_to_remove) > 0:
        spc_table = spc_table.loc[[i for i in spc_table.index if i not in contaminants_to_remove]]
        untransformed_intensity_table = untransformed_intensity_table.loc[[i for i in untransformed_intensity_table.index if i not in contaminants_to_remove]]
        intensity_table = intensity_table.loc[[i for i in intensity_table.index if i not in contaminants_to_remove]]
    
    return_dict: dict = {
        'sample groups': sample_groups,
        'data tables': {
            'raw intensity': untransformed_intensity_table.to_json(orient='split'),
            'spc': spc_table.to_json(orient='split'),
            'intensity': intensity_table.to_json(orient='split'),
            'experimental design': expdesign.to_json(orient='split'),
            'with-contaminants': {
                'raw intensity': wcont_untransformed_intensity_table.to_json(orient='split'),
                'spc': wcont_spc_table.to_json(orient='split'),
                'intensity': wcont_intensity_table.to_json(orient='split'),
            }
        }, 
        'info': {
            'discarded columns': discarded_columns,
            'used columns': used_columns,
            'data type': data_info['data type'],
            'Expdes based experiment type': 'Proteomics/Phosphoproteomics'
        },
        'file info': {
            'Data': {
                'File modified': data_info['Modified time'],
                'File name': data_info['File name']
            },
            'Sample table': {
                'File modified': expdes_info['Modified time'],
                'File name': expdes_info['File name']
            }
        },
        'other': {
            'session name': session_uid,
            'protein lengths': data_info['protein lengths'],
        }
    }
    if 'Bait uniprot' in expdesign.columns:
        return_dict['other']['bait uniprots'] = {}
        for _,row in expdesign.iterrows():
            return_dict['other']['bait uniprots'][row['Sample group']] = check_bait(row['Bait uniprot'])
        return_dict['info']['Expdes based experiment type'] = 'Interactomics'

    if len(intensity_table.columns) < 2:
        return_dict['data tables']['table to use'] = 'spc'
        return_dict['other']['all proteins'] =  list(spc_table.index)
    else:
        return_dict['data tables']['table to use'] = 'intensity'
        return_dict['other']['all proteins'] =  list(intensity_table.index)

    return_dict['sample groups']['guessed control samples'] = guess_controls(sample_groups)

    return return_dict

def delete_samples(discard_samples, data_dictionary) -> dict:
    print(f'Implement sample deletion. Selected samples: {", ".join(discard_samples)}')
    for table_name, table_json in data_dictionary['data tables'].items():
        if table_name == 'table to use':
            continue
        table_without_discarded_samples: pd.DataFrame = pd.read_json(table_json,orient='split')
        if table_name == 'experimental design':
            table_without_discarded_samples = table_without_discarded_samples[
                ~table_without_discarded_samples['Sample name'].isin(discard_samples)
            ]
        else:
            table_without_discarded_samples = table_without_discarded_samples[
                [c for c in table_without_discarded_samples.columns if c not in discard_samples]
            ]
        data_dictionary['data tables'][table_name] = table_without_discarded_samples.to_json(
            orient='split'
        )
    sg_dict: dict = {'norm': {}, 'rev': {}}
    for sample_group_name, sample_group_samples in data_dictionary['sample groups']['norm'].items():
        sg_dict['norm'][sample_group_name] = [
            s_name for s_name in sample_group_samples if s_name not in discard_samples
        ]
    for group, samples in sg_dict['norm'].items():
        for sample in samples:
            sg_dict['rev'][sample] = group
    data_dictionary['sample groups'] = sg_dict
    data_dictionary['user-discarded samples'] = discard_samples

    return data_dictionary

def rename_columns_and_update_expdesign(
        expdesign: pd.DataFrame,
        tables: list) -> tuple:
    """Modified expdes and data tables to discard samples not mentioned in the sample table.
    
    :returns: tuple of (sample_groups, discarded columns, used_columns)
    """
    # Get rid of file paths and timstof .d -file extension, if present:
    expdesign['Sample name'] = [
        oldvalue.rsplit('\\', maxsplit=1)[-1]\
            .rsplit('/', maxsplit=1)[-1]\
            .rstrip('.d') for oldvalue in expdesign['Sample name'].values
    ]
    discarded_columns:list = []
    sample_groups: dict = {}
    sample_group_columns: dict = {}
    rev_intermediate_renaming: list = []
    for table_ind, table in enumerate(tables):
        intermediate_renaming: dict = {}
        rev_intermediate_renaming.append([])
        if len(table.columns) < 2:
            continue
        for column_name in table.columns:
            # Discard samples that are not named
            col: str = column_name
            if col not in expdesign['Sample name'].values:
                # Try to see if the column without possible file path is in the expdesign:
                col = col.rsplit(
                    '\\', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1]
                if col not in expdesign['Sample name'].values:
                    col = col.rsplit('.d',maxsplit=1)[0]
                    if col not in expdesign['Sample name'].values:
                        # Discard column if not found
                        discarded_columns.append(col)
                        continue
            intermediate_renaming[column_name] = col
            sample_group: str = expdesign[expdesign['Sample name']
                                        == col].iloc[0]['Sample group']
            # If no value is available for sample in the expdesign
            # (but sample column name is there for some reason), discard column
            if pd.isna(sample_group):
                continue
            newname: str = str(sample_group)
            # We expect replicates to not be specifically named; they will be named here.
            if newname[0].isdigit():
                newname = f'Sample_{newname}'
            if newname not in sample_group_columns:
                sample_group_columns[newname] = [[] for _ in range(len(tables))]
            sample_group_columns[newname][table_ind].append(col)
        if len(intermediate_renaming.keys()) > 0:
            table.rename(columns=intermediate_renaming, inplace = True)
        rev_intermediate_renaming[-1] = {value: key for key,value in intermediate_renaming.items()}
    column_renames: list = [{} for _ in range(len(tables))]
    used_columns: list = [{} for _ in range(len(tables))]
    
    for nname, list_of_all_table_columns in sample_group_columns.items():
        first_len: int = 0
        for table_index, table_columns in enumerate(list_of_all_table_columns):
            if len(table_columns) < 2:
                continue
            if first_len == 0:
                first_len = len(table_columns)
            else:
                # Should have same number of columns/replicates for SPC and intensity tables
                assert len(table_columns) == first_len
            for column_name in table_columns:
                i: int = 1
                while f'{nname}_Rep_{i}' in column_renames[table_index]:
                    i += 1
                newname_to_use: str = f'{nname}_Rep_{i}'
                if nname not in sample_groups:
                    sample_groups[nname] = set()
                sample_groups[nname].add(newname_to_use)
                column_renames[table_index][newname_to_use] = column_name
                used_columns[table_index][newname_to_use] = rev_intermediate_renaming[table_index][column_name]
    sample_groups = {
        'norm': {k: sorted(list(v)) for k, v in sample_groups.items()},
        'rev': {}
    }
    for group, samples in sample_groups['norm'].items():
        for sample in samples:
            sample_groups['rev'][sample] = group
    for table_index, table in enumerate(tables):
        rename_columns: dict = {value: key for key, value in column_renames[table_index].items()}
        table.drop(
            columns= [c for c in table.columns if c not in rename_columns],
            inplace=True
            )
        table.rename(columns=rename_columns, inplace=True)

    return (sample_groups, discarded_columns, used_columns)


def validate_basic_inputs(*args) -> bool:
    """Validates the basic inputs of proteogyver"""
    not_valid:bool = False
    for arg in args:
        if arg is None:
            not_valid = True
    for style_arg in args[-2:]:
        if style_arg['background-color'] == 'red':
            not_valid = True
    return not_valid