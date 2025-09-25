
import pandas as pd
import numpy as np
from statsmodels.stats import multitest
from scipy.stats import ttest_ind, ttest_rel, f_oneway
from typing import Any
from components.db_functions import map_protein_info

def anova(dataframe: pd.DataFrame, sample_groups: dict) -> pd.DataFrame:
    # Create an empty DataFrame to store the results
    results = []
    index = []
    for protein in dataframe.index:
        # Extract the values for each group for the current protein
        groups = [dataframe.loc[protein, samples].dropna().values for samples in sample_groups.values()]
        # Perform the ANOVA test for the current protein across all sample groups
        f_stat, p_value = f_oneway(*groups)
        results.append([f_stat, p_value])
        index.append(protein)
    ret_df: pd.DataFrame = pd.DataFrame(data=results,index=index,columns=['F-static','p-value'])
    _, p_value_adj, _, _ = multitest.multipletests(ret_df['p-value'], method='fdr_bh')
    ret_df['q-value'] = p_value_adj
    return ret_df


def differential(data_table: pd.DataFrame, sample_groups: dict, comparisons: list, data_is_log2_transformed: bool = True, namemap: dict = None, adj_p_thr: float = 0.01, fc_thr:float = 1.0, test_type: str = 'independent', db_file_path: str = None) -> pd.DataFrame:
    sig_data: list = []
    for sample, control in comparisons:
        sample_columns: list = sample_groups[sample]
        control_columns: list = sample_groups[control]
        if data_is_log2_transformed:
            log2_fold_change: pd.Series = data_table[sample_columns].mean(
                axis=1) - data_table[control_columns].mean(axis=1)
        else:
            log2_fold_change: pd.Series = np.log2(data_table[sample_columns].mean(
                axis=1)) - np.log2(data_table[control_columns].mean(axis=1))
        sample_mean_val: pd.Series = data_table[sample_columns].mean(axis=1)
        control_mean_val: pd.Series = data_table[control_columns].mean(axis=1)
        # Calculate the p-value for each protein using a two-sample t-test
        if test_type == 'independent':
            p_value: float = data_table.apply(lambda x: ttest_ind(x[sample_columns], x[control_columns])[1], axis=1)
        elif test_type == 'paired':
            p_value: float = data_table.apply(lambda x: ttest_rel(x[sample_columns], x[control_columns])[1], axis=1)

        # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
        _: Any
        p_value_adj: np.ndarray
        _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

        # Create a new dataframe containing the fold change and adjusted p-value for each protein
        result: pd.DataFrame = pd.DataFrame(
            {
                'fold_change': log2_fold_change, 
                'p_value_adj': p_value_adj,
                'p_value_adj_neg_log10': -np.log10(p_value_adj),
                'p_value': p_value,
                'sample_mean_value': sample_mean_val,
                'control_mean_value': control_mean_val})
        if namemap:
            result['Name'] = [namemap[i] for i in data_table.index.values]
            result['Identifier'] = data_table.index
        else:
            result['Name'] = data_table.index.values
        result['Gene']  = map_protein_info(result.index, db_file_path=db_file_path)
        result['Sample'] = sample
        result['Control'] = control
        result['Significant'] = ((result['p_value_adj']<adj_p_thr) & (result['fold_change'].abs() > fc_thr))
        result.sort_values(by='Significant',ascending=True,inplace=True)
        #result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
        sig_data.append(result)
    return pd.concat(sig_data,ignore_index=True)[
        ['Sample',
         'Control',
         'Name',
         'Gene',
         'Significant',
         'fold_change',
         'p_value',
         'p_value_adj',
         'p_value_adj_neg_log10',
         'sample_mean_value',
         'control_mean_value'
         ]]

def get_count_data(data_table: pd.DataFrame, contaminant_list: list = None) -> pd.DataFrame:
    """Returns non-na count per column."""
    data: pd.DataFrame
    data = data_table.\
        notna().sum().\
        to_frame(name='Protein count')
    data.index.name = 'Sample name'
    if contaminant_list is not None:
        contaminants = data_table[data_table.index.isin(
            contaminant_list)].notna().sum()
        data['Protein count'] = data['Protein count'] - contaminants
        data['Is contaminant'] = False
        cont_data: pd.DataFrame = contaminants.to_frame(name='Protein count')
        cont_data['Is contaminant'] = True
        data = pd.concat([cont_data, data]).reset_index().rename(
            columns={'index': 'Sample name'})
        data.index = data['Sample name']
        data = data.drop(columns='Sample name')
    return data


def get_coverage_data(data_table: pd.DataFrame) -> pd.DataFrame:
    """Returns coverage pd.DataFrame."""
    return pd.DataFrame(
        data_table.notna()
        .astype(int)
        .sum(axis=1)
        .value_counts()
    ).rename(columns={'count': 'Identified in # samples'})


def get_na_data(data_table: pd.DataFrame) -> pd.DataFrame:
    """Returns na count per column."""
    data: pd.DataFrame = ((data_table.
                        isna().sum() / data_table.shape[0]) * 100).\
        to_frame(name='Missing value %')
    data.index.name = 'Sample name'
    return data


def get_sum_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.sum().\
        to_frame(name='Value sum')
    data.index.name = 'Sample name'
    return data


def get_mean_data(data_table) -> pd.DataFrame:
    data: pd.DataFrame = data_table.mean().\
        to_frame(name='Value mean')
    data.index.name = 'Sample name'
    return data


def get_comparative_data(data_table, sample_groups) -> tuple:
    sample_group_names: list = sorted(
        list(set([g for _, g in sample_groups.items()])))
    comparative_data: list = []
    for sample_group_name in sample_group_names:
        sample_columns: list = [
            sn for sn, sg in sample_groups.items() if sg == sample_group_name]
        comparative_data.append(data_table[sample_columns])
    return (
        sample_group_names,
        comparative_data
    )

def get_common_data(data_table: pd.DataFrame, rev_sample_groups: dict, only_groups: list = None) -> dict:
    group_sets: dict = {}
    for column in data_table.columns:
        col_proteins: set = set(data_table[[column]].dropna().index.values)
        group_name: str = rev_sample_groups[column]
        if (only_groups is not None) and (group_name not in only_groups):
            continue
        if group_name not in group_sets:
            group_sets[group_name] = set()
        group_sets[group_name] |= col_proteins
    return group_sets
