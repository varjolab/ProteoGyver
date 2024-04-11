
import pandas as pd
import numpy as np
from statsmodels.stats import multitest
from scipy.stats import ttest_ind, ttest_rel, f_oneway
from typing import Any
from components.data_provider import map_protein_info

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


def differential(data_table: pd.DataFrame, sample_groups: dict, comparisons: list, data_is_log2_transformed: bool = True, namemap: dict = None, adj_p_thr: float = 0.01, fc_thr:float = 1.0, test_type: str = 'independent') -> pd.DataFrame:
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
        result['Gene']  = map_protein_info(result.index)
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


