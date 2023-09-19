import pandas as pd
import numpy as np
from statsmodels.stats import multitest
from scipy.stats import ttest_ind
from typing import Any



def differential(data_table: pd.DataFrame, sample_groups: dict, comparisons: list, data_is_log2_transformed: bool = True, adj_p_thr: float = 0.01, fc_thr:float = 1.0) -> pd.DataFrame:
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
        p_value: float = data_table.apply(lambda x: ttest_ind(x[sample_columns], x[control_columns])[1], axis=1)

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
        result['Name'] = data_table.index.values
        result['Sample'] = sample
        result['Control'] = control
        result['Significant'] = ((result['p_value_adj']<adj_p_thr) & (result['fold_change'].abs() > fc_thr))
        result.sort_values(by='Significant',ascending=True,inplace=True)
        #result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
        sig_data.append(result)
    return pd.concat(sig_data).reset_index().drop(columns='index')[
        ['Sample',
         'Control',
         'Name',
         'Significant',
         'fold_change',
         'p_value',
         'p_value_adj',
         'p_value_adj_neg_log10',
         'sample_mean_value',
         'control_mean_value'
         ]]