import numpy as np
from pandas import DataFrame, Series, isna, concat
import qnorm
from math import ceil
from components.tools import R_tools
from scipy.stats import median_abs_deviation
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

def hierarchical_clustering(df, cluster='both', method='ward', fillval: float = 0.0):
    """Perform hierarchical clustering on a DataFrame.

    :param df: DataFrame with numerical values.
    :param cluster: One of ``'rows'``, ``'columns'``, or ``'both'``.
    :param method: Linkage method for clustering (e.g., ``'ward'``).
    :param fillval: Value used to fill NaNs before distance computation.
    :returns: Reordered DataFrame according to hierarchical clustering.
    :raises ValueError: If ``cluster`` is not one of the allowed values.
    """
    
    if cluster not in ['rows', 'columns', 'both']:
        raise ValueError("Parameter 'cluster' must be one of 'rows', 'columns', or 'both'.")

    if cluster == 'rows' or cluster == 'both':
        row_linkage = linkage(pdist(df.fillna(fillval), metric='euclidean'), method=method)
        row_order = leaves_list(row_linkage)
        df = df.iloc[row_order, :]
        
    if cluster == 'columns' or cluster == 'both':
        col_linkage = linkage(pdist(df.T.fillna(fillval), metric='euclidean'), method=method)
        col_order = leaves_list(col_linkage)
        df = df.iloc[:, col_order]     
    return df


def filter_missing(data_table: DataFrame, sample_groups: dict, filter_type: str, threshold_percentage: int = 60) -> DataFrame:
    """Filter rows with excessive missing values.

    Keeps a row if it meets the threshold either per group (``sample-group``)
    or across the whole table (``sample-set``).

    :param data_table: Input DataFrame.
    :param sample_groups: Mapping group -> list of column names.
    :param filter_type: ``'sample-group'`` or ``'sample-set'``.
    :param threshold_percentage: Minimum non-NA percentage required.
    :returns: Filtered copy of ``data_table``.
    """
    threshold: float = float(threshold_percentage)/100
    keeps: list = []
    for _, row in data_table.iterrows():
        keep: bool = False
        if filter_type == 'sample-group':
            for _, sample_columns in sample_groups.items():
                keep = keep | (row[sample_columns].notna().sum()
                               >= ceil(threshold*len(sample_columns)))
                if keep:
                    break
        elif filter_type == 'sample-set':
            keep = row.notna().sum() >= ceil(threshold*len(data_table.columns))
        keeps.append(keep)
    return data_table[keeps].copy()


def ranked_dist(main_df, supplemental_df):
    """Rank supplemental columns by summed distance to main columns.

    :param main_df: DataFrame providing reference columns.
    :param supplemental_df: DataFrame to compare against.
    :returns: List of [column_name, distance_sum] sorted ascending by distance.
    """
    filtered_main_df: DataFrame = main_df[main_df.index.isin(
        supplemental_df.index.values)]
    filtered_main_df.sort_index(inplace=True)
    supplemental_df.sort_index(inplace=True)

    dist_sums: list = []
    for cc in supplemental_df.columns:
        dist_sums.append([
            cc,
            sum([
                np.linalg.norm(filtered_main_df[c].fillna(0)-supplemental_df[cc].fillna(0)) for c in filtered_main_df.columns
            ])
        ])
    return sorted(dist_sums, key=lambda x: x[1])


def ranked_dist_n_per_run(main_df, supplemental_df, per_run):
    """Select top-N closest supplemental columns per main column.

    :param main_df: DataFrame providing reference columns.
    :param supplemental_df: DataFrame to compare against.
    :param per_run: Number of closest supplemental columns to take per main column.
    :returns: Sorted unique list of chosen supplemental column names.
    """
    filtered_main_df: DataFrame = main_df[main_df.index.isin(
        supplemental_df.index.values)]
    filtered_main_df.sort_index(inplace=True)
    supplemental_df.sort_index(inplace=True)

    chosen_runs: list = []
    for column in filtered_main_df:
        per_run_ranking: list = sorted([
            [c, np.linalg.norm(filtered_main_df[column].fillna(0), supplemental_df[c].fillna(0))] for c in supplemental_df.columns
            ], key=lambda x: x[1])
        chosen_runs.extend([s[0] for s in per_run_ranking[:per_run]])
    return sorted(list(set(chosen_runs)))

def count_per_sample(data_table: DataFrame, rev_sample_groups: dict) -> Series:
    """Count non-NA values per sample for given sample list.

    :param data_table: Input DataFrame.
    :param rev_sample_groups: Mapping sample -> group; keys define sample order.
    :returns: Series indexed by sample name with non-NA counts.
    :raises ValueError: If inputs are empty.
    """
    if not rev_sample_groups:
        raise ValueError("rev_sample_groups is empty")
    if data_table.empty:
        raise ValueError("data_table is empty")
    index: list = list(rev_sample_groups.keys())
    retser: Series = Series(
        index=index,
        data=[data_table[i].notna().sum() for i in index]
    )
    return retser


def do_pca(data_df: DataFrame, rev_sample_groups: dict, n_components) -> tuple:
    """Compute PCA of samples and return labeled components and DataFrame.

    :param data_df: DataFrame with features in rows and samples in columns.
    :param rev_sample_groups: Mapping sample -> group for labeling.
    :param n_components: Number of components to compute (>=2).
    :returns: Tuple ``(pc1_label, pc2_label, pca_result_df)``.
    """
    data_df: DataFrame = data_df.T
    pca: PCA = PCA(n_components=n_components)
    pca_result: np.ndarray = pca.fit_transform(data_df)

    pc1: float
    pc2: float
    pc1, pc2 = pca.explained_variance_ratio_
    pc1 = int(pc1*100)
    pc2 = int(pc2*100)
    pc1 = f'PC1 ({pc1}%)'
    pc2 = f'PC2 ({pc2}%)'

    data_df[pc1] = pca_result[:, 0]
    data_df[pc2] = pca_result[:, 1]
    data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
    data_df['Sample name'] = data_df.index

    return (pc1, pc2, data_df)


def median_normalize(data_frame: DataFrame) -> DataFrame:
    """Median-normalize a log2-transformed DataFrame.

    :param data_frame: DataFrame with samples as columns (log2-transformed).
    :returns: Median-normalized DataFrame.
    """
    medians: Series = data_frame.median(axis=0, skipna=True)
    median_of_medians: float = medians.median(skipna=True)
    normalized_df = data_frame.subtract(medians - median_of_medians, axis=1)
    return normalized_df


def quantile_normalize(dataframe: DataFrame) -> DataFrame:
    """Quantile-normalize a DataFrame.

    :param dataframe: DataFrame to normalize.
    :returns: Quantile-normalized DataFrame.
    """
    return qnorm.quantile_normalize(dataframe, ncpus=8)

def reverse_log2(value):
    """Reverse a log2 transformation.

    :param value: Log2-transformed numeric value.
    :returns: Original (base-2) value.
    """
    return 2**value

def normalize(data_table, normalization_method, errorfile: str, random_seed: int = 13) -> DataFrame:
    """Normalize a DataFrame using a specified method.

    :param data_table: Input DataFrame (log2 for median/quantile; raw for VSN).
    :param normalization_method: One of ``'no_normalization'``, ``'median'``, ``'quantile'``, ``'vsn'``.
    :param errorfile: Path used by VSN routine for diagnostics.
    :param random_seed: Random seed for VSN reproducibility.
    :returns: Normalized DataFrame.
    :raises ValueError: For invalid normalization method.
    """
    if normalization_method.lower() == 'no_normalization':
        return_table = data_table
    elif normalization_method.lower() == 'median':
        return_table = median_normalize(data_table)
    elif normalization_method.lower() == 'quantile':
        return_table = quantile_normalize(data_table)
    elif normalization_method.lower() == 'vsn':
        data_table = data_table.map(reverse_log2)
        return_table = R_tools.vsn(data_table, random_seed, errorfile)
    else:
        raise ValueError(f"Invalid normalization method: {normalization_method}")
    return return_table


def impute(data_table: DataFrame, errorfile: str, method: str, random_seed: int, rev_sample_groups: dict) -> DataFrame:
    """Impute missing values using the specified method.

    :param data_table: Input DataFrame.
    :param errorfile: Path used by external methods for diagnostics.
    :param method: One of ``'minprob'``, ``'minvalue'``, ``'gaussian'``, ``'qrilc'``, ``'random_forest'``.
    :param random_seed: Random seed for reproducibility.
    :param rev_sample_groups: Mapping sample -> group (used by some methods).
    :returns: Imputed DataFrame.
    :raises ValueError: For invalid imputation method.
    """
    if method.lower() == 'minprob':
        ret = impute_minprob_df(data_table, random_seed)
    elif method.lower() == 'minvalue':
        ret = impute_minval(data_table)
    elif method.lower() == 'gaussian':
        ret = impute_gaussian(data_table, random_seed)
    elif method.lower() == 'qrilc':
        ret = R_tools.impute_qrilc(data_table, random_seed, errorfile)
    elif method.lower() in ['random_forest', 'random forest']:
        ret = R_tools.impute_random_forest(data_table, random_seed, rev_sample_groups, errorfile)
    else:
        raise ValueError(f"Invalid imputation method: {method}")
    return ret


def impute_minval(dataframe: DataFrame, impute_zero: bool = False) -> DataFrame:
    """Impute missing values with the per-column minimum.

    :param dataframe: Numeric DataFrame with missing values.
    :param impute_zero: If ``True``, treat zeros as missing.
    :returns: DataFrame with imputed values.
    """
    newdf: DataFrame = DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        newcol: Series = dataframe[column]
        if impute_zero:
            newcol = newcol.replace(0, np.nan)
        newcol = newcol.fillna(newcol.min())
        newdf.loc[:, column] = newcol
    return newdf


def impute_gaussian(data_table: DataFrame, random_seed: int, dist_width: float = 0.15, dist_down_shift: float = 2,) -> DataFrame:
    """Impute values by sampling from a shifted/scaled Gaussian.

    Based on Perseus' method.

    :param data_table: Numeric DataFrame with missing values.
    :param random_seed: Random seed for reproducibility.
    :param dist_width: Width as a fraction of column standard deviation.
    :param dist_down_shift: Downward shift in standard deviations.
    :returns: DataFrame with imputed values.
    """
    np.random.seed(random_seed)
    newdf: DataFrame = DataFrame(index=data_table.index)
    for column in data_table.columns:
        newcol: Series = data_table[column]
        stdev: float = newcol.std()
        distribution: np.ndarray = np.random.normal(
            loc=newcol.mean() - (dist_down_shift*stdev),
            scale=dist_width*stdev,
            size=data_table.shape[0]*100
        )
        replace_values: Series = Series(
            index=data_table.index,
            data=np.random.choice(
                a=distribution, size=data_table.shape[0], replace=False)
        )
        newcol = newcol.fillna(replace_values)
        newdf.loc[:, column] = newcol
    return newdf

def impute_minprob(series_to_impute: Series, random_seed: int, scale: float = 1.0,
                   tune_sigma: float = 0.01, impute_zero=True) -> Series:
    """Impute values from a distribution near the lowest non-NA values.

    :param series_to_impute: Series with possible missing values.
    :param random_seed: Random seed for reproducibility.
    :param scale: Scale parameter for ``numpy.random.normal``.
    :param tune_sigma: Fraction of the lowest values to define the distribution.
    :param impute_zero: If ``True``, treat zeros as missing.
    :returns: Series with imputed values.
    """
    np.random.seed(random_seed)
    ser: Series = series_to_impute.sort_values(ascending=True)
    ser = ser[ser > 0].dropna()
    ser = ser[:int(len(ser)*tune_sigma)]

    # implement q value
    distribution: np.ndarray = np.random.normal(
        loc=ser.median(), scale=scale, size=len(series_to_impute*100))

    output_series: Series = series_to_impute.copy()
    for index, value in output_series.items():
        impute_value: bool = False
        if isna(value):
            impute_value = True
        elif (value == 0) and impute_zero:
            impute_value = True
        if impute_value:
            output_series[index] = np.random.choice(distribution)
    return output_series

def impute_minprob_df(dataframe: DataFrame, *args, **kwargs) -> DataFrame:
    """Impute an entire DataFrame using the minprob method.

    :param dataframe: Numeric DataFrame to impute.
    :param args: Positional args forwarded to ``impute_minprob``.
    :param kwargs: Keyword args forwarded to ``impute_minprob``.
    :returns: Imputed DataFrame.
    """
    newdf: DataFrame = DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        newdf.loc[:, column] = impute_minprob(
            dataframe[column], *args, **kwargs)
    return newdf

def compute_zscore(data: DataFrame, test_samples: list, control_samples: list, measure: str ='median', std: int =2):
    """Compute Z-scores of test samples relative to control samples.

    :param data: DataFrame with proteins in index and samples in columns (log2).
    :param test_samples: List of test sample column names.
    :param control_samples: List of control sample column names.
    :param measure: ``'mean'`` or ``'median'`` center for controls.
    :param std: Threshold; values below are set to 0 in the result.
    :returns: DataFrame of Z-scores for test samples.
    """
    control_data = data[control_samples]
    calc_data = data[test_samples]
    
    if measure == 'mean':
        mean = control_data.mean(axis=1)
        std_dev = control_data.std(axis=1)
    elif measure == 'median':
        mean = control_data.median(axis=1)
       # std_dev = control_data.std(axis=1)
        std_dev = median_abs_deviation(control_data, axis=1) * 1.4826  # To approximate standard deviation
        
    z_scores = (calc_data.subtract(mean, axis=0)).div(std_dev, axis=0)
    #z_scores = z_scores.abs()
    z_scores[z_scores < std] = 0
    
    return z_scores

def compute_zscore_based_deviation_from_control(df: DataFrame, sample_groups: dict, control_group: str, top_n: int = 50) -> tuple:
    """Compute group-wise Z-score deviations relative to a control group.

    :param df: DataFrame with proteins in rows and samples in columns (log2).
    :param sample_groups: Mapping group -> list of sample names.
    :param control_group: Name of the control group.
    :param top_n: Number of top proteins to aggregate per group.
    :returns: Tuple of (dict of Z-score summaries, per-protein summary DataFrame, top-N proteins DataFrame).
    """
    results = {}
    all_topn_proteins: set = set()
    for sample_group, sample_columns in sample_groups.items():
        if sample_group == control_group: continue
        z_score_matrix = compute_zscore(df, sample_columns, sample_groups[control_group])
        ranked_proteins = z_score_matrix.mean(axis=1).sort_values(ascending=False)
        top_prots = ranked_proteins.head(top_n)
        z_score_mean = z_score_matrix.mean(axis=0)
        z_score_mean_topn = z_score_matrix.loc[top_prots.index].mean(axis=0)
        all_topn_proteins |= set(top_prots.index.values)
        results[sample_group] = [z_score_matrix, ranked_proteins, top_prots, z_score_mean, z_score_mean_topn]
    all_topn_proteins = sorted(list(all_topn_proteins))
    for sg in results.keys():
        results[sg].append(results[sg][0].loc[all_topn_proteins].mean(axis=0))
    z_score_dfs = dict()
    for i, final_result_key in enumerate(['Z-score mean', f'Z-score top{top_n} mean', f'Z-score top{top_n} from all samplegroups']):
        z_score_dfs[final_result_key] = []
        for sample_group, sg_result in results.items():
            z_score_dfs[final_result_key].append(
                sg_result[3+i]
            )
    z_score_dfs = {key: concat(vals) for key, vals in z_score_dfs.items()}
    result_protein_data = []
    sorted_groups = [ sg for sg in results.keys()]
    final_result_cols_protein = ['Z-score mean', 'Z-score max', 'Z-score max group'] + sorted_groups
    for pi in df.index:
        vals_per_sg = []
        allvals = []
        max_group = ('',0)
        for sample_group in sorted_groups:
            allvals.extend(list(results[sample_group][0].loc[pi].values))
            sgval = results[sample_group][0].loc[pi].mean()
            if abs(sgval) > abs(max_group[1]):
                max_group = (sample_group, sgval)
            vals_per_sg.append(sgval)
        result_protein_data.append([
            sum(allvals)/len(allvals),
            max_group[1],
            max_group[0]
        ] + vals_per_sg)
    protein_df = DataFrame(data=result_protein_data, index = df.index, columns=final_result_cols_protein)
    return (
        z_score_dfs,
        protein_df,
        protein_df.loc[all_topn_proteins]
    )