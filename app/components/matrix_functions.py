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
    """
    Perform hierarchical clustering on a pandas DataFrame.
    
    Parameters:
    - df: pandas DataFrame with numerical values
    - cluster: 'rows', 'columns', or 'both' to specify which dimension to cluster
    - method: Method of linkage for hierarchical clustering. Defaults to 'ward'
    
    Returns:
    - clustered_df: pandas DataFrame reordered according to hierarchical clustering
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
    """Discards rows with more than threshold percent of missing values in all sample groups"""
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
    """Counts non-zero values per sample (sample names from rev_sample_groups.keys()) and returns a series with sample names in index and counts as values."""
    index: list = list(rev_sample_groups.keys())
    retser: Series = Series(
        index=index,
        data=[data_table[i].notna().sum() for i in index]
    )
    return retser


def do_pca(data_df: DataFrame, rev_sample_groups: dict, n_components) -> tuple:
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
    """
    Median-normalizes a dataframe by dividing each column by its median and multiplying by the median of the medians.

    Args:
        df (pandas.DataFrame): The dataframe to median-normalize.
        Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The median-normalized dataframe.
    """
    medians: Series = data_frame.median(axis=0)
    mean_of_medians: float = medians.mean()
    norm_df: DataFrame = DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        norm_df[col] = (data_frame[col] / medians[col]) * mean_of_medians
    return norm_df


def quantile_normalize(dataframe: DataFrame) -> DataFrame:
    """Quantile-normalizes a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to quantile-normalize.
        Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The quantile-normalized dataframe.
    """
    return qnorm.quantile_normalize(dataframe, ncpus=8)

def reverse_log2(value):
    return 2**value

def normalize(data_table, normalization_method, errorfile: str, random_seed: int = 13) -> DataFrame:
    """Normalizes a given dataframe with the wanted method."""
    return_table: DataFrame = data_table
    if not normalization_method:
        normalization_method = 'no_normalization'
    if normalization_method == 'Median':
        return_table = median_normalize(data_table)
    elif normalization_method == 'Quantile':
        return_table = quantile_normalize(data_table)
    elif normalization_method == 'Vsn':
        data_table = data_table.map(reverse_log2)
        return_table = R_tools.vsn(data_table, random_seed, errorfile)
    return return_table


def impute(data_table: DataFrame, errorfile: str, method: str = 'QRILC', random_seed: int = 13) -> DataFrame:
    """Imputes missing values in the dataframe with the specified method"""
    ret: DataFrame = data_table
    if method == 'minProb':
        ret = impute_minprob_df(data_table, random_seed)
    elif method == 'minValue':
        ret = impute_minval(data_table)
    elif method == 'gaussian':
        ret = impute_gaussian(data_table, random_seed)
    elif method == 'QRILC':
        ret = R_tools.impute_qrilc(data_table, random_seed, errorfile)
    return ret


def impute_minval(dataframe: DataFrame, impute_zero: bool = False) -> DataFrame:
    """Impute missing values in dataframe using minval method

    Input dataframe should only have numerical data with missing values.
    Missing values will be replaced by the minimum value of each column.

    Parameters:
    df: pandas dataframe with the missing values. Should not have any text columns
    impute_zero: True, if zero should be considered a missing value
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
    """Impute missing values in dataframe using values from random numbers from normal distribution.

    Based on the method used by Perseus (http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian)

    Parameters:
    data_table: pandas dataframe with the missing values. Should not have any text columns
    dist_width: Gaussian distribution relative to stdev of each column. 
        Value of 0.5 means the width of the distribution is half the standard deviation of the sample column values.
    dist_down_shift: How far downwards the distribution is shifted. By default, 2 standard deviations down.
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
    """Imputes missing values with randomly selected entries from a distribution \
        centered around the lowest non-NA values of the series.

    Arguments:
    series_to_impute: pandas series with possible missing values

    Keyword arguments:
    scale: passed to numpy.random.normal
    tune_sigma: fraction of values from the lowest end of the series to use for \
        generating the distribution
    impute_zero: treat 0 values as missing values and impute new values for them
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
    """imputes whole dataframe with minprob imputation. Dataframe should only have numerical columns

    Parameters:
    df: dataframe to impute
    kwargs: keyword args to pass on to impute_minprob
    """
    newdf: DataFrame = DataFrame(index=dataframe.index)
    for column in dataframe.columns:
        newdf.loc[:, column] = impute_minprob(
            dataframe[column], *args, **kwargs)
    return newdf



def compute_zscore(data: DataFrame, test_samples: list, control_samples: list, measure: str ='median', std: int =2):
    """
    Computes Z-scores for log2 transformed MS intensity data using control samples.
    
    Parameters:
    - data: DataFrame proteins in index and samples in columns.
    - control_samples: List control sample column names.
    - measure: 'mean' or 'median' to specify the central tendency measure.
    - std: Standard deviation threshold for Z-score calculation.
    
    Returns:
    - DataFrame with Z-scores, where values below 'std' are set to 0.
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