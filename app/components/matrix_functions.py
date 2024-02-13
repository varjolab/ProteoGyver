import numpy as np
from pandas import DataFrame, Series, isna
import qnorm
from math import ceil
from components.tools import R_tools
from sklearn.decomposition import PCA


def filter_missing(data_table: DataFrame, sample_groups: dict, threshold: int = 60) -> DataFrame:
    """Discards rows with more than threshold percent of missing values in all sample groups"""
    threshold: float = float(threshold)/100
    keeps: list = []
    for _, row in data_table.iterrows():
        keep: bool = False
        for _, sample_columns in sample_groups.items():
            keep = keep | (row[sample_columns].notna().sum()
                           >= ceil(threshold*len(sample_columns)))
            if keep:
                break
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
    Median-normalizes a dataframe by dividing each column by its median.

    Args:
        df (pandas.DataFrame): The dataframe to median-normalize.
        Each column represents a sample, and each row represents a measurement.

    Returns:
        pandas.DataFrame: The median-normalized dataframe.
    """
    # Calculating the medians prior to looping is about 2-3 times more efficient,
    # than calculating the median of each column inside of the loop.
    medians: Series = data_frame.median(axis=0)
    mean_of_medians: float = medians.mean()
    newdf: DataFrame = DataFrame(index=data_frame.index)
    for col in data_frame.columns:
        newdf[col] = (data_frame[col] / medians[col]) * mean_of_medians
    return newdf


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
        normalization_method = 'None'
    if normalization_method == 'Median':
        return_table = median_normalize(data_table)
    elif normalization_method == 'Quantile':
        return_table = quantile_normalize(data_table)
    elif normalization_method == 'Vsn':
        data_table = data_table.applymap(reverse_log2)
        return_table = R_tools.vsn(data_table, random_seed, errorfile)
    return return_table


def impute(data_table: DataFrame, errorfile: str, method: str = 'QRILC', random_seed: int = 13) -> DataFrame:
    """Imputes missing values into the dataframe with the specified method"""
    ret: DataFrame = data_table
    if method == 'minProb':
        ret = impute_minprob_df(data_table, random_seed)
    elif method == 'minValue':
        ret = impute_minval(data_table)
    elif method == 'gaussian':
        ret = impute_gaussian(data_table, random_seed,errorfile)
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
