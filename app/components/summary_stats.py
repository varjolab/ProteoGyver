from pandas import DataFrame, concat


def get_count_data(data_table: DataFrame, contaminant_list: list = None) -> DataFrame:
    """Returns non-na count per column."""
    data: DataFrame
    data = data_table.\
        notna().sum().\
        to_frame(name='Protein count')
    data.index.name = 'Sample name'
    if contaminant_list is not None:
        contaminants = data_table[data_table.index.isin(
            contaminant_list)].notna().sum()
        data['Protein count'] = data['Protein count'] - contaminants
        data['Is contaminant'] = False
        cont_data: DataFrame = contaminants.to_frame(name='Protein count')
        cont_data['Is contaminant'] = True
        data = concat([cont_data, data]).reset_index().rename(
            columns={'index': 'Sample name'})
        data.index = data['Sample name']
        data = data.drop(columns='Sample name')
    return data


def get_coverage_data(data_table: DataFrame) -> DataFrame:
    """Returns coverage dataframe."""
    return DataFrame(
        data_table.notna()
        .astype(int)
        .sum(axis=1)
        .value_counts(), columns=['Identified in # samples']
    )


def get_na_data(data_table: DataFrame) -> DataFrame:
    """Returns na count per column."""
    data: DataFrame = ((data_table.
                        isna().sum() / data_table.shape[0]) * 100).\
        to_frame(name='Missing value %')
    data.index.name = 'Sample name'
    return data


def get_sum_data(data_table) -> DataFrame:
    data: DataFrame = data_table.sum().\
        to_frame(name='Value sum')
    data.index.name = 'Sample name'
    return data


def get_mean_data(data_table) -> DataFrame:
    data: DataFrame = data_table.mean().\
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


def get_common_data(data_table: DataFrame, rev_sample_groups: dict) -> dict:
    group_sets: dict = {}
    for column in data_table.columns:
        col_proteins: set = set(data_table[[column]].dropna().index.values)
        group_name: str = rev_sample_groups[column]
        if group_name not in group_sets:
            group_sets[group_name] = set()
        group_sets[group_name] |= col_proteins
    return group_sets
