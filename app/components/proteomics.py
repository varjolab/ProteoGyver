import pandas as pd
from dash import html
from dash.dcc import Graph
from components.figures import before_after_plot, comparative_plot, imputation_histogram, scatter, heatmaps, volcano_plot, histogram
from components import matrix_functions, summary_stats, stats
from components.figures.figure_legends import PROTEOMICS_LEGENDS as legends
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def na_filter(input_data_dict, filtering_percentage, figure_defaults, title: str = None) -> tuple:

    logger.warning(f'nafilter - start: {datetime.now()}')
    data_table: pd.DataFrame = pd.read_json(
        input_data_dict['data tables']['intensity'],
        orient='split'
    )
    original_counts: pd.DataFrame = matrix_functions.count_per_sample(
        data_table, input_data_dict['sample groups']['rev'])
    logger.warning(
        f'nafilter - counts per sample: {datetime.now()}')

    filtered_data: pd.DataFrame = matrix_functions.filter_missing(
        data_table,
        input_data_dict['sample groups']['norm'],
        filtering_percentage,
    )
    logger.warning(
        f'nafilter - filtering done: {datetime.now()}')

    filtered_counts: pd.DataFrame = matrix_functions.count_per_sample(
        filtered_data, input_data_dict['sample groups']['rev'])
    figure_legend: html.P = legends['na_filter']
    figure_legend.children = figure_legend.children.replace(
        'FILTERPERC', f'{filtering_percentage}')
    logger.warning(
        f'nafilter - only plot left: {datetime.now()}')
    return (
        html.Div(
            id='proteomics-na-filter-plot-div',
            children=[
                html.H4(id='proteomics-na-filter-header',
                        children='Missing value filtering'),
                before_after_plot.make_graph(
                    figure_defaults, original_counts, filtered_counts, 'proteomics-na-filter-plot', title=title),
                figure_legend
            ]
        ),
        filtered_data.to_json(orient='split')
    )


def normalization(filtered_data_json: str, normalization_option: str, defaults: dict, errorfile: str, title: str = None) -> tuple:

    logger.warning(f'normalization - start: {datetime.now()}')

    data_table: pd.DataFrame = pd.read_json(filtered_data_json, orient='split')
    normalized_table: pd.DataFrame = matrix_functions.normalize(
        data_table, normalization_option, errorfile)
    logger.warning(
        f'normalization - normalized: {datetime.now()}')

    sample_groups_rev: dict = {
        column_name: 'Before normalization' for column_name in data_table.columns
    }
    for column_name in normalized_table.columns:
        new_column_name: str = f'Normalized {column_name}'
        data_table[new_column_name] = normalized_table[column_name]
        sample_groups_rev[new_column_name] = 'After normalization'

    logger.warning(
        f'normalization - normalized applied: {datetime.now()}')

    names: list
    comparative_data: dict
    names, comparative_data = summary_stats.get_comparative_data(
        data_table, sample_groups_rev
    )
    logger.warning(
        f'normalization - comparative data generated, only plotting left: {datetime.now()}')

    plot_colors: dict = {
        'sample groups': {
            'Before normalization': 'rgb(235,100,50)',
            'After normalization': 'rgb(50,100,235)'
        }
    }
    plot: Graph = comparative_plot.make_graph(
        'proteomics-normalization-plot',
        comparative_data,
        defaults,
        names=names,
        title=title,
        replicate_colors=plot_colors,
        plot_type='box'
    )
    logger.warning(
        f'normalization - graph done, writing: {datetime.now()}')
    logger.warning(
        f'normalization - graph done, returning: {datetime.now()}')
    return (
        html.Div(
            id='proteomics-normalization-plot-div',
            children=[
                html.H4(id='proteomics-normalization-header',
                        children='Normalization'),
                plot,
                legends['normalization']
            ]
        ),
        normalized_table.to_json(orient='split')
    )

def missing_values_in_other_samples(filtered_data_json,defaults) -> html.Div:
    data_table: pd.DataFrame = pd.read_json(filtered_data_json, orient='split')
    missing_series: pd.Series = pd.Series(data_table.loc[data_table.isna().sum(axis=1)>0].values.flatten())
    valid_series: pd.Series = pd.Series(data_table.loc[data_table.isna().sum(axis=1)==0].values.flatten())
    missing_series = missing_series[missing_series.notna()]
    missing_data: pd.DataFrame = pd.DataFrame({'Protein intensity in all samples': missing_series})
    valid_data: pd.DataFrame = pd.DataFrame({'Protein intensity in all samples': valid_series})
    plot_data = pd.concat([missing_data, valid_data],ignore_index=True)
    plot_data['Protein'] = [
        'has missing values' if i < missing_data.shape[0] else 'present in all samples' \
        for i in range(plot_data.shape[0])
    ]
    plot_data.sort_values(by='Protein',ascending=False,inplace=True)
    figure = histogram.make_figure(
        plot_data,
        x_column = 'Protein intensity in all samples',
        title = '',
        color='Protein',
        defaults = defaults
    )
    figure.update_layout(
        barmode='overlay'
    )
    figure.update_traces(opacity=0.75)
    return html.Div(
        id='proteomics-missing-in-other-samples-graph-div',
        children = [
            html.H4(
                id='proteomics-missing-in-other-samples-header',
                children='Intensity of proteins with missing values in other samples'
            ),
            Graph(
                config=defaults['config'],
                id='proteomics-missing-in-other-samples-graph',
                figure=figure
            ),
            legends['missing-in-other-samples']
        ]
    )
    return None

def imputation(filtered_data_json, imputation_option, defaults, errorfile:str, title: str = None) -> tuple:

    logger.warning(f'imputation - start: {datetime.now()}')

    data_table: pd.DataFrame = pd.read_json(filtered_data_json, orient='split')
    imputed_table: pd.DataFrame = matrix_functions.impute(
        data_table, errorfile, imputation_option)
    logger.warning(
        f'imputation - imputed, only plot left: {datetime.now()}')
    return (
        html.Div(
            id='proteomics-imputation-plot-div',
            children=[
                html.H4(id='proteomics-imputation-header',
                        children='Imputation'),
                imputation_histogram.make_graph(
                    data_table,
                    imputed_table,
                    defaults,
                    id_name='proteomics-imputation-plot',
                    title=title,

                ),
                legends['imputation']
            ]
        ),
        imputed_table.to_json(orient='split')
    )


def pca(imputed_data_json: dict, sample_groups_rev: dict, defaults: dict) -> tuple:

    logger.warning(f'PCA - start: {datetime.now()}')
    data_table: pd.DataFrame = pd.read_json(imputed_data_json, orient='split')
    pc1: str
    pc2: str
    pca_result: pd.DataFrame
    # Compute PCA of the data
    pc1, pc2, pca_result = matrix_functions.do_pca(
        data_table, sample_groups_rev, n_components=2)
    pca_result.sort_values(by=pc1, ascending=True, inplace=True)
    logger.warning(
        f'PCA - done, only plotting left: {datetime.now()}')

    return (
        html.Div(
            id='proteomics-pca-plot-div',
            children=[
                html.H4(id='proteomics-pca-header', children='PCA'),
                scatter.make_graph(
                    'proteomics-pca-plot',
                    defaults,
                    pca_result,
                    pc1,
                    pc2,
                    'Sample group'
                ),
                legends['pca']
            ]
        ),
        pca_result.to_json(orient='split')
    )


def clustermap(imputed_data_json: dict, defaults: dict) -> tuple:
    """Draws a correltion clustergram figure from the given data_table.

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    id_name: name for the plot. will be used for the id of the returned dcc.Graph object.

    Returns: 
    dcc.Graph containing a dash_bio.Clustergram describing correlation between samples.
    """
    corrdata: pd.DataFrame = pd.read_json(
        imputed_data_json, orient='split').corr()
    logger.warning(
        f'clustermap - only plotting left: {datetime.now()}')
    return (
        html.Div(
            id='proteomics-clustermap-plot-div',
            children=[
                html.H4(id='proteomics-clustermap-header',
                        children='Sample correlation clustering'),
                Graph(
                    id='proteomics-clustermap-plot',
                    config=defaults['config'],
                    figure=heatmaps.draw_clustergram(
                        corrdata, defaults
                    )
                ),
                legends['clustermap']
            ]
        ),
        corrdata.to_json(orient='split')
    )


def volcano_plots(imputed_data_json: dict, sample_groups: dict, comparisons: list, fc_thr: float, p_thr: float, defaults: dict) -> tuple:

    logger.warning(f'volcano - start: {datetime.now()}')
    data: pd.DataFrame = pd.read_json(imputed_data_json, orient='split')
    significant_data: pd.DataFrame = stats.differential(
        data, sample_groups, comparisons, fc_thr=fc_thr, adj_p_thr=p_thr)
    logger.warning(
        f'volcano - significants calculated: {datetime.now() }')

    graphs_div: html.Div = volcano_plot.generate_graphs(
        significant_data, defaults, fc_thr, p_thr, 'proteomics')
    logger.warning(
        f'volcano - volcanoes generated: {datetime.now()}')
    return ([
        html.H3(id='proteomics-volcano-header', children='Volcano plots'),
        graphs_div
    ],
        significant_data.to_json(orient='split')
    )
