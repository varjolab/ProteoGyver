from pandas import DataFrame
from pandas import read_json as pd_read_json
from dash import html
from dash.dcc import Graph
from components.figures import before_after_plot, comparative_violin_plot, imputation_histogram, scatter, heatmaps, volcano_plot
from components import matrix_functions, summary_stats, stats
from components.figures.figure_legends import PROTEOMICS_LEGENDS as legends
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def na_filter(input_data_dict, filtering_percentage, figure_defaults, title: str = None) -> tuple:

    logger.debug(f'nafilter - start: {datetime.now()}')
    data_table: DataFrame = pd_read_json(
        input_data_dict['data tables']['intensity'],
        orient='split'
    )
    original_counts: DataFrame = matrix_functions.count_per_sample(
        data_table, input_data_dict['sample groups']['rev'])
    logger.debug(
        f'nafilter - counts per sample: {datetime.now()}')

    filtered_data: DataFrame = matrix_functions.filter_missing(
        data_table,
        input_data_dict['sample groups']['norm'],
        filtering_percentage,
    )
    logger.debug(
        f'nafilter - filtering done: {datetime.now()}')

    filtered_counts: DataFrame = matrix_functions.count_per_sample(
        filtered_data, input_data_dict['sample groups']['rev'])
    figure_legend: html.P = legends['na_filter']
    figure_legend.children = figure_legend.children.replace(
        'FILTERPERC', f'{filtering_percentage}')
    logger.debug(
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


def normalization(filtered_data_json, normalization_option, defaults, title: str = None) -> tuple:

    logger.debug(f'normalization - start: {datetime.now()}')

    data_table: DataFrame = pd_read_json(filtered_data_json, orient='split')
    normalized_table: DataFrame = matrix_functions.normalize(
        data_table, normalization_option)
    logger.debug(
        f'normalization - normalized: {datetime.now()}')

    sample_groups_rev: dict = {
        column_name: 'Before normalization' for column_name in data_table.columns
    }
    for column_name in normalized_table.columns:
        new_column_name: str = f'Normalized {column_name}'
        data_table[new_column_name] = normalized_table[column_name]
        sample_groups_rev[new_column_name] = 'After normalization'

    logger.debug(
        f'normalization - normalized applied: {datetime.now()}')

    names: list
    comparative_data: dict
    names, comparative_data = summary_stats.get_comparative_data(
        data_table, sample_groups_rev
    )
    logger.debug(
        f'normalization - comparative data generated, only plotting left: {datetime.now()}')

    plot_colors: dict = {
        'sample groups': {
            'Before normalization': 'rgb(235,100,50)',
            'After normalization': 'rgb(50,100,235)'
        }
    }
    plot: Graph = comparative_violin_plot.make_graph(
        'proteomics-normalization-plot',
        comparative_data,
        defaults,
        names=names,
        title=title,
        replicate_colors=plot_colors
    )
    logger.debug(
        f'normalization - graph done, writing: {datetime.now()}')
    plot.figure.write_json('test.json', pretty=True)
    plot.figure.write_html('test.html', config=defaults['config'])
    logger.debug(
        f'normalization - graph done, returning: {datetime.now()}')
    return (
        html.Div(
            id='proteomics-normalization-plot-div',
            children=[
                html.H4(id='proteomics-normalization-header',
                        children='Normalization'),
                plot,
                legends['comparative-violin-plot']
            ]
        ),
        normalized_table.to_json(orient='split')
    )


def imputation(filtered_data_json, imputation_option, defaults, title: str = None) -> tuple:

    logger.debug(f'imputation - start: {datetime.now()}')

    data_table: DataFrame = pd_read_json(filtered_data_json, orient='split')
    imputed_table: DataFrame = matrix_functions.impute(
        data_table, imputation_option)
    logger.debug(
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

    logger.debug(f'PCA - start: {datetime.now()}')
    data_table: DataFrame = pd_read_json(imputed_data_json, orient='split')
    pc1: str
    pc2: str
    pca_result: DataFrame
    # Compute PCA of the data
    pc1, pc2, pca_result = matrix_functions.do_pca(
        data_table, sample_groups_rev, n_components=2)
    pca_result.sort_values(by=pc1, ascending=True, inplace=True)
    logger.debug(
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
    corrdata: DataFrame = pd_read_json(
        imputed_data_json, orient='split').corr()
    logger.debug(
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

    logger.debug(f'volcano - start: {datetime.now()}')
    data: DataFrame = pd_read_json(imputed_data_json, orient='split')
    significant_data: DataFrame = stats.differential(
        data, sample_groups, comparisons, fc_thr=fc_thr, adj_p_thr=p_thr)
    logger.debug(
        f'volcano - significants calculated: {datetime.now() }')

    graphs_div: html.Div = volcano_plot.generate_graphs(
        significant_data, defaults, fc_thr, p_thr, 'proteomics')
    logger.debug(
        f'volcano - volcanoes generated: {datetime.now()}')
    return ([
        html.H3(id='proteomics-volcano-header', children='Volcano plots'),
        graphs_div
    ],
        significant_data.to_json(orient='split')
    )
