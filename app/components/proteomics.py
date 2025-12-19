"""
Proteomics analysis figure builders.

Implements plots and computations for missing value filtering,
normalization, distributions, imputation, PCA, clustermap, CV plots, and
volcano analyses used by the proteomics workflow.
"""
import pandas as pd
from io import StringIO
from dash import html
from dash.dcc import Graph
from components.figures import bar_graph, before_after_plot, comparative_plot, imputation_histogram, scatter, heatmaps, volcano_plot, histogram, cvplot
from components import matrix_functions, quick_stats
from components.figures.figure_legends import PROTEOMICS_LEGENDS as legends
from components.figures.figure_legends import leg_rep
from datetime import datetime
from dash_bootstrap_components import Card, CardBody, Tab, Tabs, Col, Row
import logging
logger = logging.getLogger(__name__)

def na_filter(input_data_dict, filtering_percentage, figure_defaults, title: str = None, filter_type: str = 'sample-group') -> tuple:
    """Apply NA filtering and visualize before/after counts.

    :param input_data_dict: Data dictionary containing intensity tables and sample groups.
    :param filtering_percentage: Threshold percentage for presence filtering.
    :param figure_defaults: Figure defaults and component config.
    :param title: Optional plot title.
    :param filter_type: ``'sample-group'`` or ``'sample-set'``.
    :returns: Tuple of (graph div, filtered data JSON).
    """

    logger.info(f'nafilter - start: {datetime.now()}')
    data_table: pd.DataFrame = pd.read_json(
        StringIO(input_data_dict['data tables']['intensity']),
        orient='split'
    )
    original_counts: pd.Series = matrix_functions.count_per_sample(
        data_table, input_data_dict['sample groups']['rev'])
    logger.info(
        f'nafilter - counts per sample: {datetime.now()}')

    filtered_data: pd.DataFrame = matrix_functions.filter_missing(
        data_table,
        input_data_dict['sample groups']['norm'],
        filter_type,
        filtering_percentage
    )
    logger.info(
        f'nafilter - filtering done: {datetime.now()}')

    filtered_counts: pd.Series = matrix_functions.count_per_sample(
        filtered_data, input_data_dict['sample groups']['rev'])
    figure_legend: html.P = legends['na_filter']
    figure_legend.children = figure_legend.children.replace(
        'FILTERPERC', f'{filtering_percentage}')
    logger.info(
        f'nafilter - only plot left: {datetime.now()}')
    figtitle = 'Missing value filtering'
    return (
        html.Div(
            id='proteomics-na-filter-plot-div',
            children=[
                html.H4(id='proteomics-na-filter-header',
                        children=figtitle),
                before_after_plot.make_graph(
                    figure_defaults, original_counts, filtered_counts, 'proteomics-na-filter-plot', figtitle, title=title),
                figure_legend
            ],
            style={
                'overflowX': 'auto',
                'whiteSpace': 'nowrap'
            }
        ),
        filtered_data.to_json(orient='split')
    )

def normalization(filtered_data_json: str, normalization_option: str, defaults: dict, errorfile: str, title: str = None) -> tuple:
    """Normalize filtered data and show distributions before/after.

    :param filtered_data_json: Filtered data in JSON split format.
    :param normalization_option: Normalization method name.
    :param defaults: Figure defaults and component config.
    :param errorfile: Path for logging/diagnostics from normalization.
    :param title: Optional plot title.
    :returns: Tuple of (graph div, normalized table JSON).
    """

    logger.info(f'normalization - start: {datetime.now()}')

    data_table: pd.DataFrame = pd.read_json(StringIO(filtered_data_json),orient='split')
    normalized_table: pd.DataFrame = matrix_functions.normalize(
        data_table, normalization_option, errorfile)
    logger.info(
        f'normalization - normalized: {datetime.now()}')

    sample_groups_rev: dict = {
        column_name: 'Before normalization' for column_name in data_table.columns
    }
    sample_groups_rev.update({
        f'Normalized {column_name}': 'After normalization' for column_name in data_table.columns
    })
    norm_cols_rename = {column_name: f'Normalized {column_name}' for column_name in data_table.columns}
    data_table = pd.concat([data_table, normalized_table.rename(columns=norm_cols_rename)], axis=1)

    logger.info(
        f'normalization - normalized applied: {datetime.now()}')

    names: list
    comparative_data: dict
    names, comparative_data = quick_stats.get_comparative_data(
        data_table, sample_groups_rev
    )
    logger.info(
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
        'Normalization',
        names=names,
        title=title,
        replicate_colors=plot_colors,
        plot_type='box'
    )
    logger.info(
        f'normalization - graph done, writing: {datetime.now()}')
    logger.info(
        f'normalization - graph done, returning: {datetime.now()}')
    return (
        html.Div(
            id='proteomics-normalization-plot-div',
            children=[
                html.H4(id='proteomics-normalization-header',
                        children='Normalization'),
                plot,
                legends['normalization']
            ],
            style={
                'overflowX': 'auto',
                'whiteSpace': 'nowrap'
            }
        ),
        normalized_table.to_json(orient='split')
    )

def missing_values_in_other_samples(filtered_data_json,defaults) -> html.Div:
    """Histogram comparing intensities of proteins with/without missing values.

    :param filtered_data_json: Filtered data in JSON split format.
    :param defaults: Figure defaults and component config.
    :returns: Div containing the histogram and legend.
    """
    data_table: pd.DataFrame = pd.read_json(StringIO(filtered_data_json),orient='split')
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
    dlname = 'Intensity of proteins with missing values in other samples'
    config = defaults['config'].copy()
    config['toImageButtonOptions'] = config['toImageButtonOptions'].copy()
    config['toImageButtonOptions']['filename'] = dlname
    return html.Div(
        id='proteomics-missing-in-other-samples-graph-div',
        children = [
            html.H4(
                id='proteomics-missing-in-other-samples-header',
                children=dlname
            ),
            Graph(
                config=config,
                id='proteomics-missing-in-other-samples-graph',
                figure=figure
            ),
            legends['missing-in-other-samples']
        ]
    )

def perc_cvplot(raw_int_data: str, na_filtered_data: str, sample_groups: dict, replicate_colors: dict, defaults: dict) -> tuple:
    """Compute and plot coefficient of variation per group.

    :param raw_int_data: Raw intensity data in JSON split format.
    :param na_filtered_data: NA-filtered intensity data in JSON split format.
    :param sample_groups: Mapping group -> list of columns.
    :param replicate_colors: Mapping for group colors.
    :param defaults: Figure defaults and component config.
    :returns: Tuple of (graph div, stats JSON).
    """
    raw_int_df: pd.DataFrame = pd.read_json(StringIO(raw_int_data), orient='split')
    na_filtered_df: pd.DataFrame = pd.read_json(StringIO(na_filtered_data), orient='split')
    # Drop rows that are no longer present in filtered data
    raw_int_df.drop(index=list(set(raw_int_df.index)-set(na_filtered_df.index)),inplace=True)
    dlname = 'Coefficients of variation'
    graph, data = cvplot.make_graph(raw_int_df,sample_groups, replicate_colors, defaults, 'proteomics-cv-plot', dlname)
    return (
        html.Div(
            id = 'proteomics-cv-div',
            children = [
                html.H4(id='proteomics-cv-header', children=dlname),
                graph,
                legends['cv']
            ],
            style={
                'overflowX': 'auto',
                'whiteSpace': 'nowrap'
            }
        ),
        data
    )

def imputation(filtered_data_json, imputation_option, defaults, errorfile:str, sample_groups_rev: dict, title: str = None) -> tuple:
    """Impute missing values and render distribution comparison.

    :param filtered_data_json: Filtered data in JSON split format.
    :param imputation_option: Imputation method name.
    :param defaults: Figure defaults and component config.
    :param errorfile: Path for logging/diagnostics from imputation.
    :param sample_groups_rev: Mapping sample -> group (used by imputation).
    :param title: Optional plot title.
    :returns: Tuple of (graph div, imputed table JSON).
    """

    logger.info(f'imputation - start: {datetime.now()}')

    data_table: pd.DataFrame = pd.read_json(StringIO(filtered_data_json),orient='split')
    imputed_table: pd.DataFrame = matrix_functions.impute(
        data_table, errorfile, imputation_option, random_seed=13, rev_sample_groups=sample_groups_rev)
    logger.info(
        f'imputation - imputed, only plot left: {datetime.now()}')
    dlname = 'Imputation'
    return (
        html.Div(
            id='proteomics-imputation-plot-div',
            children=[
                html.H4(id='proteomics-imputation-header',
                        children=dlname),
                imputation_histogram.make_graph(
                    data_table,
                    imputed_table,
                    defaults,
                    dlname,
                    id_name='proteomics-imputation-plot',
                    title=title,

                ),
                legends['imputation']
            ]
        ),
        imputed_table.to_json(orient='split')
    )


def pca(imputed_data_json: str, sample_groups_rev: dict, defaults: dict, replicate_colors: dict) -> tuple:
    """Compute PCA and plot the first two components.

    :param imputed_data_json: Imputed data in JSON split format.
    :param sample_groups_rev: Mapping sample -> group.
    :param defaults: Figure defaults and component config.
    :param replicate_colors: Mapping for group colors.
    :returns: Tuple of (graph div, PCA result JSON).
    """

    logger.info(f'PCA - start: {datetime.now()}')
    data_table: pd.DataFrame = pd.read_json(StringIO(imputed_data_json),orient='split')
    pc1: str
    pc2: str
    pca_result: pd.DataFrame
    # Compute PCA of the data
    pc1, pc2, pca_result = matrix_functions.do_pca(
        data_table, sample_groups_rev, n_components=2)
    pca_result.sort_values(by=pc1, ascending=True, inplace=True)
    logger.info(
        f'PCA - done, only plotting left: {datetime.now()}')
    pca_result['Sample group color'] = [replicate_colors['sample groups'][grp] for grp in pca_result['Sample group']]
    return (
        html.Div(
            id='proteomics-pca-plot-div',
            children=[
                html.H4(id='proteomics-pca-header', children='PCA'),
                scatter.make_graph(
                    'proteomics-pca-plot',
                    defaults,
                    'PCA',
                    pca_result,
                    pc1,
                    pc2,
                    'Sample group color',
                    'Sample group',
                    hover_data=['Sample group', 'Sample name']
                ),
                legends['pca']
            ]
        ),
        pca_result.to_json(orient='split')
    )


def clustermap(imputed_data_json: str, defaults: dict) -> tuple:
    """Draws a correltion clustergram figure from the given data_table.

    :param imputed_data_json: Imputed data in JSON split format.
    :param defaults: Figure defaults and component config.
    :returns: Tuple of (graph div, correlation matrix JSON).
    """
    corrdata: pd.DataFrame = pd.read_json(
        StringIO(imputed_data_json), orient='split').corr()
    logger.info(
        f'clustermap - only plotting left: {datetime.now()}')
    dlname = 'Sample correlation clustering'
    config = defaults['config'].copy()
    config['toImageButtonOptions'] = config['toImageButtonOptions'].copy()
    config['toImageButtonOptions']['filename'] = dlname
    return (
        html.Div(
            id='proteomics-clustermap-plot-div',
            children=[
                html.H4(id='proteomics-clustermap-header',
                        children=dlname),
                Graph(
                    id='proteomics-clustermap-plot',
                    config=config,
                    figure=heatmaps.draw_clustergram(
                        corrdata, defaults
                    )
                ),
                legends['clustermap']
            ]
        ),
        corrdata.to_json(orient='split')
    )


def differential_abundance(imputed_data_json: str, sample_groups: dict, comparisons: list, fc_thr: float, p_thr: float, defaults: dict, test_type:str = 'independent', db_file_path: str = None) -> tuple:
    """Run differential analysis and generate volcano plots.

    :param imputed_data_json: Imputed data in JSON split format.
    :param sample_groups: Mapping group -> list of columns.
    :param comparisons: List of ``(sample, control)`` group pairs.
    :param fc_thr: Absolute log2 fold change threshold.
    :param p_thr: Adjusted p-value threshold.
    :param defaults: Figure defaults and component config.
    :param test_type: ``'independent'`` or ``'paired'``.
    :param db_file_path: Optional DB path for gene mapping.
    :returns: Tuple of (components, significant data JSON).
    """

    logger.info(f'volcano - start: {datetime.now()}')
    data: pd.DataFrame = pd.read_json(StringIO(imputed_data_json),orient='split')
    significant_data: pd.DataFrame = quick_stats.differential(
        data, sample_groups, comparisons, fc_thr=fc_thr, adj_p_thr=p_thr, test_type = test_type, db_file_path = db_file_path)
    if significant_data is None:
        return ('', None)
    logger.info(
        f'volcano - significants calculated: {datetime.now() }')

    graphs_div: html.Div = volcano_plot.generate_graphs(
        significant_data, defaults, fc_thr, p_thr, 'proteomics')
    logger.info(
        f'volcano - volcanoes generated: {datetime.now()}')
    return (
        [
            html.H3(id='proteomics-volcano-header', children='Differential abundance'),
            graphs_div
        ],
        significant_data.to_json(orient='split')
    )
