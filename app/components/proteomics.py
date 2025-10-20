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
    return (
        html.Div(
            id='proteomics-na-filter-plot-div',
            children=[
                html.H4(id='proteomics-na-filter-header',
                        children='Missing value filtering'),
                before_after_plot.make_graph(
                    figure_defaults, original_counts, filtered_counts, 'proteomics-na-filter-plot', title=title),
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

def perc_cvplot(raw_int_data: str, na_filtered_data: str, sample_groups: dict, replicate_colors: dict, defaults: dict) -> tuple:
    raw_int_df: pd.DataFrame = pd.read_json(StringIO(raw_int_data), orient='split')
    na_filtered_df: pd.DataFrame = pd.read_json(StringIO(na_filtered_data), orient='split')
    # Drop rows that are no longer present in filtered data
    raw_int_df.drop(index=list(set(raw_int_df.index)-set(na_filtered_df.index)),inplace=True)

    graph, data = cvplot.make_graph(raw_int_df,sample_groups, replicate_colors, defaults, 'proteomics-cv-plot')
    return (
        html.Div(
            id = 'proteomics-cv-div',
            children = [
                html.H4(id='proteomics-cv-header', children='Coefficients of variation'),
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

    logger.info(f'imputation - start: {datetime.now()}')

    data_table: pd.DataFrame = pd.read_json(StringIO(filtered_data_json),orient='split')
    imputed_table: pd.DataFrame = matrix_functions.impute(
        data_table, errorfile, imputation_option, random_seed=13, rev_sample_groups=sample_groups_rev)
    logger.info(
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


def pca(imputed_data_json: str, sample_groups_rev: dict, defaults: dict, replicate_colors: dict) -> tuple:

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

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    id_name: name for the plot. will be used for the id of the returned dcc.Graph object.

    Returns: 
    dcc.Graph containing a dash_bio.Clustergram describing correlation between samples.
    """
    corrdata: pd.DataFrame = pd.read_json(
        StringIO(imputed_data_json), orient='split').corr()
    logger.info(
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
                        corrdata, defaults, center_values=False
                    )
                ),
                legends['clustermap']
            ]
        ),
        corrdata.to_json(orient='split')
    )


def differential_abundance(imputed_data_json: str, sample_groups: dict, comparisons: list, fc_thr: float, p_thr: float, defaults: dict, test_type:str = 'independent', db_file_path: str = None) -> tuple:

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
