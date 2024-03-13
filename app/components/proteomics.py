import pandas as pd
from dash import html
from dash.dcc import Graph
from components.figures import bar_graph, before_after_plot, comparative_plot, imputation_histogram, scatter, heatmaps, volcano_plot, histogram, cvplot
from components import matrix_functions, summary_stats, stats
from components.figures.figure_legends import PROTEOMICS_LEGENDS as legends
from components.figures.figure_legends import leg_rep
from datetime import datetime
from dash_bootstrap_components import Card, CardBody, Tab, Tabs, Col, Row
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

def pertubation(filtered_and_normalized_data_json: str, sample_groups: dict, control_groups: str, replicate_colors:dict, figure_defaults_bars:dict, figure_defaults_matrix: dict):
    pertu_tabs: list = []
    pertubation_dict: dict = {}
    figdef_for_plots = figure_defaults_matrix.copy()
    figdef_for_plots['height'] = None
    figdef_for_plots['width'] = None
    done_groups = set()
    for control_group in control_groups:
        if control_group in done_groups:
            continue
        done_groups.add(control_group)
        top_n = 50
        results = matrix_functions.compute_zscore_based_deviation_from_control(
           pd.read_json(filtered_and_normalized_data_json,orient='split'),
            sample_groups,
            control_group,
            top_n
        )
        pertubation_dict[control_group] = {
            'All proteins': results[1].to_json(orient='split'),
            f'Top {top_n} proteins': results[2].to_json(orient='split')
        }
        for key, ser in results[0].items():
            pertubation_dict[control_group]['key'] = ser.to_dict()
        bar_graph_col_contents: list = []
        bar_big_legend_done = False
        for key, valser in results[0].items():
            resdf: pd.DataFrame = pd.DataFrame(valser,columns=[key])
            print(sample_groups)
            resdf['Color'] = resdf.index#[sample_groups[i] for i in resdf.index]
            pertu_bar_graph: Graph = bar_graph.make_graph(
                f'proteomics-pertubation-{control_group}-{key}',
                figdef_for_plots,
                resdf,
                f'Sample Z-score mean vs {control_group}',
                x_name = None,
                y_name = key,
                x_label = 'Sample name'
            )
            if not bar_big_legend_done:
                lkey = 'pertubation-bar'
                bar_big_legend_done = True
            else:
                lkey = 'pertubation-bar-2'
            bar_leg = leg_rep(
                leg_rep(
                    legends[lkey],
                    'CONTROLSTRING',
                    control_group
                ),
                'BARVALS',
                key
            )
            bar_graph_col_contents.extend(
                [
                    html.H5(f'{key} based pertubation against {control_group}'),
                    pertu_bar_graph,
                    bar_leg
                ]
            )
        matrix_data: pd.DataFrame = results[2][results[2].columns[3:]]
        matrix_col_contents: list = [
            html.H5(f'Per-sample group mean Z-score based pertubation against {control_group}'),
            heatmaps.make_heatmap_graph(
                matrix_data,
                f'proteomics-pertubation-matrix-vs{control_group}',
                f'Z-score group mean vs {control_group}',
                figdef_for_plots,
                cmap = 'dense'
            ),
            #heatmap here
            leg_rep(
                leg_rep(
                    legends['pertubation-matrix'],
                    'CONTROLSTRING',
                    control_group
                ),
                'TOPN',
                f'{top_n}'
            ),
            html.H5(f'Mean Z-score based pertubation against {control_group} top {top_n} proteins'),
            bar_graph.make_graph(
                f'proteomics-pertubation-{control_group}-protein-z-score',
                figdef_for_plots,
                results[2][['Z-score mean']],
                f'Protein max Z-score mean vs {control_group}',
                x_name = None,
                y_name = 'Z-score mean',
                x_label = 'Protein name',
                color = None
            ),
            leg_rep(
                leg_rep(
                    legends['pertubation-bar-2'],
                    'CONTROLSTRING',
                    control_group
                ),
                'BARVALS',
                f'mean of {top_n} proteins across all sample groups.'
            )
        ]

        pertu_tabs.append(
            Tab(
                label = control_group,
                children=[
                    Card(
                        CardBody(
                            [
                                Row(
                                    [
                                        Col(bar_graph_col_contents,width=6),
                                        Col(matrix_col_contents, width=6)
                                    ]
                                )
                            ]
                        )
                    )
                ]
            )
        )
    return (
        html.Div(
            id='proteomics-pertubation',
            children = [
                html.H4(id='proteomics-pertubation-header',children='Z-score based pertubation vs control group(s)'),
                Tabs(
                    id='proteomics-pertubation-tabs',
                    children = pertu_tabs,
                    style = {'width': '98%'}
                )
            ]
        ),
        pertubation_dict
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

def perc_cvplot(data_json: str, sample_groups: dict, replicate_colors: dict, defaults: dict) -> Graph:
    graph, data = cvplot.make_graph(pd.read_json(data_json, orient='split'),sample_groups, replicate_colors, defaults, 'proteomics-cv-plot')
    return (
        html.Div(
            id = 'proteomics-cv-div',
            children = [
                html.H4(id='proteomics-cv-header', children='Coefficients of variation'),
                graph,
                legends['cv']
            ]
        ),
        data
    )

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


def pca(imputed_data_json: str, sample_groups_rev: dict, defaults: dict, replicate_colors: dict) -> tuple:

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


def volcano_plots(imputed_data_json: str, sample_groups: dict, comparisons: list, fc_thr: float, p_thr: float, defaults: dict) -> tuple:

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
    return (
        [
            html.H3(id='proteomics-volcano-header', children='Volcano plots'),
            graphs_div
        ],
        significant_data.to_json(orient='split')
    )
