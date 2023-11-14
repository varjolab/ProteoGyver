import json
from pandas import DataFrame
from pandas import read_json as pd_read_json
from dash import html
from components.figures import bar_graph, comparative_violin_plot, commonality_graph, reproducibility_graph
from components import summary_stats
from components.figures.figure_legends import QC_LEGENDS as legends
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def count_plot(pandas_json: str, replicate_colors: dict, contaminant_list: list, defaults: dict, title: str = None) -> tuple:
    """Generates a bar plot of given data"""
    start_time: datetime = datetime.now()
    logger.debug(f'count_plot - started: {start_time}')
    count_data: DataFrame = summary_stats.get_count_data(
        pd_read_json(pandas_json, orient='split'),
        contaminant_list
    )

    logger.debug(f'count_plot - summary stats calculated: {datetime.now()}')
    color_col: list = []
    for index, row in count_data.iterrows():
        if row['Is contaminant']:
            color_col.append(replicate_colors['contaminant']['samples'][index])
        else:
            color_col.append(
                replicate_colors['non-contaminant']['samples'][index])
    count_data['Color'] = color_col
    logger.debug(
        f'count_plot - color_added: {datetime.now() }')

    graph_div: html.Div = html.Div(
        id='qc-count-div',
        children=[
            html.H4(id='qc-heading-proteins_per_sample',
                    children='Proteins per sample'),
            bar_graph.make_graph(
                'qc-count-plot',
                defaults,
                count_data,
                title, color_discrete_map=True
            ),
            legends['count-plot']
        ]
    )
    logger.debug(
        f'count_plot - graph drawn: {datetime.now() }')
    return (graph_div, count_data.to_json(orient='split'))


def coverage_plot(pandas_json: str, defaults: dict, title: str = None) -> tuple:
    logger.debug(f'coverage - started: {datetime.now()}')
    coverage_data: DataFrame = summary_stats.get_coverage_data(
        pd_read_json(pandas_json, orient='split'))

    logger.debug(
        f'coverage - summary stats calculated: {datetime.now() }')
    graph_div: html.Div = html.Div(
        id='qc-coverage-div',
        children=[
            html.H4(id='qc-heading-id_coverage',
                    children='Protein identification coverage'),
            bar_graph.make_graph('qc-coverage-plot', defaults,
                                 coverage_data, title, color=False, sort_x=False, x_label='Protein identified in N samples'),
            legends['coverage-plot']
        ]
    )
    logger.debug(
        f'coverage - graph drawn: {datetime.now() }')
    return (graph_div, coverage_data.to_json(orient='split')
            )


def reproducibility_plot(pandas_json: str, sample_groups: dict, table_type: str, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.debug(f'reproducibility_plot - started: {start_time}')
    data_table: DataFrame = pd_read_json(pandas_json, orient='split')
    repro_data: dict = reproducibility_graph.get_reproducibility_dataframe(
        data_table, sample_groups)

    logger.debug(
        f'reproducibility_plot - summary stats calculated: {datetime.now()}')

    graph_div: html.Div = html.Div(
        id='qc-reproducibility-div',
        children=[
            html.H4(id='qc-heading-reproducibility',
                    children='Sample reproducibility'),
            reproducibility_graph.make_graph(
                'qc-reproducibility-plot', defaults, repro_data, title, table_type),
            legends['reproducibility-plot']
        ]
    )
    logger.debug(
        f'reproducibility_plot - graph drawn: {datetime.now() }')
    return (graph_div,  json.dumps(repro_data))


def missing_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.debug(f'missing_plot - started: {start_time}')
    na_data: DataFrame = summary_stats.get_na_data(
        pd_read_json(pandas_json, orient='split'))
    na_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in na_data.index.values
    ]

    logger.debug(
        f'missing_plot - summary stats calculated: {datetime.now() }')
    graph_div: html.Div = html.Div(
        id='qc-missing-div',
        children=[
            html.H4(id='qc-heading-missing_values',
                    children='Missing values per sample'),
            bar_graph.make_graph(
                'qc-missing-plot',
                defaults,
                na_data,
                title, color_discrete_map=True
            ),
            legends['missing_values-plot']
        ]
    )
    logger.debug(
        f'missing_plot - graph drawn: {datetime.now() }')
    return (graph_div, na_data.to_json(orient='split'))


def sum_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.debug(f'sum_plot - started: {start_time}')
    sum_data: DataFrame = summary_stats.get_sum_data(
        pd_read_json(pandas_json, orient='split'))
    sum_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in sum_data.index.values
    ]

    logger.debug(
        f'sum_plot - summary stats calculated: {datetime.now() }')
    graph_div: html.Div = html.Div(
        id='qc-sum-div',
        children=[
            html.H4(id='qc-heading-value_sum',
                    children='Sum of values per sample'),
            bar_graph.make_graph(
                'qc-sum-plot',
                defaults,
                sum_data,
                title, color_discrete_map=True
            ),
            legends['value_sum-plot']
        ]
    )
    logger.debug(
        f'sum_plot - graph drawn: {datetime.now() }')
    return (graph_div, sum_data.to_json(orient='split'))


def mean_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.debug(f'mean_plot - started: {start_time}')
    if title is None:
        title = 'Value mean per sample'
    mean_data: DataFrame = summary_stats.get_mean_data(
        pd_read_json(pandas_json, orient='split'))
    mean_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in mean_data.index.values
    ]

    logger.debug(
        f'mean_plot - summary stats calculated: {datetime.now() }')
    graph_div: html.Div = html.Div(
        id='qc-mean-div',
        children=[
            html.H4(id='qc-heading-value_mean', children='Value mean'),
            bar_graph.make_graph(
                'qc-mean-plot',
                defaults,
                mean_data,
                title, color_discrete_map=True
            ),
            legends['value_mean-plot']
        ]
    )
    logger.debug(
        f'mean_plot - graph drawn: {datetime.now() }')
    return (graph_div, mean_data.to_json(orient='split'))


def distribution_plot(pandas_json: str, replicate_colors: dict, sample_groups: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.debug(f'distribution_plot - started: {start_time}')
    names: list
    comparative_data: list
    names, comparative_data = summary_stats.get_comparative_data(
        pd_read_json(pandas_json, orient='split'),
        sample_groups
    )

    logger.debug(
        f'distribution_plot - summary stats calculated: {datetime.now() }')
    graph_div: html.Div = html.Div(
        id='qc-distribution-div',
        children=[
            html.H4(id='qc-heading-value_dist',
                    children='Value distribution per sample'),
            comparative_violin_plot.make_graph(
                'qc-value_distribution-plot',
                comparative_data,
                defaults,
                names=names,
                title=title,
                replicate_colors=replicate_colors
            ),
            legends['value_dist-plot']
        ]
    )
    logger.debug(
        f'distribution_plot - graph drawn: {datetime.now() }')
    return (graph_div, pandas_json)


def commonality_plot(pandas_json: str, rev_sample_groups: dict, defaults: dict) -> tuple:
    start_time: datetime = datetime.now()
    logger.debug(f'commonality_plot - started: {start_time}')
    common_data: dict = summary_stats.get_common_data(
        pd_read_json(pandas_json, orient='split'),
        rev_sample_groups
    )
    logger.debug(
        f'commonality_plot - summary stats calculated: {datetime.now() }')
    graph, image_str = commonality_graph.make_graph(
        common_data, 'qc-commonality-plot', defaults)
    graph_div: html.Div = html.Div(
        id='qc-supervenn-div',
        children=[
            html.H4(id='qc-heading-shared_id',
                    children='Shared identifications'),
            graph,
            legends['shared_id-plot']
        ])
    common_data = {gk: list(gs) for gk, gs in common_data.items()}
    logger.debug(
        f'commonality_plot - graph drawn: {datetime.now() }')
    return (graph_div, json.dumps(common_data), image_str)
