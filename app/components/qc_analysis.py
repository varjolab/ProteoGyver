import json
from pandas import DataFrame, Series
from pandas import read_json as pd_read_json
from dash import html
import dash_bootstrap_components as dbc
from components.figures import bar_graph, comparative_plot, commonality_graph, reproducibility_graph
from components import summary_stats
from components.figures.figure_legends import QC_LEGENDS as legends
from components.ui_components import checklist
from components.tooltips import use_svenn_tooltip
from datetime import datetime
from dash.dcc import Graph, Dropdown, Loading
import logging
from components import db_functions
logger = logging.getLogger(__name__)


def count_plot(pandas_json: str, replicate_colors: dict, contaminant_list: list, defaults: dict, title: str = None) -> tuple:
    """Generates a bar plot of given data"""
    start_time: datetime = datetime.now()
    logger.warning(f'count_plot - started: {start_time}')
    count_data: DataFrame = summary_stats.get_count_data(
        pd_read_json(pandas_json, orient='split'),
        contaminant_list
    )

    logger.warning(f'count_plot - summary stats calculated: {datetime.now()}')
    color_col: list = []
    for index, row in count_data.iterrows():
        if row['Is contaminant']:
            color_col.append(replicate_colors['contaminant']['samples'][index])
        else:
            color_col.append(
                replicate_colors['non-contaminant']['samples'][index])
    count_data['Color'] = color_col
    logger.warning(
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
    logger.warning(
        f'count_plot - graph drawn: {datetime.now() }')
    return (graph_div, count_data.to_json(orient='split'))

def common_proteins(data_table: str, db_file: str, figure_defaults: dict, additional_groups: dict = None) -> tuple:
    table: DataFrame = pd_read_json(data_table,orient='split')
    db_conn = db_functions.create_connection(db_file)
    common_proteins: DataFrame = db_functions.get_from_table_by_list_criteria(db_conn, 'common_proteins','uniprot_id',list(table.index))
    common_proteins.index = common_proteins['uniprot_id']
    if additional_groups is None:
        additional_groups = {}
    additional_proteins = {}
    for k, plist in additional_groups.items():
        plist = [p for p in plist if p not in common_proteins.index.values]
        for p in plist:
            if p not in additional_proteins:
                additional_proteins[p] = []
            additional_proteins[p].append(k)
    additional_groups = {}
    for protid, groups in additional_proteins.items():
        gk = ', '.join(groups)
        if gk not in additional_groups: additional_groups[gk] = set()
        additional_groups[gk].add(protid)
    
    plot_headers: list = ['Sample name','Protein class','Proteins', 'ValueSum','Count']
    plot_data: list = []
    for c in table.columns:
        col_data: Series = table[c]
        col_data = col_data[col_data.notna()]
        com_for_col: DataFrame = common_proteins.loc[common_proteins.index.isin(col_data.index)]
        for pclass in com_for_col['protein_type'].unique():
            class_prots = com_for_col[com_for_col['protein_type']==pclass].index.values
            plot_data.append([
                c, pclass, ', '.join(class_prots), col_data.loc[class_prots].sum(), len(class_prots)
            ])
        remaining_proteins: Series = col_data[~col_data.index.isin(com_for_col.index)]
        for groupname, group_prots in additional_groups.items():
            in_both = group_prots & set(remaining_proteins.index.values)
            if len(in_both) > 0:
                plot_data.append([
                    c, groupname, ', '.join(in_both), col_data.loc[list(in_both)].sum(), len(in_both) 
                ])

        remaining_proteins = remaining_proteins[~remaining_proteins.index.isin(additional_proteins)]
        plot_data.append([
            c, 'None', ','.join(remaining_proteins.index.values), remaining_proteins.sum(), remaining_proteins.shape[0]
        ])
    plot_frame: DataFrame = DataFrame(data=plot_data,columns=plot_headers)
    plot_frame.sort_values(by='Protein class',ascending=False)
    
    return (
        html.Div(
            id='qc-common-proteins-plot',
            children=[
                html.H4(id='qc-common-proteins-header',
                        children='Common proteins in data'),
                bar_graph.make_graph(
                    'qc-common-proteins-graph',
                    figure_defaults,
                    plot_frame,
                    '', color_col='Protein class',y_name='ValueSum', x_name='Sample name'
                ),
                legends['common-protein-plot'],
            ]
        ),
        plot_frame.to_json(orient='split')
    )



def parse_tic_data(expdesign_json: str, replicate_colors: dict, db_file: str,defaults: dict) -> tuple:
    expdesign = pd_read_json(expdesign_json, orient='split')
    expdesign['color'] = [replicate_colors['samples'][rep_name] for rep_name in expdesign['Sample name']]
    expdesign['Sampleid'] = [s.split('_')[0] for s in expdesign['Sample name']]
    db_conn = db_functions.create_connection(db_file)
    ms_runs = db_functions.get_from_table_by_list_criteria(db_conn, 'ms_runs','run_id',expdesign['Sampleid'].values)
    db_conn.close()
    dtypes: list = ['TIC','MSn_unfiltered']
    tic_dic: dict = {t_type.lower(): {'traces': [] } for t_type in dtypes}
    for trace_type in dtypes:
        trace_type = trace_type.lower()
        max_x: float = 2000.0
        max_y: float = 2000.0
        for _,row in ms_runs.iterrows():
            sample_row = expdesign[expdesign['Sampleid']==row['run_id']].iloc[0]
            trace = json.loads(row[f'{trace_type}_trace'])
            max_x = max(max_x, max(trace['x']))
            max_y = max(max_y, max(trace['y']))
            trace['line'] = {'color': sample_row['color'], 'width': 1}
            tic_dic[trace_type]['traces'].append(trace)
        tic_dic[trace_type]['max_x'] = max_x
        tic_dic[trace_type]['max_y'] = max_y
    if ms_runs.shape[0] == 0:
        return (html.Div(),{})
    else:
        graph_div: html.Div = html.Div(
            id = 'qc-tic-div',
            children = [
                html.H4(id='qc-heading-tic-graph',
                        children='Sample run TICs'),
                Graph(id='qc-tic-plot', config=defaults['config']),
                legends['tic'],
                Dropdown(id='qc-tic-dropdown',options=dtypes, value='TIC')
            ]
        )
        return (graph_div, tic_dic)

def coverage_plot(pandas_json: str, defaults: dict, title: str = None) -> tuple:
    logger.warning(f'coverage - started: {datetime.now()}')
    coverage_data: DataFrame = summary_stats.get_coverage_data(
        pd_read_json(pandas_json, orient='split'))

    logger.warning(
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
    logger.warning(
        f'coverage - graph drawn: {datetime.now() }')
    return (graph_div, coverage_data.to_json(orient='split')
            )


def reproducibility_plot(pandas_json: str, sample_groups: dict, table_type: str, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.warning(f'reproducibility_plot - started: {start_time}')
    data_table: DataFrame = pd_read_json(pandas_json, orient='split')
    repro_data: dict = reproducibility_graph.get_reproducibility_dataframe(
        data_table, sample_groups)

    logger.warning(
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
    logger.warning(
        f'reproducibility_plot - graph drawn: {datetime.now() }')
    return (graph_div,  json.dumps(repro_data))


def missing_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.warning(f'missing_plot - started: {start_time}')
    na_data: DataFrame = summary_stats.get_na_data(
        pd_read_json(pandas_json, orient='split'))
    na_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in na_data.index.values
    ]

    logger.warning(
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
    logger.warning(
        f'missing_plot - graph drawn: {datetime.now() }')
    return (graph_div, na_data.to_json(orient='split'))


def sum_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.warning(f'sum_plot - started: {start_time}')
    sum_data: DataFrame = summary_stats.get_sum_data(
        pd_read_json(pandas_json, orient='split'))
    sum_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in sum_data.index.values
    ]

    logger.warning(
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
    logger.warning(
        f'sum_plot - graph drawn: {datetime.now() }')
    return (graph_div, sum_data.to_json(orient='split'))


def mean_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.warning(f'mean_plot - started: {start_time}')
    if title is None:
        title = 'Value mean per sample'
    mean_data: DataFrame = summary_stats.get_mean_data(
        pd_read_json(pandas_json, orient='split'))
    mean_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in mean_data.index.values
    ]

    logger.warning(
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
    logger.warning(
        f'mean_plot - graph drawn: {datetime.now() }')
    return (graph_div, mean_data.to_json(orient='split'))


def distribution_plot(pandas_json: str, replicate_colors: dict, sample_groups: dict, defaults: dict, title: str = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.warning(f'distribution_plot - started: {start_time}')
    names: list
    comparative_data: list
    names, comparative_data = summary_stats.get_comparative_data(
        pd_read_json(pandas_json, orient='split'),
        sample_groups
    )

    logger.warning(
        f'distribution_plot - summary stats calculated: {datetime.now() }')
    graph_div: html.Div = html.Div(
        id='qc-distribution-div',
        children=[
            html.H4(id='qc-heading-value_dist',
                    children='Value distribution per sample'),
            comparative_plot.make_graph(
                'qc-value_distribution-plot',
                comparative_data,
                defaults,
                names=names,
                title=title,
                replicate_colors=replicate_colors,
                plot_type='box'
            ),
            legends['value_dist-plot']
        ]
    )
    logger.warning(
        f'distribution_plot - graph drawn: {datetime.now() }')
    return (graph_div, pandas_json)


def commonality_plot(pandas_json: str, rev_sample_groups: dict, defaults: dict, force_svenn: bool, only_groups: list = None) -> tuple:
    start_time: datetime = datetime.now()
    logger.warning(f'commonality_plot - started: {start_time}')
    common_data: dict = summary_stats.get_common_data(
        pd_read_json(pandas_json, orient='split'),
        rev_sample_groups,
        only_groups = only_groups
    )
    logger.warning(
        f'commonality_plot - summary stats calculated: {datetime.now() }')
    graph, image_str = commonality_graph.make_graph(
        common_data, 'qc-commonality-plot', force_svenn, defaults)
    if image_str == '':
        legend = legends['shared_id-plot-hm']
    else:
        legend = legends['shared_id-plot-sv']
    graph_area: html.Div = html.Div(       
        id='qc-supervenn-div',
        children=[
            html.H4(id='qc-heading-shared_id',
                    children='Shared identifications'),
            graph,
            legend,
        ]
    )
    common_data = {gk: list(gs) for gk, gs in common_data.items()}
    logger.warning(
        f'commonality_plot - graph drawn: {datetime.now() }')
    return (graph_area, json.dumps(common_data), image_str)

def generate_commonality_container(sample_groups):
    def_for_force: list = []
    if len(sample_groups) <= 6:
        def_for_force.append('Use supervenn')
    return dbc.Row(
        [
            dbc.Col(
            checklist(
                label='qc-commonality-select-visible-sample-groups',
                id_only=True,
                options=sample_groups,
                default_choice=sample_groups,
                clean_id = False,
                prefix_list = [html.H5('Select visible sample groups', style={'padding': '75px 0px 0px 0px'})],
                postfix_list=checklist(
                    label='toggle-additional-supervenn-options',
                    options=['Use supervenn'],
                    id_only=True,
                    default_choice = def_for_force,
                    clean_id = False,
                    postfix_list = [
                        dbc.Button('Update plot',id='qc-commonality-plot-update-plot-button'),
                        use_svenn_tooltip()
                    ]
                )
            ), width=2),
            dbc.Col(
                [
                    Loading(html.Div(id = 'qc-commonality-graph-div'))
                ],
                width = 10
            )
        ]
    )