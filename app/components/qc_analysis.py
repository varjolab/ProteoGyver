"""
QC analysis figure builders and containers.

Generates standard QC visualizations (counts, coverage, missing values,
distribution, reproducibility, commonality, TIC) and their wrappers for
use in the UI.
"""
import json
from pandas import DataFrame, Series
from pandas import read_json as pd_read_json
from dash import html
import dash_bootstrap_components as dbc
from components.figures import bar_graph, comparative_plot, commonality_graph, reproducibility_graph, color_tools
from components import quick_stats, db_functions
from components.figures.figure_legends import QC_LEGENDS as legends
from components.ui_components import checklist
from datetime import datetime
from io import StringIO
from dash.dcc import Graph, Dropdown, Loading
import logging
logger = logging.getLogger(__name__)


def count_plot(pandas_json: str, replicate_colors: dict, contaminant_list: list, defaults: dict, title: str = None) -> tuple:
    """Generate protein count bar plot per sample.

    :param pandas_json: Data table in JSON split format.
    :param replicate_colors: Color mappings for samples and contaminants.
    :param contaminant_list: Optional list of contaminants to exclude.
    :param defaults: Figure defaults and component config.
    :param title: Optional plot title.
    :returns: Tuple of (graph div, count data as JSON split).
    """
    start_time: datetime = datetime.now()
    logger.info(f'count_plot - started: {start_time}')
    count_data: DataFrame = quick_stats.get_count_data(
        pd_read_json(StringIO(pandas_json),orient='split'),
        contaminant_list
    )

    logger.info(f'count_plot - summary stats calculated: {datetime.now()}')

    color_col: list = []
    for index, row in count_data.iterrows():
        if row['Is contaminant']:
            color_col.append(replicate_colors['contaminant']['samples'][index])
        else:
            color_col.append(
                replicate_colors['non-contaminant']['samples'][index])
    count_data['Color'] = color_col
    logger.info(
        f'count_plot - color_added: {datetime.now() }')

    figtitle = 'Proteins per sample'
    graph_div: html.Div = html.Div(
        id='qc-count-div',
        children=[
            html.H4(id='qc-heading-proteins_per_sample',
                    children=figtitle),
            bar_graph.make_graph(
                'qc-count-plot',
                defaults,
                figtitle,
                count_data,
                title, color_discrete_map=True,
                hide_legend=False
            ),
            legends['count-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'count_plot - graph drawn: {datetime.now() }')
    return (graph_div, count_data.to_json(orient='split'))

def common_proteins(data_table: str, db_file: str, figure_defaults: dict, additional_groups: dict = None, id_str: str = 'qc') -> tuple:
    """Summarize common proteins by class and sample.

    :param data_table: Input matrix in JSON split format.
    :param db_file: Path to SQLite database file.
    :param figure_defaults: Figure defaults and component config.
    :param additional_groups: Optional mapping of group name to protein list to include.
    :param id_str: ID prefix for generated components.
    :returns: Tuple of (graph div, plot data as JSON split).
    """
    table: DataFrame = pd_read_json(StringIO(data_table),orient='split')
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
    figtitle = f'Common proteins in data ({id_str})'
    return (
        html.Div(
            id=f'{id_str}-common-proteins-plot',
            children=[
                html.H4(id=f'{id_str}-common-proteins-header',
                        children=figtitle),
                bar_graph.make_graph(
                    f'{id_str}-common-proteins-graph',
                    figure_defaults,
                    figtitle,
                    plot_frame,
                    '', color_col='Protein class',y_name='ValueSum', x_name='Sample name'
                ),
                legends['common-protein-plot'],
            ],
            style={
                'overflowX': 'auto',
                'whiteSpace': 'nowrap'
            }
        ),
        plot_frame.to_json(orient='split')
    )


def parse_tic_data(expdesign_json: str, replicate_colors: dict, db_file: str,defaults: dict) -> tuple:
    """Prepare TIC/BPC/MSn trace bundles for plotting.

    :param expdesign_json: Experimental design table in JSON split format.
    :param replicate_colors: Mapping of sample names to colors.
    :param db_file: Path to SQLite database file.
    :param defaults: Figure defaults and component config.
    :returns: Tuple of (graph div scaffold, trace dictionary).
    """
    expdesign = pd_read_json(StringIO(expdesign_json),orient='split')
    expdesign['color'] = [replicate_colors['samples'][rep_name] for rep_name in expdesign['Sample name']]
    expdesign['Sample name nopath'] = [sn.split('/')[-1].split('\\')[-1] for sn in expdesign['Sample name'].values]
    with db_functions.create_connection(db_file, mode='rw') as db_conn:# Need rw connection for this one to be able to do this
        run_data = db_functions.get_from_table_match_with_priority(
            db_conn,
            list(expdesign['Sample name nopath'].values),
            'ms_runs',
            ['file_name', 'file_name_clean', 'sample_id', 'sample_name' ],
            case_insensitive = True,
            return_cols = ['internal_run_id']
        )
    int_run_ids = []
    int_id_to_sample = {}
    for _,row in expdesign.iterrows():
        sample = row['Sample name nopath']
        if run_data[sample]:
            int_run_ids.append(run_data[sample]['internal_run_id'])
            int_id_to_sample[run_data[sample]['internal_run_id']] = sample
    if len(int_run_ids) == 0:
        return (html.Div(), {})
    with db_functions.create_connection(db_file, mode='ro') as db_conn:# No need for rw anymore
        graph_data = db_functions.get_from_table_by_list_criteria(db_conn, 'ms_plots', 'internal_run_id', int_run_ids)
    tracetypes = ['TIC','MSn','BPC']
    tic_dic = {}
    for trace_type in tracetypes:
        tracelist = []
        max_x: float = 1.0
        max_y: float = 1.0
        for index,row in graph_data.iterrows():
            sample_row = expdesign[expdesign['Sample name nopath'] == int_id_to_sample[row['internal_run_id']]].iloc[0]
            trace = json.loads(row[f'{trace_type}_trace'])
            max_x = max(max_x, row[f'{trace_type}_maxtime'])
            max_y = max(max_y, row[f'{trace_type}_max_intensity'])
            trace['line'] = {'color': sample_row['color'], 'width': 1}
            tracelist.append(trace)
        tic_dic[trace_type] = {
            'traces': tracelist,
            'max_x': max_x,
            'max_y': max_y
        }
    dlname = 'Chromatogram'
    config = defaults['config'].copy()
    config['toImageButtonOptions']['filename'] = dlname
    graph_div = html.Div(
        id = 'qc-tic-fic',
        children = [
                html.H4(id='qc-heading-tic-graph',
                        children='Sample run TICs'),
                Graph(id='qc-tic-plot', config=config),
                legends['tic'],
                Dropdown(id='qc-tic-dropdown',options=tracetypes, value='TIC')
        ]
    )
    return (graph_div, tic_dic)

def coverage_plot(pandas_json: str, defaults: dict, title: str = None) -> tuple:
    """Create coverage bar plot (proteins identified in N samples).

    :param pandas_json: Data table in JSON split format.
    :param defaults: Figure defaults and component config.
    :param title: Optional plot title.
    :returns: Tuple of (graph div, coverage data as JSON split).
    """
    logger.info(f'coverage - started: {datetime.now()}')
    coverage_data: DataFrame = quick_stats.get_coverage_data(pd_read_json(StringIO(pandas_json),orient='split'))
    logger.info(
        f'coverage - summary stats calculated: {datetime.now() }')
    figtitle = 'Protein identification coverage'
    graph_div: html.Div = html.Div(
        id='qc-coverage-div',
        children=[
            html.H4(id='qc-heading-id_coverage',
                    children='Protein identification coverage'),
            bar_graph.make_graph('qc-coverage-plot', defaults, figtitle,
                                 coverage_data, title, color=False, sort_x=False, x_label='Protein identified in N samples'),
            legends['coverage-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'coverage - graph drawn: {datetime.now() }')
    return (graph_div, coverage_data.to_json(orient='split')
            )


def reproducibility_plot(pandas_json: str, sample_groups: dict, table_type: str, defaults: dict, title: str = None) -> tuple:
    """Plot per-replicate deviations from group mean.

    :param pandas_json: Data table in JSON split format.
    :param sample_groups: Mapping group -> list of columns.
    :param table_type: Label for axis title.
    :param defaults: Figure defaults and component config.
    :param title: Optional title.
    :returns: Tuple of (graph div, reproducibility data JSON).
    """
    start_time: datetime = datetime.now()
    logger.info(f'reproducibility_plot - started: {start_time}')
    data_table: DataFrame = pd_read_json(StringIO(pandas_json),orient='split')
    repro_data: dict = reproducibility_graph.get_reproducibility_dataframe(
        data_table, sample_groups)

    logger.info(
        f'reproducibility_plot - summary stats calculated: {datetime.now()}')

    dlname = 'Sample reproducibility'
    graph_div: html.Div = html.Div(
        id='qc-reproducibility-div',
        children=[
            html.H4(id='qc-heading-reproducibility',
                    children=dlname),
            reproducibility_graph.make_graph(
                'qc-reproducibility-plot', defaults, repro_data, title, table_type, dlname),
            legends['reproducibility-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'reproducibility_plot - graph drawn: {datetime.now() }')
    return (graph_div,  json.dumps(repro_data))


def missing_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    """Plot missing value percentage per sample.

    :param pandas_json: Data table in JSON split format.
    :param replicate_colors: Mapping for sample colors.
    :param defaults: Figure defaults and component config.
    :param title: Optional title.
    :returns: Tuple of (graph div, NA data JSON).
    """
    start_time: datetime = datetime.now()
    logger.info(f'missing_plot - started: {start_time}')
    na_data: DataFrame = quick_stats.get_na_data(
        pd_read_json(StringIO(pandas_json),orient='split'))
    na_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in na_data.index.values
    ]

    logger.info(
        f'missing_plot - summary stats calculated: {datetime.now() }')
    figtitle = 'Missing values per sample'
    graph_div: html.Div = html.Div(
        id='qc-missing-div',
        children=[
            html.H4(id='qc-heading-missing_values',
                    children=figtitle),
            bar_graph.make_graph(
                'qc-missing-plot',
                defaults,
                figtitle,
                na_data,
                title, color_discrete_map=True
            ),
            legends['missing_values-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'missing_plot - graph drawn: {datetime.now() }')
    return (graph_div, na_data.to_json(orient='split'))


def sum_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    """Plot sum of values per sample.

    :param pandas_json: Data table in JSON split format.
    :param replicate_colors: Mapping for sample colors.
    :param defaults: Figure defaults and component config.
    :param title: Optional title.
    :returns: Tuple of (graph div, sum data JSON).
    """
    start_time: datetime = datetime.now()
    logger.info(f'sum_plot - started: {start_time}')
    sum_data: DataFrame = quick_stats.get_sum_data(
        pd_read_json(StringIO(pandas_json),orient='split'))
    sum_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in sum_data.index.values
    ]

    logger.info(
        f'sum_plot - summary stats calculated: {datetime.now() }')
    figtitle = 'Sum of values per sample'
    graph_div: html.Div = html.Div(
        id='qc-sum-div',
        children=[
            html.H4(id='qc-heading-value_sum',
                    children=figtitle),
            bar_graph.make_graph(
                'qc-sum-plot',
                defaults,
                figtitle,
                sum_data,
                title, color_discrete_map=True
            ),
            legends['value_sum-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'sum_plot - graph drawn: {datetime.now() }')
    return (graph_div, sum_data.to_json(orient='split'))


def mean_plot(pandas_json: str, replicate_colors: dict, defaults: dict, title: str = None) -> tuple:
    """Plot mean of values per sample.

    :param pandas_json: Data table in JSON split format.
    :param replicate_colors: Mapping for sample colors.
    :param defaults: Figure defaults and component config.
    :param title: Optional title.
    :returns: Tuple of (graph div, mean data JSON).
    """
    start_time: datetime = datetime.now()
    logger.info(f'mean_plot - started: {start_time}')
    if title is None:
        title = 'Value mean per sample'
    mean_data: DataFrame = quick_stats.get_mean_data(
        pd_read_json(StringIO(pandas_json),orient='split'))
    mean_data['Color'] = [
        replicate_colors['samples'][rep_name] for rep_name in mean_data.index.values
    ]

    logger.info(
        f'mean_plot - summary stats calculated: {datetime.now() }')
    figtitle = 'Value mean'
    graph_div: html.Div = html.Div(
        id='qc-mean-div',
        children=[
            html.H4(id='qc-heading-value_mean', children=figtitle),
            bar_graph.make_graph(
                'qc-mean-plot',
                defaults,
                figtitle, 
                mean_data,
                title, color_discrete_map=True
            ),
            legends['value_mean-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'mean_plot - graph drawn: {datetime.now() }')
    return (graph_div, mean_data.to_json(orient='split'))


def distribution_plot(pandas_json: str, replicate_colors: dict, sample_groups: dict, defaults: dict, title: str = None) -> tuple:
    """Plot value distributions per group as box plots.

    :param pandas_json: Data table in JSON split format.
    :param replicate_colors: Mapping for group colors.
    :param sample_groups: Mapping sample -> group.
    :param defaults: Figure defaults and component config.
    :param title: Optional title.
    :returns: Tuple of (graph div, original JSON string).
    """
    start_time: datetime = datetime.now()
    logger.info(f'distribution_plot - started: {start_time}')
    names: list
    comparative_data: list
    names, comparative_data = quick_stats.get_comparative_data(
        pd_read_json(StringIO(pandas_json),orient='split'),
        sample_groups
    )

    logger.info(
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
                'Value distribution per sample',
                names=names,
                title=title,
                replicate_colors=replicate_colors,
                plot_type='box'
            ),
            legends['value_dist-plot']
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    logger.info(
        f'distribution_plot - graph drawn: {datetime.now() }')
    return (graph_div, pandas_json)


def commonality_plot(pandas_json: str, rev_sample_groups: dict, defaults: dict, only_groups: list = None) -> tuple:
    """Plot commonality across groups using heatmap or Supervenn.

    :param pandas_json: Data table in JSON split format.
    :param rev_sample_groups: Mapping sample -> group.
    :param defaults: Figure defaults and component config.
    :param only_groups: Optional subset of group names to include.
    :returns: Tuple of (graph div, common proteins string, optional base64 PDF string).
    """
    start_time: datetime = datetime.now()
    logger.info(f'commonality_plot - started: {start_time}')
    if only_groups is not None:
        if only_groups == 'all':
            only_groups = None
        elif len(only_groups) == 1:
            only_groups = None
    common_data: dict = quick_stats.get_common_data(
        pd_read_json(StringIO(pandas_json),orient='split'),
        rev_sample_groups,
        only_groups = only_groups
    )
    logger.info(
        f'commonality_plot - summary stats calculated: {datetime.now() }')
    graph, image_str = commonality_graph.make_graph(
        common_data, 'qc-commonality-plot', defaults)
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
        ],
        style={
            'overflowX': 'auto',
            'whiteSpace': 'nowrap'
        }
    )
    common_data = {gk: list(gs) for gk, gs in common_data.items()}
    logger.info(
        f'commonality_plot - graph drawn: {datetime.now() }')
    com = {}
    for sg, sgprots in common_data.items():
        for p in sgprots:
            if p not in com: com[p] = []
            com[p].append(sg)
    common_data = {}
    for p, sets in com.items():
        sk: str = ','.join(sorted(sets))
        if sk not in common_data:
            common_data[sk] = set()
        common_data[sk].add(p)
    common_str:str = ''
    for group, nset in common_data.items():
        common_str += f'Group: {group.replace(";", ", ")} :: {len(nset)} protein groups\n{",".join(sorted(list(nset)))}\n----------\n'

    return (graph_area, common_str, image_str)

def generate_commonality_container(sample_groups):
    """Build the selection controls and display container for commonality.

    :param sample_groups: List of group names.
    :returns: Bootstrap ``Row`` with controls and graph area.
    """
    return dbc.Row(
        [
            dbc.Col(
            checklist(
                label='qc-commonality-select-visible-sample-groups',
                id_only=True,
                options=sample_groups,
                default_choice=sample_groups,
                clean_id = False,
                prefix_list = [html.H4('Select visible sample groups', style={'padding': '75px 0px 0px 0px'})]
            ), width=1),
            dbc.Col(
                [
                    Loading(html.Div(id = 'qc-commonality-graph-div'))
                ],
                width = 11
            )
        ]
    )