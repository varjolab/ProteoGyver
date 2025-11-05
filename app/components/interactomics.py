"""Functions for processing and visualizing protein-protein interaction data.

This module provides functionality for analyzing mass spectrometry-based
interactomics data, including:
- Running and processing SAINT analysis for scoring protein interactions
- Filtering results based on BFDR and CRAPome metrics
- Creating visualizations (networks, heatmaps, PCA plots)
- Performing enrichment analysis
- MS-microscopy analysis for protein localization
- Processing known interaction data

The module integrates with a SQLite database for retrieving reference data
and uses Dash components for creating interactive visualizations.

Typical usage example:
    >>> saint_dict = make_saint_dict(spc_table, sample_groups, controls, proteins)
    >>> saint_output = run_saint(saint_dict, temp_dir, session_id, bait_ids)
    >>> filtered_output = saint_filtering(saint_output, bfdr=0.01, crapome_pct=0.1)
    >>> network_plot = do_network(filtered_output, plot_height=600)

Attributes:
    logger: Logger instance for module-level logging
"""

from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dash import html, dash_table
import pandas as pd
from io import StringIO
from components import db_functions
import numpy as np
import shutil
import os
import tempfile
import sh
import sqlite3
from components.figures import histogram, bar_graph, scatter, heatmaps, network_plot
from components import matrix_functions, db_functions, ms_microscopy
from components.figures.figure_legends import INTERACTOMICS_LEGENDS as legends
from components.figures.figure_legends import enrichment_legend, leg_rep
from components.text_handling import replace_accent_and_special_characters
from components import EnrichmentAdmin as ea
from dash_bootstrap_components import Card, CardBody, Tab, Tabs
from datetime import datetime
import logging
logger = logging.getLogger(__name__)

def count_knowns(saint_output: pd.DataFrame, 
                replicate_colors: Dict[str, Dict[str, Dict[str, str]]]) -> pd.DataFrame:
    """Count known interactions per bait protein.

    :param saint_output: SAINT output with columns including ``Bait`` and ``Known interaction``.
    :param replicate_colors: Mapping with structure ``{'contaminant': {'sample groups': {bait: color}}, 'non-contaminant': {...}}``.
    :returns: DataFrame with columns ``Bait``, ``Known interaction``, ``Prey count``, and ``Color``.
    """
    data: pd.DataFrame = saint_output[['Bait', 'Known interaction']].\
        value_counts().to_frame().reset_index().rename(
            columns={'count': 'Prey count'})
    color_col: list = []
    for _, row in data.iterrows():
        if row['Known interaction']:
            color_col.append(
                replicate_colors['contaminant']['sample groups'][row['Bait']])
        else:
            color_col.append(
                replicate_colors['non-contaminant']['sample groups'][row['Bait']])
    data['Color'] = color_col
    return data


def do_network(saint_output_json: str, 
              plot_height: int) -> Tuple[html.Div, List[Dict[str, Any]], Dict[str, Any]]:
    """Create a Cytoscape network from filtered SAINT output.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param plot_height: Height of the network plot in pixels.
    :returns: Tuple of (plot container Div, cytoscape elements, interactions dict).
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    cyto_elements, interactions = network_plot.get_cytoscape_elements_and_ints(saint_output)
    plot_container = network_plot.get_cytoscape_container(cyto_elements, full_height=plot_height)
    return (plot_container, cyto_elements, interactions)


def network_display_data(
    node_data: dict[str, list[dict]], 
    int_data: dict[str, dict[str, list[str|float]]], 
    table_height: int, 
    datatype: str = 'Cytoscape'
) -> list[html.Label | dash_table.DataTable]:
    """Create a table for network connections.

    :param node_data: Node data; for Cytoscape use ``{'edgesData': [{'source','target'},...]}``; for visdcc use ``{'edges': ['source_-_target', ...]}``.
    :param int_data: Mapping ``source -> target -> [gene_name, avg_spec]``.
    :param table_height: Table height in pixels.
    :param datatype: ``'Cytoscape'`` or ``'visdcc'``.
    :returns: List containing a label and a DataTable with Bait, Prey, PreyGene, AvgSpec.
    """
    ret = [['Bait','Prey', 'PreyGene','AvgSpec']]
    if datatype == 'Cytoscape':
        for e in node_data['edgesData']: 
            ret.append([e['source'], e['target']])
            ret[-1].extend(int_data[e['source']][e['target']])
    elif datatype == 'visdcc':
        for e in node_data['edges']:
            source, target = e.split('_-_')
            ret.append([
                source,
                target,
                int_data[source][target]
            ])

    df = pd.DataFrame(data=ret[1:], columns=ret[0])
    div_contents = [
        html.Label('Selected node connections:'),
        dash_table.DataTable(
            df.to_dict('records'), 
            [{"name": i, "id": i} for i in df.columns],
            fixed_rows={'headers': True},
            style_table={'height': table_height}
        )
    ]
    return div_contents

def known_plot(filtered_saint_input_json: str, 
              db_file: str, 
              rep_colors_with_cont: Dict[str, Dict[str, str]], 
              figure_defaults: Dict[str, Any], 
              isoform_agnostic: bool = False) -> Tuple[html.Div, str]:
    """Plot known interactions per bait.

    :param filtered_saint_input_json: Filtered SAINT output in pandas split-JSON format.
    :param db_file: Path to SQLite database file.
    :param rep_colors_with_cont: Mapping for contaminant and non-contaminant colors by bait.
    :param figure_defaults: Figure defaults for plotting.
    :param isoform_agnostic: If ``True``, match using base UniProt IDs (no isoforms).
    :returns: Tuple of (plot Div, processed SAINT output JSON).
    """
    logger.info(f'known_plot - started: {datetime.now()}')
    upid_a_col: str = 'uniprot_id_a'
    upid_b_col: str = 'uniprot_id_b'
    if isoform_agnostic:
        upid_a_col += '_noiso'
        upid_b_col += '_noiso'
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(filtered_saint_input_json), orient='split')
    db_conn = db_functions.create_connection(db_file)
    col_order: list = list(saint_output.columns)
    knowns: pd.DataFrame = db_functions.get_from_table_by_list_criteria(
        db_conn, 'known_interactions', upid_a_col, list(
            saint_output['Bait uniprot'].unique())
    )
    db_conn.close()
    # TODO: multibait
    saint_output = pd.merge(
        saint_output,
        knowns,
        left_on=['Bait uniprot', 'Prey'],
        right_on=[upid_a_col, upid_b_col],
        how='left'
    )
    saint_output['Known interaction'] = saint_output['update_time'].notna()
    logger.info(
        f'known_plot - knowns: {saint_output["Known interaction"].value_counts()}')
    col_order.append('Known interaction')
    col_order.extend([c for c in saint_output.columns if c not in col_order])
    saint_output = saint_output[col_order]
    figure_data: pd.DataFrame = count_knowns(
        saint_output, rep_colors_with_cont)
    figure_data.sort_values(by=['Bait', 'Known interaction'], ascending=[
                            True, False], inplace=True)
    figure_data.index = figure_data['Bait']
    figure_data.drop(columns=['Bait'], inplace=True)

    bait_map: dict = {bu: b for b, bu in saint_output[[
        'Bait', 'Bait uniprot']].drop_duplicates().values if bu != 'No bait uniprot'}

    known_str: str = 'Known interactions found per bait (Known / All):'
    no_knowns_found: set = set()
    for bait in figure_data.index:
        bdata: pd.DataFrame = figure_data[figure_data.index == bait]
        known_sum: int = bdata[bdata["Known interaction"]]["Prey count"].sum()
        if known_sum == 0:
            no_knowns_found.add(bait)
        else:
            known_str += f'{bait}: {known_sum} / {bdata["Prey count"].sum()}, '
    known_str = known_str.strip().strip(', ') + '. '
    known_str += f'No known interactions found: {", ".join(sorted(list(no_knowns_found)))}. '

    more_known = 'Known preys available for these baits in the database: '
    for index, value in knowns[upid_a_col].value_counts().items():
        more_known += f'{bait_map[index]} ({value}), '
    more_known = more_known.strip().strip(', ') + '. '
    if len(no_knowns_found) == len(figure_data.index.values):
        more_known = ''
    figtitle = 'High-confidence interactions and identified known interactions'
    return (
        html.Div(
            id='interactomics-saint-known-plot',
            children=[
                html.H4(id='interactomics-known-header',
                        children=figtitle),
                bar_graph.make_graph(
                    'interactomics-saint-filt-int-known-graph',
                    figure_defaults,
                    figtitle, 
                    figure_data,
                    '', color_discrete_map=True, y_name='Prey count', x_label='Bait'
                ),
                legends['known'],
                html.P(known_str),
                html.P(more_known)
            ],
            style={
                'overflowX': 'auto',
                'whiteSpace': 'nowrap'
            }
        ),
        saint_output.to_json(orient='split')
    )



def pca(saint_output_data: str, 
        defaults: Dict[str, Any], 
        replicate_colors: Dict[str, str]) -> Tuple[html.Div, str]:
    """Perform PCA on SAINT output and plot bait relationships.

    :param saint_output_data: SAINT output in pandas split-JSON format.
    :param defaults: Figure defaults.
    :param replicate_colors: Mapping ``'sample groups'`` -> color.
    :returns: Tuple of (plot Div, PCA data JSON). Returns empty plot if <2 baits.
    """
    data_table: pd.DataFrame = pd.read_json(StringIO(saint_output_data),orient='split')
    if len(data_table['Bait'].unique()) < 2:
        gdiv = ['Too few samle groups for PCA']
        pca_data = ''
    else:
        data_table = data_table.pivot_table(
            index='Prey', columns='Bait', values='AvgSpec')
        pc1: str
        pc2: str
        pca_result: pd.DataFrame
        # Compute PCA of the data
        spoofed_sample_groups: dict = {i: i for i in data_table.columns}
        pc1, pc2, pca_result = matrix_functions.do_pca(
            data_table.fillna(0), spoofed_sample_groups, n_components=2)
        pca_result.sort_values(by=pc1, ascending=True, inplace=True)
        pca_result['Sample group color'] = [replicate_colors['sample groups'][grp] for grp in pca_result['Sample group']]
        dlname = 'SPC PCA'
        gdiv = [
            html.H4(id='interactomics-pca-header', children=dlname),
            scatter.make_graph(
                'interactomics-pca-plot',
                defaults,
                dlname,
                pca_result,
                pc1,
                pc2,
                'Sample group color',
                'Sample group',
                hover_data=['Sample group', 'Sample name', pc1,pc2]
            ),
            legends['pca']
        ]
        pca_data = pca_result.to_json(orient='split')
    return (
        html.Div(
            id='interactomics-pca-plot-div',
            children=gdiv
        ),
        pca_data
    )

def enrich(saint_output_json: str, 
          chosen_enrichments: List[str], 
          figure_defaults: Dict[str, Any], 
          keep_all: bool = False, 
          sig_threshold: float = 0.01,
          parameters_file: str = 'config/parameters.toml') -> Tuple[List[html.Div], Dict[str, Any], List[Any]]:
    """Run selected enrichment methods and visualize results.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param chosen_enrichments: List of enrichment method names.
    :param figure_defaults: Figure defaults for plotting.
    :param keep_all: If ``True``, include non-significant rows meeting fold criteria.
    :param sig_threshold: Significance cutoff.
    :param parameters_file: Path to parameters TOML used by enrichment admin.
    :returns: Tuple of (list of result Divs, dict of enrichment data, list of info).
    """
    div_contents:list = []
    enrichment_data: dict = {}
    enrichment_information: list = []
    chosen_enrichments = [e for e in chosen_enrichments if len(e.strip()) > 0]
    if len(chosen_enrichments) == 0:
        return (
            div_contents,
            enrichment_data,
            enrichment_information
        )   
    e_admin = ea.EnrichmentAdmin(parameters_file)
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    enrichment_names: list
    enrichment_results: list
    enrichment_names, enrichment_results, enrichment_information = e_admin.enrich_all(
        saint_output,
        chosen_enrichments,
        id_column='Prey',
        split_by_column='Bait',
        split_name='Bait'
    )

    tablist: list = []
    for i, (rescol, sigcol, namecol, result) in enumerate(enrichment_results):
        if keep_all:
            keep_these: set = set(result[result[rescol] >= 2][namecol].values)
            keep_these = keep_these & set(
                result[result[sigcol] < sig_threshold][namecol].values)
            filtered_result: pd.DataFrame = result[result[namecol].isin(
                keep_these)]
        else:
            filtered_result = result[(result[sigcol]<sig_threshold) & (result[rescol]>=2)]
        matrix: pd.DataFrame = pd.pivot_table(
            filtered_result,
            index=namecol,
            columns='Bait',
            values=rescol
        ).fillna(0)
        if filtered_result.shape[0] == 0:
            graph = html.P('Nothing enriched.')
        else:
            enrichment_data[enrichment_names[i]] = {
                'sigcol': sigcol,
                'rescol': rescol,
                'namecol': namecol,
                'result': result.to_json(orient='split')
            }
            graph = heatmaps.make_heatmap_graph(
                matrix,
                f'interactomics-enrichment-{enrichment_names[i]}',
                rescol.replace('_', ' '),
                figure_defaults,
                cmap = 'dense',
                dlname = enrichment_names[i],
                symmetrical = False
            )

        table_label: str = f'{enrichment_names[i]} data table'
        table: dash_table.DataTable = dash_table.DataTable(
            data=filtered_result.to_dict('records'),
            columns=[{"name": i, "id": i} for i in filtered_result.columns],
            page_size=15,
            style_table={
                'maxHeight': 600
            },
            style_data={
                'width': '100px', 'minWidth': '25px', 'maxWidth': '250px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
            filter_action='native',
            id=f'interactomics-enrichment-{table_label.replace(" ","-")}',
        )
        e_legend: html.P = enrichment_legend(
            replace_accent_and_special_characters(enrichment_names[i]),
            enrichment_names[i],
            rescol,
            2,
            sigcol,
            sig_threshold
        )
        enrichment_tab: Card = Card(
            CardBody(
                [
                    html.H5(f'{enrichment_names[i]} heatmap'),
                    graph,
                    e_legend,
                    html.P(f'{enrichment_names[i]} data table'),
                    table
                ],
             #   style={'width': '98%'}
            ),
            #style={'width': '98%'}
        )

        tablist.append(
            Tab(
                enrichment_tab, label=enrichment_names[i],
               # style={'width': '98%'}
            )
        )
    if len(enrichment_results) > 0:
        div_contents: list = [
            html.H4(id='interactomics-enrichment-header', children='Enrichment'),
            Tabs(
                id='interactomics-enrichment-tabs',
                children=tablist,
                style={'width': '98%'}
            )]
    return (div_contents,
        enrichment_data,
        enrichment_information
    )


def map_intensity(saint_output_json: str, 
                 intensity_table_json: str, 
                 sample_groups: Dict[str, str]) -> str:
    """Map averaged intensity per group onto SAINT output rows.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param intensity_table_json: Intensity table in pandas split-JSON format.
    :param sample_groups: Mapping bait -> group name.
    :returns: SAINT output JSON with optional ``Averaged intensity`` column.
    """
    intensity_table: pd.DataFrame = pd.read_json(
        StringIO(intensity_table_json), orient='split')
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    has_intensity: bool = False
    intensity_column: list = [np.nan for _ in saint_output.index]
    if intensity_table.shape[0] > 1:
        if intensity_table.shape[1] > 1:
            if intensity_table.columns[0] != 'No data':
                has_intensity = True
    if has_intensity:
        intensity_column = []
        for _, row in saint_output.iterrows():
            try:
                intensity_column.append(
                    intensity_table[sample_groups[row['Bait']]].loc[row['Prey']].mean())
            except KeyError:
                intensity_column.append(np.nan)
        saint_output['Averaged intensity'] = intensity_column
    return saint_output.to_json(orient='split')


def saint_histogram(saint_output_json: str, 
                   figure_defaults: Dict[str, Any]) -> Tuple[html.Div, str]:
    """Create a histogram of BFDR scores from SAINT output.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param figure_defaults: Figure defaults for plotting.
    :returns: Tuple of (histogram Div, histogram data JSON).
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    return (
        histogram.make_figure(saint_output, 'BFDR', '', figure_defaults),
        saint_output.to_json(orient='split')
    )


def add_bait_column(saint_output: pd.DataFrame, 
                    bait_uniprot_dict: Dict[str, str]) -> pd.DataFrame:
    """Add bait UniProt and bait-self flags to SAINT output.

    :param saint_output: SAINT output DataFrame with ``Bait`` and ``Prey``.
    :param bait_uniprot_dict: Mapping bait name -> UniProt IDs (``;`` separated allowed).
    :returns: DataFrame with ``Bait uniprot`` and ``Prey is bait`` added.
    """
    saint_output['Bait'] = [b.rsplit('_',maxsplit=1)[0] for b in saint_output['Bait'].values]
    bu_column: list = []
    prey_is_bait: list = []
    for _, row in saint_output.iterrows():
        if row['Bait'] in bait_uniprot_dict:
            bu_column.append(bait_uniprot_dict[row['Bait']])
            prey_is_bait.append(row['Prey'].lower().strip() in [b.lower().strip() for b in bu_column[-1].split(';')])
        else:
            bu_column.append('No bait uniprot')
            prey_is_bait.append(False)
    saint_output['Bait uniprot'] = bu_column
    saint_output['Prey is bait'] = prey_is_bait
    return saint_output

def saint_cmd(saint_input: Dict[str, List[List[str]]], 
             saint_tempdir: List[str], 
             session_uid: str) -> str:
    """Run SAINTexpressSpc on prepared input files.

    :param saint_input: Dict with keys ``bait``, ``prey``, ``int`` containing row lists.
    :param saint_tempdir: List of path segments for temp dir base.
    :param session_uid: Unique identifier to isolate run directory.
    :returns: Path to directory containing ``list.txt`` (or dummy if SAINT missing).
    :raises OSError: On temp dir creation failure.
    :raises sh.CommandNotFound: If SAINTexpressSpc is not available.
    """
    temp_dir: str = os.path.join(*(saint_tempdir))
    temp_dir = os.path.join(temp_dir, session_uid)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    with (
        tempfile.NamedTemporaryFile() as baitfile,
        tempfile.NamedTemporaryFile() as preyfile,
        tempfile.NamedTemporaryFile() as intfile,
    ):
        baitfile.write(
            ('\n'.join([
                '\t'.join(x) for x in saint_input['bait']
            ])).encode('utf-8')
        )
        preyfile.write(
            ('\n'.join([
                '\t'.join(x) for x in saint_input['prey']
            ])).encode('utf-8')
        )
        intfile.write(
            ('\n'.join([
                '\t'.join(x) for x in saint_input['int']
            ])).encode('utf-8')
        )
        baitfile.flush()
        preyfile.flush()
        intfile.flush()
        try:
            sh.SAINTexpressSpc(intfile.name, preyfile.name, baitfile.name, _cwd=temp_dir)
        except sh.CommandNotFound:
            create_dummy_list_txt(temp_dir, saint_input)
    return temp_dir

def create_dummy_list_txt(temp_dir: str, saint_input: Dict[str, List[List[str]]]) -> None:
    """Create a dummy SAINT ``list.txt`` when SAINTexpressSpc is unavailable.

    Generates a plausible-looking SAINT output file using random values so that
    downstream steps can proceed in demo or fallback mode.

    :param temp_dir: Target directory to write ``list.txt`` and marker file.
    :param saint_input: SAINT input dict with keys ``bait``, ``prey``, ``int``.
    :returns: None
    """
    baits = {}
    baitmap = {}
    for baitrun, group, ctrl in saint_input['bait']:
        baits.setdefault(ctrl, {}).setdefault(group, [])
        baits[ctrl][group].append(baitrun)
        baitmap[baitrun] = (ctrl, group)

    preys = {}
    for prey, _, gname in saint_input['prey']:
        preys[prey] = gname

    counts = {}
    max_b_len = 0
    for baitrun, group, prey, spc in saint_input['int']:
        counts.setdefault(group, {}).setdefault(prey, [])
        counts[group][prey].append(spc)
        max_b_len = max(max_b_len, len(counts[group][prey]))

    control_counts = {}
    max_ctrl_len = 0
    for baitgrp in baits['C'].keys():
        for prey, spc in counts[baitgrp].items():
            control_counts.setdefault(prey,[])
            control_counts[prey].extend(spc)
            max_ctrl_len = max(max_ctrl_len, len(control_counts[prey]))

    def pad(li, le):
        if len(li) > le:
            return li
        rlist = li
        for i in range(len(li), le):
            rlist.append('0')
        return rlist

    list_txt = []
    alpha = 1
    beta = 0.3
    for group, pdic in counts.items():
        if group in baits['C']: continue
        for prey, spclist in pdic.items():
            bfdr_random = np.random.beta(alpha, beta)
            score_random = 1-bfdr_random*3
            p_ctrl_list = []
            if prey in control_counts:
                p_ctrl_list = control_counts[prey]
            spclist = pad(spclist, max_b_len)
            p_ctrl_list = pad(p_ctrl_list, max_ctrl_len)
            list_txt.append([
                group,
                prey, 
                preys[prey], 
                '|'.join(spclist), 
                sum([int(x) for x in spc])/len(spc),
                sum([int(x) for x in spc]),
                len(baits['T'][group]),
                '|'.join(p_ctrl_list),
                0,0,0,0,score_random, 1200, bfdr_random, np.nan])
    lt = pd.DataFrame(data=list_txt, columns=['Bait', 'Prey', 'PreyGene', 'Spec', 'SpecSum', 'AvgSpec', 'NumReplicates', 'ctrlCounts', 'AvgP', 'MaxP', 'TopoAvgP', 'TopoMaxP', 'SaintScore', 'FoldChange', 'BFDR', 'boosted_by'])
    lt.to_csv(os.path.join(temp_dir, 'list.txt'), sep='\t', index=False)
    with open(os.path.join(temp_dir, 'list_is_dummy.txt'), 'w') as f:
        f.write('this list has been created by dummy saint simulator that produces nonsense. This happened because SAINTexpressSpc was not found.')

def run_saint(saint_input: Dict[str, List[List[str]]], 
             saint_tempdir: List[str], 
             session_uid: str, 
             bait_uniprots: Dict[str, str], 
             cleanup: bool = True) -> Tuple[str, bool]:
    """Execute SAINT pipeline and return processed output.

    :param saint_input: SAINT input dict.
    :param saint_tempdir: Temp directory base as path segments.
    :param session_uid: Unique run identifier.
    :param bait_uniprots: Mapping bait -> UniProt IDs.
    :param cleanup: If ``True``, remove temp files after success.
    :returns: Tuple of (output JSON or error string, saint_missing_flag).
    """
    # Can not use logging in this function, since it's called from a background_callback_manager using celery, and logging will lead to a hang.
    # Instead, we can use print statements, and they will show up as WARNINGS in celery log.
    temp_dir: str = ''
    if ('bait' in saint_input) and ('prey' in saint_input):
        temp_dir = saint_cmd(saint_input, saint_tempdir, session_uid)
    failed: bool = not os.path.isfile(os.path.join(temp_dir, 'list.txt'))
    saintfail: bool = os.path.isfile(os.path.join(temp_dir, 'list_is_dummy.txt'))
    if failed:
        ret: str = 'SAINT failed. Can not proceed.'
    else:
        ret = add_bait_column(pd.read_csv(os.path.join(
            temp_dir, 'list.txt'), sep='\t'), bait_uniprots)
        ret = ret.to_json(orient='split')
        if cleanup:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError as e:
                print(
                    f'run_saint:  Could not clean up after SAINT run: {datetime.now()} {e}')
    return (ret, saintfail)


def prepare_crapome(db_conn: sqlite3.Connection, 
                   crapomes: List[str]) -> pd.DataFrame:
    """Prepare CRAPome tables for downstream filtering.

    :param db_conn: SQLite connection.
    :param crapomes: List of CRAPome table names (possibly with suffixes).
    :returns: DataFrame with per-CRAPome frequency and spc averages plus max frequency.
    """
    crapomes = [c.rsplit('_(',maxsplit=1)[0] for c in crapomes]
    crapome_tables: list = [
        db_functions.get_full_table_as_pd(db_conn, tablename, index_col='protein_id') for tablename in crapomes
    ]
    crapome_table: pd.DataFrame = pd.concat(
        [
            crapome_tables[i][['frequency', 'spc_avg']].
            rename(columns={
                'frequency': f'{table_name}_frequency',
                'spc_avg': f'{table_name}_spc_avg'
            })
            for i, table_name in enumerate(crapomes)
        ],
        axis=1
    )
    crapome_freq_cols: list = [
        c for c in crapome_table.columns if '_frequency' in c]
    crapome_table['Max crapome frequency'] = crapome_table[crapome_freq_cols].max(
        axis=1)
    return crapome_table

def prepare_controls(input_data_dict: Dict[str, Any], 
                    uploaded_controls: List[str], 
                    additional_controls: List[str], 
                    db_conn: sqlite3.Connection, 
                    select_most_similar_only: bool = False, 
                    top_n: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble uploaded and DB controls for SAINT.

    :param input_data_dict: Inputs including sample groups and SPC data tables.
    :param uploaded_controls: Names of uploaded control groups.
    :param additional_controls: Additional DB control table names.
    :param db_conn: SQLite connection.
    :param select_most_similar_only: If ``True``, keep only most similar controls.
    :param top_n: Number of controls to keep per-sample when filtering.
    :returns: Tuple of (SPC table without control columns, combined control table).
    """
    
    logger.debug(f'additional controls: {additional_controls}')
    additional_controls = [c.rsplit('_(',maxsplit=1)[0] for c in additional_controls]
    logger.debug(f'preparing uploaded controls: {uploaded_controls}')
    logger.debug(f'preparing additional controls: {additional_controls}')
    sample_groups: dict = input_data_dict['sample groups']['norm']
    spc_table: pd.DataFrame = pd.read_json(
        StringIO(input_data_dict['data tables']['spc']), orient='split')
    controls: list = []
    for control_name in additional_controls:
        ctable: pd.DataFrame = db_functions.get_full_table_as_pd(
            db_conn, control_name, index_col='PROTID')
        ctable.index.name = ''
        controls.append(ctable)
        logger.debug(f'control {control_name} shape: {ctable.shape}, indexvals: {list(ctable.index)[:5]}')
        
    if (len(controls) > 0) and select_most_similar_only:
        # groupby to merge possible duplicate columns that are annotated in multiple sets
        # mean grouping should have no effect, since PSM values SHOULD be the same in any case.
        control_table = filter_controls_by_similarity(spc_table, controls, top_n)
        controls = [control_table]
    control_cols: list = []
    for cg in uploaded_controls:
        control_cols.extend(sample_groups[cg])
    controls.append(spc_table[control_cols])
    spc_table = spc_table[[c for c in spc_table.columns if c not in control_cols]]
    control_table: pd.DataFrame = pd.concat(controls, axis=1).T.groupby(level=0).mean().T
    logger.debug(f'Controls concatenated: {control_table.shape}, indexvals: {list(control_table.index)[:5]}')
    logger.debug(f'SPC table index: {list(spc_table.index)[:5]}')
    # Discard any control preys that are not identified in baits. It will not affect SAINT results.
    control_table.drop(index=set(control_table.index) -
                       set(spc_table.index), inplace=True)
    logger.debug(f'non-detected preys dropped: {control_table.shape}')

    return (spc_table, control_table)

def filter_controls_by_similarity(spc_table: pd.DataFrame, 
                   controls: List[pd.DataFrame], 
                   top_n: int) -> pd.DataFrame:
    """Filter control runs by similarity to experiment runs.

    :param spc_table: Spectral count table for experiment samples.
    :param controls: List of candidate control tables.
    :param top_n: Number of top similar controls to keep per sample.
    :returns: Filtered control table with selected columns.
    """
    control_table: pd.DataFrame = pd.concat(controls, axis=1).T.groupby(level=0).mean().T
    chosen_controls: list = []
    for c in spc_table.columns:
        controls_ranked_by_similarity: list = matrix_functions.ranked_dist(
            spc_table[[c]], control_table)
        chosen_controls.extend([s[0] for s in controls_ranked_by_similarity[:top_n]])
    control_table = control_table[list(set(chosen_controls))]
    return control_table

def add_crapome(saint_output_json: str, 
                crapome_json: str) -> str:
    """Merge CRAPome annotations into SAINT output JSON.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param crapome_json: CRAPome table in pandas split-JSON format.
    :returns: Merged SAINT output JSON.
    """
    if 'Saint failed.' in saint_output_json:
        return saint_output_json
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    crapome: pd.DataFrame = pd.read_json(StringIO(crapome_json),orient='split')

    return pd.merge(
        saint_output,
        crapome,
        left_on='Prey',
        right_index=True,
        how='left'
    ).to_json(orient='split')


def make_saint_dict(spc_table: pd.DataFrame, 
                   rev_sample_groups: Dict[str, str], 
                   control_table: pd.DataFrame, 
                   protein_table: pd.DataFrame) -> Dict[str, List[List[str]]]:
    """Create SAINT input dict from SPC and metadata tables.

    :param spc_table: Spectral count data.
    :param rev_sample_groups: Mapping sample -> group.
    :param control_table: Control spectral count table.
    :param protein_table: Protein info with columns ``uniprot_id``, ``length``, ``gene_name``.
    :returns: Dict with keys ``bait``, ``prey``, ``int`` as lists of rows.
    """
    protein_lenghts_and_names = {}
    logger.info(
        f'make_saint_dict: start: {datetime.now()}')
    for _, row in protein_table.iterrows():
        protein_lenghts_and_names[row['uniprot_id']] = {
            'length': row['length'], 'gene name': row['gene_name']}

    bait: list = []
    prey: list = []
    inter: list = []
    for col in spc_table.columns:
        bait.append([col, rev_sample_groups[col]+'_bait', 'T'])
    for col in control_table.columns:
        if col in rev_sample_groups:
            bait.append([col, rev_sample_groups[col]+'_bait', 'C'])
        else:
            bait.append([col, 'inbuilt_ctrl', 'C'])
    logger.info(
        f'make_saint_dict: Baits prepared: {datetime.now()}')
    logger.info(
        f'make_saint_dict: Control table shape: {control_table.shape}')
    control_melt: pd.DataFrame = pd.melt(
        control_table, ignore_index=False).replace(0, np.nan).dropna().reset_index()
    sgroups = []
    for _, srow in control_melt.iterrows():
        sgroup = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]+'_bait'
        sgroups.append(sgroup)
    control_melt['sgroup'] = sgroups
    control_melt = control_melt.reindex(
        columns=['variable', 'sgroup', 'index', 'value'])
    control_melt['value'] = control_melt['value'].astype(int)
    inter.extend(control_melt.values
                 .astype(str).tolist())
    logger.info(
        f'make_saint_dict: Control table melted: {control_melt.shape}: {datetime.now()}')
    logger.info(
        f'make_saint_dict: Control interactions prepared: {datetime.now()}')
    for uniprot, srow in pd.melt(spc_table, ignore_index=False).replace(0, np.nan).dropna().iterrows():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]+'_bait'
        inter.append([srow['variable'], sgroup, uniprot, str(int(srow['value']))])
    logger.info(
        f'make_saint_dict: SPC table interactions prepared: {datetime.now()}')
    for uniprotid in (set(control_table.index.values) | set(spc_table.index.values)):
        try:
            plen: str = str(protein_lenghts_and_names[uniprotid]['length'])
            gname: str = str(protein_lenghts_and_names[uniprotid]['gene name'])
        except KeyError:
            logger.warning(
                f'make_saint_dict: No length found for uniprot: {uniprotid}')
            plen = '200'
            gname = str(uniprotid)
        prey.append([str(uniprotid), plen, gname])
    logger.info(
        f'make_saint_dict: Preys prepared: {datetime.now()}')
    return {'bait': bait, 'prey': prey, 'int': inter}

def do_ms_microscopy(saint_output_json: str, 
                    db_file: str, 
                    figure_defaults: Dict[str, Any], 
                    version: str = 'v1.0') -> Tuple[html.Div, str]:
    """Perform MS-microscopy localization analysis and visualize.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param db_file: SQLite DB path for MS-microscopy reference.
    :param figure_defaults: Figure defaults.
    :param version: Analysis version tag.
    :returns: Tuple of (plots Div, results JSON).
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split'
    )
    db_conn = db_functions.create_connection(db_file)
    msmic_reference = db_functions.get_full_table_as_pd(
        db_conn, 'msmicroscopy', index_col='Interaction'
    )
    db_conn.close() # type: ignore
    msmic_results: pd.DataFrame = ms_microscopy.generate_msmic_dataframes(saint_output, msmic_reference, )

    polar_plots: list = [
        (bait, ms_microscopy.localization_graph(f'interactomics-msmic-{bait}',figure_defaults, 'polar', bait, data_row))
        for bait, data_row in msmic_results.iterrows()
    ]
    msmic_heatmap = ms_microscopy.localization_graph(f'interactomics-msmic-heatmap', figure_defaults, 'heatmap', 'All baits', msmic_results)

    tablist: list = [
        Tab(
            Card(
                CardBody(
                    [
                        html.H5('MS-microscopy heatmap'),
                        msmic_heatmap, 
                        legends['ms-microscopy-all']
                    ],
          #          style={'width': '98%'}
                ),
         #       style={'width': '98%'}
            ),
        label = 'Overall results',
        #style={'width': '98%'}
        )
    ]

    i = 0
    for bait, polar_graph in polar_plots:
        tablist.append(
            Tab(
                Card(
                    CardBody(
                        [
                            html.H5(f'MS-microscopy for {bait}'),
                            polar_graph,
                            leg_rep(
                                legends['ms-microscopy-single'],
                                'BAITSTRING',
                                bait
                            )
                        ],
                #        style={'width': '98%'}
                    ),
                #style={'width': '98%'}
                ),
                label = bait,
                #style={'width': '98%'}
            )
        )
    return (
        html.Div(
            id='interactomics-msmicroscopy-plot-div',
            children=[
                html.H4(id='interactomics-msmic-header', children='MS-microscopy'),
                Tabs(
                    id = 'interactomics-msmicroscopy-tabs',
                    children = tablist,
                    style = {'width': '98%'}
                ),
            ]
        ),
        msmic_results.to_json(orient='split')
    )

def generate_saint_container(input_data_dict: Dict[str, Any], 
                           uploaded_controls: List[str], 
                           additional_controls: List[str], 
                           crapomes: List[str], 
                           db_file: str, 
                           select_most_similar_only: bool, 
                           n_controls: int) -> Tuple[html.Div, Dict[str, List[List[str]]], str]:
    """Build SAINT UI container and prepare inputs.

    :param input_data_dict: Input data and metadata including sample groups.
    :param uploaded_controls: Uploaded control group names.
    :param additional_controls: Additional DB control names.
    :param crapomes: CRAPome dataset names.
    :param db_file: SQLite database path.
    :param select_most_similar_only: If ``True``, filter controls by similarity.
    :param n_controls: Number of controls to keep when filtering.
    :returns: Tuple of (container Div, SAINT input dict, CRAPome JSON).
    """
    if '["No data"]' in input_data_dict['data tables']['spc']:
        return (html.Div(['No spectral count data in input, cannot run SAINT.']),{},'')
    logger.info(
        f'generate_saint_container: preparations started: {datetime.now()}')
    db_conn = db_functions.create_connection(db_file)
    additional_controls = [
        f'control_{ctrl_name.lower().replace(" ","_")}' for ctrl_name in additional_controls]
    crapomes = [
        f'crapome_{crap_name.lower().replace(" ","_")}' for crap_name in crapomes]
    logger.info(f'generate_saint_container: DB connected')
    spc_table: pd.DataFrame
    control_table: pd.DataFrame
    spc_table, control_table = prepare_controls(
        input_data_dict, uploaded_controls, additional_controls, db_conn, select_most_similar_only, n_controls)
    logger.info(f'generate_saint_container: Controls prepared')
    protein_list: list = list(
        set(spc_table.index.values) | set(control_table.index))
    protein_table: pd.DataFrame = db_functions.get_from_table(
        db_conn,
        'proteins',
        select_col=[
            'uniprot_id',
            'length',
            'gene_name'
        ],
        as_pandas=True
    )
    logger.info(f'generate_saint_container: Protein table retrieved')
    protein_table = protein_table[protein_table['uniprot_id'].isin(
        protein_list)]
    if len(crapomes) > 0:
        crapome: pd.DataFrame = prepare_crapome(db_conn, crapomes)
        crapome.drop(index=set(crapome.index) -
                     set(spc_table.index), inplace=True)
    else:
        crapome = pd.DataFrame()
    db_conn.close()

    saint_dict: dict = make_saint_dict(
        spc_table, input_data_dict['sample groups']['rev'], control_table, protein_table)
    logger.info(
        f'generate_saint_container: SAINT dict done: {datetime.now()}')
    return (
        html.Div(
            id='interactomics-saint-container',
            children=[
                html.Div(id='interactomics-saint-filtering-container')
            ]
        ),
        saint_dict,
        crapome.to_json(orient='split')
    )


def saint_filtering(saint_output_json: str, 
                   bfdr_threshold: float, 
                   crapome_percentage: float, 
                   crapome_fc: float, 
                   do_rescue: bool = False) -> str:
    """Filter SAINT output by BFDR and CRAPome thresholds.

    :param saint_output_json: SAINT output in pandas split-JSON format.
    :param bfdr_threshold: BFDR threshold for filtering.
    :param crapome_percentage: CRAPome frequency threshold.
    :param crapome_fc: CRAPome fold-change threshold for rescue.
    :param do_rescue: If ``True``, keep preys that pass in any bait.
    :returns: Filtered SAINT output JSON.
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    logger.info(f'saint filtering - beginning: {saint_output.shape}')
    logger.info(
        f'saint filtering - beginning nodupes: {saint_output.drop_duplicates().shape}')
    saint_output = saint_output.drop_duplicates()
    crapome_columns: list = []
    for column in saint_output.columns:
        if '_frequency' in column:
            crapome_columns.append(
                (column, column.replace('_frequency', '_spc_avg')))
    keep_col: list = []
    bfdr_disc = 0
    crapome_disc = 0
    keep_preys: set = set()
    for _, row in saint_output.iterrows():
        keep: bool = True
        if row['BFDR'] > bfdr_threshold:
            keep = False
            bfdr_disc += 1
        elif 'Max crapome frequency' in saint_output.columns:
            if row['Max crapome frequency'] > crapome_percentage:
                for freq_col, fc_col in crapome_columns:
                    if row[freq_col] >= crapome_percentage:
                        if row[fc_col] <= crapome_fc:
                            keep = False
                            crapome_disc += 1
                            break
        if keep:
            keep_preys.add(row['Prey'])
        keep_col.append(keep)

    logger.info(
        f'saint filtering - Preys pass filter: {len(keep_preys)}')
    saint_output['Passes filter'] = keep_col
    logger.info(
        f'saint filtering - Saint output pass filter: {saint_output["Passes filter"].value_counts()}')
    saint_output['Passes filter with rescue'] = saint_output['Prey'].isin(
        keep_preys)
    logger.info(
        f'saint filtering - Saint output pass filter with rescue: {saint_output["Passes filter with rescue"].value_counts()}')
    if do_rescue:
        use_col: str = 'Passes filter with rescue'
    else:
        use_col = 'Passes filter'
    filtered_saint_output: pd.DataFrame = saint_output[
        saint_output[use_col]
    ].copy()

    logger.info(
        f'saint filtering - filtered size: {filtered_saint_output.shape}')
    if 'Bait uniprot' in filtered_saint_output.columns:
        filtered_saint_output = filtered_saint_output[
            filtered_saint_output['Prey is bait']==False
        ]
    colorder: list = ['Bait', 'Bait uniprot', 'Prey', 'PreyGene', 'Prey is bait',
                      'Passes filter', 'Passes filter with rescue', 'AvgSpec']
    colorder.extend(
        [c for c in filtered_saint_output.columns if c not in colorder])
    filtered_saint_output = filtered_saint_output[colorder]
    logger.info(
        f'saint filtering - bait removed filtered size: {filtered_saint_output.shape}')
    logger.info(
        f'saint filtering - bait removed filtered size nodupes: {filtered_saint_output.drop_duplicates().shape}')
    return filtered_saint_output.reset_index().drop(columns=['index']).to_json(orient='split')

def get_saint_matrix(saint_data_json: str) -> pd.DataFrame:
    """Convert SAINT output JSON to prey x bait matrix of AvgSpec.

    :param saint_data_json: SAINT output in pandas split-JSON format.
    :returns: Pivot table DataFrame (rows=Prey, cols=Bait, values=AvgSpec).
    """
    df = pd.read_json(StringIO(saint_data_json),orient='split')
    return df.pivot_table(index='Prey',columns='Bait',values='AvgSpec')

def saint_counts(filtered_output_json: str, 
                figure_defaults: Dict[str, Any], 
                replicate_colors: Dict[str, str]) -> Tuple[html.Div, str]:
    """Count prey per bait and plot as a bar chart.

    :param filtered_output_json: Filtered SAINT output in pandas split-JSON format.
    :param figure_defaults: Figure defaults for plotting.
    :param replicate_colors: Mapping ``'sample groups'`` -> color.
    :returns: Tuple of (bar plot Div, count data JSON).
    """
    count_df: pd.DataFrame = pd.read_json(StringIO(filtered_output_json),orient='split')['Bait'].\
        value_counts().\
        to_frame(name='Prey count')
    count_df['Color'] = [
        replicate_colors['sample groups'][index] for index in count_df.index.values
    ]
    return (
        bar_graph.bar_plot(
            figure_defaults,
            count_df,
            title='',
            hide_legend=True,
            x_label='Sample group'
        ),
        count_df.to_json(orient='split')
    )
