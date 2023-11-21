from dash import html, dash_table
import pandas as pd
from components import db_functions
import numpy as np
import shutil
import os
import subprocess
import json
from components.figures import histogram, bar_graph, scatter, heatmaps
from components import matrix_functions, db_functions
from components.figures.figure_legends import INTERACTOMICS_LEGENDS as legends
from components.figures.figure_legends import enrichment_legend
from components.text_handling import replace_accent_and_special_characters
from components import EnrichmentAdmin as ea
from dash_bootstrap_components import Card, CardBody, Tab, Tabs
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def count_knowns(saint_output, replicate_colors) -> pd.DataFrame:
    data: pd.DataFrame = saint_output[['Bait', 'Known interaction']].\
        value_counts().to_frame().reset_index().rename(
            columns={0: 'Prey count'})
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


def known_plot(filtered_saint_input_json, db_file, rep_colors_with_cont, figure_defaults, isoform_agnostic=False) -> tuple:
    logger.debug(f'known_plot - started: {datetime.now()}')
    upid_a_col: str = 'uniprot_id_a'
    upid_b_col: str = 'uniprot_id_b'
    if isoform_agnostic:
        upid_a_col += '_noiso'
        upid_b_col += '_noiso'
    saint_output: pd.DataFrame = pd.read_json(
        filtered_saint_input_json, orient='split')
    db_conn = db_functions.create_connection(db_file)
    col_order: list = list(saint_output.columns)
    knowns: pd.DataFrame = db_functions.get_from_table_by_list_criteria(
        db_conn, 'known_interactions', upid_a_col, list(
            saint_output['Bait uniprot'].unique())
    )
    saint_output = pd.merge(
        saint_output,
        knowns,
        left_on=['Bait uniprot', 'Prey'],
        right_on=[upid_a_col, upid_b_col],
        how='left'
    )
    saint_output['Known interaction'] = saint_output['update_time'].notna()
    logger.debug(
        f'known_plot - knowns: {saint_output["Known interaction"].value_counts()}')
    db_conn.close()
    col_order.append('Known interaction')
    col_order.append(
        'Following columns are information about known interactions.')
    saint_output[col_order[-1]] = np.nan
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
    more_known = known_str.strip().strip(', ') + '. '
    return (
        html.Div(
            id='interactomics-saint-known-plot',
            children=[
                html.H4(id='interactomics-known-header',
                        children='Identified known interactions'),
                bar_graph.make_graph(
                    'interactomics-saint-filt-int-known-graph',
                    figure_defaults,
                    figure_data,
                    '', color_discrete_map=True, y_name='Prey count', x_label='Bait'
                ),
                legends['known'],
                html.P(known_str),
                html.Br(),
                html.P(more_known)
            ]
        ),
        saint_output.to_json(orient='split')
    )


def pca(saint_output_data: dict, defaults: dict) -> tuple:
    data_table: pd.DataFrame = pd.read_json(saint_output_data, orient='split')
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

    return (
        html.Div(
            id='interactomics-pca-plot-div',
            children=[
                html.H4(id='interactomics-pca-header', children='SPC PCA'),
                scatter.make_graph(
                    'interactomics-pca-plot',
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


def enrich(saint_output_json: str, chosen_enrichments: list, figure_defaults, sig_threshold: float = 0.01) -> tuple:
    e_admin = ea.EnrichmentAdmin()
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
    enrichment_names: list
    enrichment_results: list
    enrichment_information: list
    enrichment_names, enrichment_results, enrichment_information = e_admin.enrich_all(
        saint_output,
        chosen_enrichments,
        id_column='Prey',
        split_by_column='Bait',
        split_name='Bait'
    )

    enrichment_data: dict = {}

    tablist: list = []
    for i, (rescol, sigcol, namecol, result) in enumerate(enrichment_results):
        keep_these: set = set(result[result[rescol] >= 2][namecol].values)
        keep_these = keep_these & set(
            result[result[sigcol] < sig_threshold][namecol].values)
        filtered_result: pd.DataFrame = result[result[namecol].isin(
            keep_these)]
        matrix: pd.DataFrame = pd.pivot_table(
            filtered_result,
            index=namecol,
            columns='Bait',
            values=rescol
        )
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
                figure_defaults
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
        e_legend: str = enrichment_legend(
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
                style={'width': '98%'}
            ),
            className="mt-3",
            style={'width': '98%'}
        )

        tablist.append(
            Tab(
                enrichment_tab, label=enrichment_names[i]
            )
        )
    return ([
        html.H4(id='interactomics-enrichment-header', children='Enrichment'),
        Tabs(
            id='interactomics-enrichment-tabs',
            children=tablist,
            style={'width': '98%'}
        )],
        enrichment_data,
        enrichment_information
    )


def map_intensity(saint_output_json: str, intensity_table_json: str, sample_groups: dict) -> list:
    intensity_table: pd.DataFrame = pd.read_json(
        intensity_table_json, orient='split')
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
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


def saint_histogram(saint_output_json: str, figure_defaults):
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
    return (
        histogram.make_figure(saint_output, 'BFDR', '', figure_defaults),
        saint_output.to_json(orient='split')
    )


def add_bait_column(saint_output, bait_uniprot_dict) -> pd.DataFrame:
    bu_column: list = []
    for _, row in saint_output.iterrows():
        if row['Bait'] in bait_uniprot_dict:
            bu_column.append(bait_uniprot_dict[row['Bait']])
        else:
            bu_column.append('No bait uniprot')
    saint_output['Bait uniprot'] = bu_column
    return saint_output


def run_saint(saint_input: dict, saint_path: list, session_uid: str, bait_uniprots: dict, cleanup: bool = True) -> str:
    # Can not use logging in this function, since it's called from a long_callback using celery, and logging will lead to a hang.
    # Instead, we can use print statements, and they will show up as WARNINGS in celery log.
    print(f'run_saint run: {datetime.now()}')
    temp_dir: str = os.path.join(*(saint_path[:-1]))
    saint_cmd: str = os.path.join(temp_dir, saint_path[-1])
    temp_dir = os.path.join(temp_dir, session_uid)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    paths: list = [os.path.join(temp_dir, x)
                   for x in 'inter.dat,prey.dat,bait.dat'.split(',')]
    with open(paths[0], 'w', encoding='utf-8') as fil:
        fil.write('\n'.join([
            '\t'.join(x) for x in saint_input['int']
        ]))
    with open(paths[1], 'w', encoding='utf-8') as fil:
        fil.write('\n'.join([
            '\t'.join(x) for x in saint_input['prey']
        ]))
    with open(paths[2], 'w', encoding='utf-8') as fil:
        fil.write('\n'.join([
            '\t'.join(x) for x in saint_input['bait']
        ]))
    print(f'run_saint written: {datetime.now()}')
    try:
        print(f'run_saint running: {datetime.now()}')
        subprocess.check_output(
            [saint_cmd], stderr=subprocess.STDOUT, cwd=temp_dir, text=True)
        failed: bool = not os.path.isfile(os.path.join(temp_dir, 'list.txt'))
    except subprocess.CalledProcessError as e:
        print(f'run_saint: SAINT failed: {datetime.now()} {e} ')
        failed = True
    print(f'run_saint done: {datetime.now()}')
    if failed:
        ret: str = 'SAINT failed. Can not proceed.'
    else:
        ret = add_bait_column(pd.read_csv(os.path.join(
            temp_dir, 'list.txt'), sep='\t'), bait_uniprots).to_json(orient='split')
        if cleanup:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError as e:
                print(
                    f'run_saint:  Could not clean up after SAINT run: {datetime.now()} {e}')
    return ret


def prepare_crapome(db_conn, crapomes: list) -> pd.DataFrame:
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


def prepare_controls(input_data_dict, uploaded_controls, additional_controls, db_conn, do_proximity_filtering: bool = True, top_n: int = 30) -> tuple:
    logger.debug(f'preparing uploaded controls: {uploaded_controls}')
    logger.debug(f'preparing additional controls: {additional_controls}')
    sample_groups: dict = input_data_dict['sample groups']['norm']
    spc_table: pd.DataFrame = pd.read_json(
        input_data_dict['data tables']['spc'], orient='split')
    controls: list = []
    for control_name in additional_controls:
        ctable: pd.DataFrame = db_functions.get_full_table_as_pd(
            db_conn, control_name, index_col='PROTID')
        ctable.index.name = ''
        controls.append(ctable)
        logger.debug(f'control {control_name} shape: {ctable.shape}')

    if (len(controls) > 0) and do_proximity_filtering:
        # groupby to merge possible duplicate columns that are annotated in multiple sets
        # mean grouping should have no effect, since PSM values SHOULD be the same in any case.
        control_table: pd.DataFrame = pd.concat(
            controls, axis=1).groupby(level=0, axis=1).mean()
        controls_ranked_by_similarity: list = matrix_functions.ranked_dist(
            spc_table, control_table)
        control_table = control_table[[s[0]
                                       for s in controls_ranked_by_similarity[:top_n]]]
        controls = [control_table]
    control_cols: list = []
    for cg in uploaded_controls:
        control_cols.extend(sample_groups[cg])
    controls.append(spc_table[control_cols])
    spc_table = spc_table[[
        c for c in spc_table.columns if c not in control_cols]]
    control_table: pd.DataFrame = pd.concat(
        controls, axis=1).groupby(level=0, axis=1).mean()
    # Discard any control preys that are not identified in baits. It will not affect SAINT results.
    control_table.drop(index=set(control_table.index) -
                       set(spc_table.index), inplace=True)

    return (spc_table, control_table)


def add_crapome(saint_output_json, crapome_json) -> str:
    if 'Saint failed.' in saint_output_json:
        return saint_output_json
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
    crapome: pd.DataFrame = pd.read_json(crapome_json, orient='split')

    return pd.merge(
        saint_output,
        crapome,
        left_on='Prey',
        right_index=True,
        how='left'
    ).to_json(orient='split')


def make_saint_dict(spc_table, rev_sample_groups, control_table, protein_table) -> dict:
    protein_lenghts_and_names = {}
    logger.debug(
        f'make_saint_dict: start: {datetime.now()}')
    for _, row in protein_table.iterrows():
        protein_lenghts_and_names[row['uniprot_id']] = {
            'length': row['length'], 'gene name': row['gene_name']}

    bait: list = []
    prey: list = []
    inter: list = []
    for col in spc_table.columns:
        bait.append([col, rev_sample_groups[col], 'T'])
    for col in control_table.columns:
        if col in rev_sample_groups:
            bait.append([col, rev_sample_groups[col], 'C'])
        else:
            bait.append([col, 'inbuilt_ctrl', 'C'])
    logger.debug(
        f'make_saint_dict: Baits prepared: {datetime.now()}')
    logger.debug(
        f'make_saint_dict: Control table shape: {control_table.shape}')
    control_melt: pd.DataFrame = pd.melt(
        control_table, ignore_index=False).replace(0, np.nan).dropna().reset_index()
    control_melt['sgroup'] = 'inbuilt_ctrl'
    control_melt = control_melt.reindex(
        columns=['variable', 'sgroup', 'index', 'value'])
    inter.extend(control_melt.values.astype(str).tolist())
    logger.debug(
        f'make_saint_dict: Control table melted: {control_melt.shape}: {datetime.now()}')
    logger.debug(
        f'make_saint_dict: Control interactions prepared: {datetime.now()}')
    for uniprot, srow in pd.melt(spc_table, ignore_index=False).replace(0, np.nan).dropna().iterrows():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]
        inter.append([srow['variable'], sgroup, uniprot, str(srow['value'])])
    logger.debug(
        f'make_saint_dict: SPC table interactions prepared: {datetime.now()}')
    for uniprotid in (set(control_table.index.values) | set(spc_table.index.values)):
        try:
            plen: str = str(protein_lenghts_and_names[uniprotid]['length'])
            gname: str = str(protein_lenghts_and_names[uniprotid]['gene name'])
        except KeyError:
            logger.debug(
                f'make_saint_dict: No length found for uniprot: {uniprotid}')
            plen = '200'
            gname = str(uniprotid)
        prey.append([str(uniprotid), plen, gname])
    logger.debug(
        f'make_saint_dict: Preys prepared: {datetime.now()}')

    return {'bait': bait, 'prey': prey, 'int': inter}


def generate_saint_container(input_data_dict, uploaded_controls, additional_controls: list, crapomes: list, db_file: str, do_proximity_filtering: bool, n_controls: int) -> tuple:
    if '["No data"]' in input_data_dict['data tables']['spc']:
        return html.Div(['No spectral count data in input, cannot run SAINT.'])
    logger.debug(
        f'generate_saint_container: preparations started: {datetime.now()}')
    db_conn = db_functions.create_connection(db_file)
    additional_controls = [
        f'control_{ctrl_name[0].lower().replace(" ","_")}' for ctrl_name in additional_controls]
    crapomes = [
        f'crapome_{crap_name[0].lower().replace(" ","_")}' for crap_name in crapomes]
    logger.debug(f'generate_saint_container: DB connected')
    spc_table: pd.DataFrame
    control_table: pd.DataFrame
    spc_table, control_table = prepare_controls(
        input_data_dict, uploaded_controls, additional_controls, db_conn, do_proximity_filtering, n_controls)
    logger.debug(f'generate_saint_container: Controls prepared')
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
    logger.debug(f'generate_saint_container: Protein table retrieved')
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
    logger.debug(
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


def saint_filtering(saint_output_json, bfdr_threshold, crapome_percentage, crapome_fc, do_rescue: bool = False):
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
    logger.debug(f'saint filtering - beginning: {saint_output.shape}')
    logger.debug(
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
        if row['BFDR'] >= bfdr_threshold:
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

    logger.debug(
        f'saint filtering - Preys pass filter: {len(keep_preys)}')
    saint_output['Passes filter'] = keep_col
    logger.debug(
        f'saint filtering - Saint output pass filter: {saint_output["Passes filter"].value_counts()}')
    saint_output['Passes filter with rescue'] = saint_output['Prey'].isin(
        keep_preys)
    logger.debug(
        f'saint filtering - Saint output pass filter with rescue: {saint_output["Passes filter with rescue"].value_counts()}')
    if do_rescue:
        use_col: str = 'Passes filter with rescue'
    else:
        use_col = 'Passes filter'
    filtered_saint_output: pd.DataFrame = saint_output[
        saint_output[use_col]
    ].copy()

    logger.debug(
        f'saint filtering - filtered size: {filtered_saint_output.shape}')
    if 'Bait uniprot' in filtered_saint_output.columns:
        # Immplement multiple baits per file, e.g. for fusions?
        filtered_saint_output = filtered_saint_output[
            filtered_saint_output['Prey'] != filtered_saint_output['Bait uniprot']
        ]
    colorder: list = ['Bait', 'Bait uniprot', 'Prey',
                      'Passes filter', 'Passes filter with rescue', 'AvgSpec']
    colorder.extend(
        [c for c in filtered_saint_output.columns if c not in colorder])
    filtered_saint_output = filtered_saint_output[colorder]
    logger.debug(
        f'saint filtering - bait removed filtered size: {filtered_saint_output.shape}')
    logger.debug(
        f'saint filtering - bait removed filtered size nodupes: {filtered_saint_output.drop_duplicates().shape}')
    return filtered_saint_output.reset_index().drop(columns=['index']).to_json(orient='split')


def saint_counts(filtered_output_json, figure_defaults, replicate_colors):
    count_df: pd.DataFrame = pd.read_json(filtered_output_json, orient='split')['Bait'].\
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
