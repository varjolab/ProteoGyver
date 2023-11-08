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
from components import EnrichmentAdmin as ea
from dash_bootstrap_components import Card, CardBody, Tab, Tabs


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
    upid_a_col: str = 'uniprot_id_a'
    upid_b_col: str = 'uniprot_id_b'
    if isoform_agnostic:
        upid_a_col += '_noiso'
        upid_b_col += '_noiso'
    saint_output: pd.DataFrame = pd.read_json(
        filtered_saint_input_json, orient='split')
    db_conn = db_functions.create_connection(db_file)
    col_order: list = list(saint_output.columns)
    saint_output = pd.merge(
        saint_output,
        db_functions.get_from_table_by_list_criteria(
            db_conn, 'known_interactions', upid_a_col, list(
                saint_output['Bait'].unique())
        ),
        left_on=['Bait', 'Prey'],
        right_on=[upid_a_col, upid_b_col],
        how='left'
    )
    saint_output['Known interaction'] = saint_output['update_time'].notna()
    db_conn.close()
    col_order.append('Known interaction')
    col_order.append(
        'Following columns are information about known interactions.')
    saint_output[col_order[-1]] = np.nan
    col_order.extend([c for c in saint_output.columns if c not in col_order])
    saint_output = saint_output[col_order]
    figure_data: pd.DataFrame = count_knowns(
        saint_output, rep_colors_with_cont)
    figure_data.index = figure_data['Bait']

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
                html.P(
                    f'Known counts per bait: {", ".join([b + ": " + str(figure_data.loc[b]["Prey count"]) for b in figure_data.index])}')
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
                html.H4(id='interactomics-pca-header', children='PCA'),
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
            graph = 'Nothing enriched.'
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
        enrichment_tab: Card = Card(
            CardBody(
                [
                    html.P(f'{enrichment_names[i]} heatmap'),
                    graph,
                    html.P(f'{enrichment_names[i]} data table'),
                    table
                ]
            ),
            className="mt-3",
        )

        tablist.append(
            Tab(
                enrichment_tab, label=enrichment_names[i]
            )
        )
    return (
        Tabs(
            id='interactomics-enrichment-tabs',
            children=tablist
        ),
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
    saint_output['Bait uniprot'] = [bait_uniprot_dict[bait]
                                    for bait in saint_output['Bait'].values]
    return saint_output


def run_saint(saint_input: dict, saint_path: list, error_log_file: str, session_uid: str, bait_uniprots: dict, cleanup: bool = True) -> str:
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
    with open(paths[2].replace('bait.dat', 'saint_data.json'), 'w') as fil:
        json.dump(saint_input, fil)
    try:
        subprocess.check_output(
            [saint_cmd], stderr=subprocess.STDOUT, cwd=temp_dir, text=True)
        failed: bool = not os.path.isfile(os.path.join(temp_dir, 'list.txt'))
    except subprocess.CalledProcessError as e:
        print(e)
        failed = True
    if failed:
        with open(error_log_file, 'a', encoding='utf-8') as fil:
            fil.write(
                f'SAINT run failed: {session_uid}. Cleanup not performed.\n')
        ret: str = 'SAINT failed. Can not proceed.'
    else:
        ret = add_bait_column(pd.read_csv(os.path.join(
            temp_dir, 'list.txt'), sep='\t'), bait_uniprots).to_json(orient='split')
        if cleanup:
            try:
                shutil.rmtree(temp_dir)
            except PermissionError as e:
                with open(error_log_file, 'a', encoding='utf-8') as fil:
                    fil.write(f'Could not clean up after SAINT run:\n{e}\n')
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
        ]
    )
    crapome_freq_cols: list = [
        c for c in crapome_table.columns if '_frequency' in c]
    crapome_table['Max crapome frequency'] = crapome_table[crapome_freq_cols].max(
        axis=1)
    return crapome_table


def prepare_controls(input_data_dict, uploaded_controls, additional_controls, db_conn) -> tuple:
    sample_groups: dict = input_data_dict['sample groups']['norm']
    spc_table: pd.DataFrame = pd.read_json(
        input_data_dict['data tables']['spc'], orient='split')
    controls = []
    for control_name in additional_controls:
        ctable = db_functions.get_full_table_as_pd(
            db_conn, control_name, index_col='protein_id')
        ctable.drop(
            columns=[c for c in ctable.columns if c != 'spc_avg'], inplace=True)
        ctable.rename(columns={'spc_avg': control_name}, inplace=True)
        ctable.index.name = ''
        controls.append(ctable)
    control_cols: list = []
    for cg in uploaded_controls:
        control_cols.extend(sample_groups[cg])
    controls.append(spc_table[control_cols])
    spc_table = spc_table[[
        c for c in spc_table.columns if c not in control_cols]]
    control_table: pd.DataFrame = pd.concat(controls)

    return (spc_table, control_table)


def add_crapome(saint_output_json, crapome_json) -> str:
    if 'Saint failed.' in saint_output_json:
        return saint_output_json
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
    return pd.merge(
        saint_output,
        pd.read_json(crapome_json, orient='split'),
        left_on='Prey',
        right_index=True,
        how='left'
    ).to_json(orient='split')


def make_saint_dict(spc_table, rev_sample_groups, control_table, protein_table) -> dict:
    protein_lenghts_and_names = {}
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
    for uniprot, srow in pd.melt(control_table, ignore_index=False).replace(0, np.nan).dropna().iterrows():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]
        inter.append([srow['variable'], sgroup, uniprot, str(srow['value'])])
    for uniprot, srow in pd.melt(spc_table, ignore_index=False).replace(0, np.nan).dropna().iterrows():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]
        inter.append([srow['variable'], sgroup, uniprot, str(srow['value'])])
    for uniprotid in (set(control_table.index.values) | set(spc_table.index.values)):
        try:
            plen: str = str(protein_lenghts_and_names[uniprotid]['length'])
            gname: str = protein_lenghts_and_names[uniprotid]['gene name']
        except KeyError:
            print('NO LENGTH FOUND', uniprotid)
            plen = '200'
            gname = uniprotid
        prey.append([uniprotid, plen, gname])

    return {'bait': bait, 'prey': prey, 'int': inter}


def generate_saint_container(input_data_dict, uploaded_controls, additional_controls: list, crapomes: list, db_file) -> tuple:
    if '["No data"]' in input_data_dict['data tables']['spc']:
        return html.Div(['No spectral count data in input, cannot run SAINT.'])
    db_conn = db_functions.create_connection(db_file)
    additional_controls = [
        f'control_{ctrl_name[0].lower().replace(" ","_")}' for ctrl_name in additional_controls]
    crapomes = [
        f'crapome_{crap_name[0].lower().replace(" ","_")}' for crap_name in crapomes]
    spc_table: pd.DataFrame
    control_table: pd.DataFrame
    spc_table, control_table = prepare_controls(
        input_data_dict, uploaded_controls, additional_controls, db_conn)
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
    protein_table = protein_table[protein_table['uniprot_id'].isin(
        protein_list)]
    if len(crapomes) > 0:
        crapome: pd.DataFrame = prepare_crapome(db_conn, crapomes)
    else:
        crapome = pd.DataFrame()
    db_conn.close()

    saint_dict: dict = make_saint_dict(
        spc_table, input_data_dict['sample groups']['rev'], control_table, protein_table)
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


def saint_filtering(saint_output_json, bfdr_threshold, crapome_percentage, crapome_fc):
    saint_output: pd.DataFrame = pd.read_json(
        saint_output_json, orient='split')
    crapome_columns: list = []
    for column in saint_output.columns:
        if '_frequency' in column:
            crapome_columns.append(
                (column, column.replace('_frequency', '_spc_avg')))
    keep_col: list = []
    for _, row in saint_output.iterrows():
        keep: bool = True
        if row['BFDR'] >= bfdr_threshold:
            keep = False
        elif 'Max crapome frequency' in saint_output.columns:
            if row['Max crapome frequency'] > crapome_percentage:
                for freq_col, fc_col in crapome_columns:
                    if row[freq_col] >= crapome_percentage:
                        if row[fc_col] <= crapome_fc:
                            keep = False
                            break
        keep_col.append(keep)
    filtered_saint_output: pd.DataFrame = saint_output[
        saint_output['Prey'].isin(saint_output[keep_col]['Prey'])
    ]
    if 'Bait uniprot' in filtered_saint_output.columns:
        print('removing baits')
        print(filtered_saint_output.shape)

        # Immplement multiple baits per file, e.g. for fusions?
        filtered_saint_output = filtered_saint_output[
            filtered_saint_output['Prey'] != filtered_saint_output['Bait uniprot']
        ]
        print(filtered_saint_output.shape)
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
