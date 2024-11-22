from dash import html, dash_table
import pandas as pd
from io import StringIO
from components import db_functions
import numpy as np
import shutil
import os
import tempfile
import sh
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

def count_knowns(saint_output: pd.DataFrame, replicate_colors: dict) -> pd.DataFrame:
    """Counts the number of known interactions for each bait protein.

    Takes a SAINT output DataFrame and processes it to count known interactions,
    adding color coding based on whether interactions are known or not.

    Args:
        saint_output: DataFrame containing SAINT output with at least 'Bait' and 
            'Known interaction' columns
        replicate_colors: Dictionary with structure:
            {
                'contaminant': {'sample groups': {bait_name: color}},
                'non-contaminant': {'sample groups': {bait_name: color}}
            }

    Returns:
        pd.DataFrame: Contains columns:
            - Bait: Name of the bait protein
            - Known interaction: Boolean indicating if row count is for known interactors or not
            - Prey count: Number of prey proteins
            - Color: Color code for visualization

    Example:
        >>> colors = {
        ...     'contaminant': {'sample groups': {'BaitA': 'red'}},
        ...     'non-contaminant': {'sample groups': {'BaitA': 'blue'}}
        ... }
        >>> count_knowns(saint_df, colors)
    """
    saint_output.to_csv('saint_output.csv')
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


def do_network(saint_output_json, plot_height):
    """
    Creates a network plot from filtered SAINT output data.

    Args:
        saint_output_json: JSON string containing SAINT output data
        plot_height: Height of the network plot

    Returns:
        tuple: (
            plot_container: HTML element containing the network plot,
            cyto_elements: List of Cytoscape elements,
            interactions: List of interactions
        )
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    cyto_elements, interactions = network_plot.get_cytoscape_elements_and_ints(saint_output)
    plot_container = network_plot.get_cytoscape_container(cyto_elements, full_height=plot_height)
    return (plot_container, cyto_elements, interactions)


def network_display_data(node_data, int_data, table_height, datatype: str='Cytoscape') -> list:
    """
    Creates a table displaying the connections between nodes in the network plot.

    Args:
        node_data: Dictionary containing node data
        int_data: Dictionary containing interaction data
        table_height: Height of the table
        datatype: Type of data to display ('Cytoscape' or 'visdcc')

    Returns:
        list: List of HTML elements containing the table
    """
    ret = [['Bait','Prey', 'PreyGene','AvgSpec']]
    if datatype == 'Cytoscape':
        for e in node_data['edgesData']: 
            ret.append([e['source'], e["target"]])
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
            style_table={'height': table_height}  # defaults to 500
        )
    ]
    return div_contents

def known_plot(filtered_saint_input_json: str, db_file: str, rep_colors_with_cont: dict, 
               figure_defaults: dict, isoform_agnostic: bool = False) -> tuple:
    """Creates a plot showing known interactions for each bait protein.

    Processes SAINT output data and compares it against known interactions from the database
    to generate a visualization of known vs discovered interactions.

    Args:
        filtered_saint_input_json: JSON string containing filtered SAINT output data
        db_file: Path to the SQLite database file
        rep_colors_with_cont: Dictionary mapping sample groups to their display colors
        figure_defaults: Dictionary containing default figure parameters
        isoform_agnostic: If True, ignores protein isoform differences when matching

    Returns:
        tuple: (
            html.Div containing the plot and related elements,
            JSON string containing the processed SAINT output
        )

    Raises:
        sqlite3.Error: If there is an error accessing the database
        json.JSONDecodeError: If the input JSON is invalid
    """
    logger.warning(f'known_plot - started: {datetime.now()}')
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
    knowns.to_csv('knowns.csv')
    saint_output.to_csv('saint_output_orig.csv')
    # TODO: multibait
    saint_output = pd.merge(
        saint_output,
        knowns,
        left_on=['Bait uniprot', 'Prey'],
        right_on=[upid_a_col, upid_b_col],
        how='left'
    )
    saint_output['Known interaction'] = saint_output['update_time'].notna()
    logger.warning(
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
    return (
        html.Div(
            id='interactomics-saint-known-plot',
            children=[
                html.H4(id='interactomics-known-header',
                        children='High-confidence interactions and identified known interactions'),
                bar_graph.make_graph(
                    'interactomics-saint-filt-int-known-graph',
                    figure_defaults,
                    figure_data,
                    '', color_discrete_map=True, y_name='Prey count', x_label='Bait'
                ),
                legends['known'],
                html.P(known_str),
                html.P(more_known)
            ]
        ),
        saint_output.to_json(orient='split')
    )



def pca(saint_output_data: dict, defaults: dict, replicate_colors: dict) -> tuple:
    """
    Performs Principal Component Analysis (PCA) on SAINT output data.

    Args:
        saint_output_data: Dictionary containing SAINT output data
        defaults: Dictionary containing default parameters
        replicate_colors: Dictionary mapping sample groups to their display colors

    Returns:
        tuple: (
            html.Div containing the PCA plot and related elements,
            JSON string containing the PCA data
        )
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
        gdiv = [
            html.H4(id='interactomics-pca-header', children='SPC PCA'),
            scatter.make_graph(
                'interactomics-pca-plot',
                defaults,
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

def enrich(parameters, saint_output_json: str, chosen_enrichments: list, figure_defaults, keep_all: bool = False, sig_threshold: float = 0.01) -> tuple:
    """
    Enriches SAINT output data using selected enrichment methods.

    Args:
        parameters: Dictionary containing enrichment parameters
        saint_output_json: JSON string containing SAINT output data
        chosen_enrichments: List of selected enrichment methods
        figure_defaults: Dictionary containing default figure parameters
        keep_all: If True, keeps all enriched pathways otherwise, filters by significance. 
        sig_threshold: Significance threshold for filtering

    Returns:
        tuple: (
            list of HTML elements containing the enrichment results,
            dictionary containing enrichment data,
            list of enrichment information
        )
    """
    div_contents:list = []
    enrichment_data: dict = {}
    enrichment_information: list = []
    if len(chosen_enrichments) == 0:
        return (
            div_contents,
            enrichment_data,
            enrichment_information
        )
    e_admin = ea.EnrichmentAdmin()
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    enrichment_names: list
    enrichment_results: list
    enrichment_names, enrichment_results, enrichment_information = e_admin.enrich_all(
        parameters,
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
                cmap = 'dense'
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


def map_intensity(saint_output_json: str, intensity_table_json: str, sample_groups: dict) -> list:
    """
    Maps intensity data to SAINT output data.

    Args:
        saint_output_json: JSON string containing a pd.DataFrame with SAINT output data
        intensity_table_json: JSON string containing a pd.DataFrame with intensity data
        sample_groups: Dictionary mapping sample groups to their display colors

    Returns:
        list: JSON string containing the processed SAINT output
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


def saint_histogram(saint_output_json: str, figure_defaults):
    """
    Creates a histogram plot from SAINT output data.

    Args:
        saint_output_json: JSON string containing a pd.DataFrame with SAINT output data
        figure_defaults: Dictionary containing default figure parameters

    Returns:
        tuple: (
            html.Div containing the histogram plot and related elements,
            JSON string containing the histogram data
        )
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    return (
        histogram.make_figure(saint_output, 'BFDR', '', figure_defaults),
        saint_output.to_json(orient='split')
    )


def add_bait_column(saint_output, bait_uniprot_dict) -> pd.DataFrame:
    """
    Adds bait column to SAINT output data.

    Args:
        saint_output: DataFrame containing SAINT output data
        bait_uniprot_dict: Dictionary mapping baits to their uniprot IDs

    Returns:
        DataFrame: DataFrame containing the processed SAINT output
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

def saint_cmd(saint_input: dict, saint_tempdir: list, session_uid: str):
    """
    Runs SAINT command on SAINT input data.

    Args:
        saint_input: Dictionary containing SAINT input data
        saint_tempdir: a list of directories that points to a temporary directory when joined by os.sep. e.g. ['home','user','tmp'] corresponds to /home/user/tmp
        session_uid: Session ID

    Returns:
        str: Path to the temporary directory
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
        
        print(f'running saint in {temp_dir}, {intfile.name} {preyfile.name} {baitfile.name}: {datetime.now()}')
        sh.SAINTexpressSpc(intfile.name, preyfile.name, baitfile.name, _cwd=temp_dir)
    return temp_dir

def run_saint(saint_input: dict, saint_tempdir: list, session_uid: str, bait_uniprots: dict, cleanup: bool = True) -> str:
    """
    Runs SAINT on SAINT input data.

    Args:
        saint_input: Dictionary containing SAINT input data
        saint_tempdir: List of temporary directory paths
        session_uid: Session ID
        bait_uniprots: Dictionary mapping baits to their uniprot IDs
        cleanup: If True, cleans up the temporary directory after running SAINT

    Returns:
        str: JSON string containing the dataframe of processed SAINT output
    """
    # Can not use logging in this function, since it's called from a long_callback using celery, and logging will lead to a hang.
    # Instead, we can use print statements, and they will show up as WARNINGS in celery log.
    temp_dir: str = ''
    if ('bait' in saint_input) and ('prey' in saint_input):
        temp_dir = saint_cmd(saint_input, saint_tempdir, session_uid)
    failed: bool = not os.path.isfile(os.path.join(temp_dir, 'list.txt'))
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
    return ret


def prepare_crapome(db_conn, crapomes: list) -> pd.DataFrame:
    """
    Prepares crapome data for SAINT analysis.

    Args:
        db_conn: SQLite database connection
        crapomes: List of crapome names

    Returns:
        DataFrame: DataFrame containing the processed crapome data
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

def prepare_controls(input_data_dict, uploaded_controls, additional_controls, db_conn, do_proximity_filtering: bool = False, top_n: int = 30) -> tuple:
    """
    Prepares control data for SAINT analysis.

    Args:
        input_data_dict: Dictionary containing input data
        uploaded_controls: List of uploaded control names
        additional_controls: List of additional control names
        db_conn: SQLite database connection
        do_proximity_filtering: If True, performs proximity filtering
        top_n: Number of top controls to keep

    Returns:
        tuple: (
            DataFrame containing the processed SPC table,
            DataFrame containing the processed control table
        )
    """
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
        
    if (len(controls) > 0) and do_proximity_filtering:
        # groupby to merge possible duplicate columns that are annotated in multiple sets
        # mean grouping should have no effect, since PSM values SHOULD be the same in any case.
        control_table = filter_controls(spc_table, controls, top_n)
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

def filter_controls(spc_table, controls, top_n):
    """
    Filters controls based on how similar to sample runs they are.

    Args:
        spc_table: DataFrame containing SPC table data
        controls: List of control tables
        top_n: Number of top controls to keep

    Returns:
        DataFrame: DataFrame containing the filtered control table
    """
    control_table: pd.DataFrame = pd.concat(controls, axis=1).T.groupby(level=0).mean().T
    controls_ranked_by_similarity: list = matrix_functions.ranked_dist(
        spc_table, control_table)
    control_table = control_table[[s[0] for s in controls_ranked_by_similarity[:top_n]]]
    return control_table

def add_crapome(saint_output_json, crapome_json) -> str:
    """
    Adds crapome data to SAINT output data.

    Args:
        saint_output_json: JSON string containing SAINT output data
        crapome_json: JSON string containing crapome data

    Returns:
        str: JSON string containing the processed SAINT output
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


def make_saint_dict(spc_table, rev_sample_groups, control_table, protein_table) -> dict:
    """
    Creates a dictionary containing SAINT input data.

    Args:
        spc_table: DataFrame containing SPC table data
        rev_sample_groups: Dictionary mapping sample groups to their display colors
        control_table: DataFrame containing control table data
        protein_table: DataFrame containing protein table data

    Returns:
        dict: Dictionary containing the SAINT input data
    """
    protein_lenghts_and_names = {}
    logger.warning(
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
    logger.warning(
        f'make_saint_dict: Baits prepared: {datetime.now()}')
    logger.warning(
        f'make_saint_dict: Control table shape: {control_table.shape}')
    control_melt: pd.DataFrame = pd.melt(
        control_table, ignore_index=False).replace(0, np.nan).dropna().reset_index()
    control_melt['sgroup'] = 'inbuilt_ctrl'
    control_melt = control_melt.reindex(
        columns=['variable', 'sgroup', 'index', 'value'])
    control_melt['value'] = control_melt['value'].astype(int)
    inter.extend(control_melt.values
                 .astype(str).tolist())
    logger.warning(
        f'make_saint_dict: Control table melted: {control_melt.shape}: {datetime.now()}')
    logger.warning(
        f'make_saint_dict: Control interactions prepared: {datetime.now()}')
    for uniprot, srow in pd.melt(spc_table, ignore_index=False).replace(0, np.nan).dropna().iterrows():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]+'_bait'
        inter.append([srow['variable'], sgroup, uniprot, str(int(srow['value']))])
    logger.warning(
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
    logger.warning(
        f'make_saint_dict: Preys prepared: {datetime.now()}')
    return {'bait': bait, 'prey': prey, 'int': inter}

def do_ms_microscopy(saint_output_json:str, db_file: str, figure_defaults: dict, version: str = 'v1.0') -> tuple:
    """
    Performs MS-microscopy analysis on SAINT output data.

    Args:
        saint_output_json: JSON string containing a dataframe with SAINT output data
        db_file: Path to the SQLite database file
        figure_defaults: Dictionary containing default figure parameters
        version: Version of the MS-microscopy analysis

    Returns:
        tuple: (
            html.Div containing the MS-microscopy plot and related elements,
            JSON string containing the MS-microscopy data
        )
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split'
    )
    db_conn = db_functions.create_connection(db_file)
    msmic_reference = db_functions.get_full_table_as_pd(
        db_conn, 'msmicroscopy', index_col='Interaction'
    )
    db_conn.close()
    msmic_results: pd.DataFrame = ms_microscopy.generate_msmic_dataframes(saint_output, msmic_reference, )

    polar_plots: list = [
        (bait, ms_microscopy.localization_graph(f'interactomics-msmic-{bait}',figure_defaults, 'polar', data_row))
        for bait, data_row in msmic_results.iterrows()
    ]
    msmic_heatmap = ms_microscopy.localization_graph(f'interactomics-msmic-heatmap', figure_defaults, 'heatmap', msmic_results)

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
    return(
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

def generate_saint_container(input_data_dict, uploaded_controls, additional_controls: list, crapomes: list, db_file: str, do_proximity_filtering: bool, n_controls: int) -> tuple:
    """
    Generates a container for SAINT analysis.

    Args:
        input_data_dict: Dictionary containing input data
        uploaded_controls: List of uploaded control names
        additional_controls: List of additional control names
        crapomes: List of crapome names
        db_file: Path to the SQLite database file
        do_proximity_filtering: If True, performs proximity filtering
        n_controls: Number of controls

    Returns:
        tuple: (
            html.Div containing the SAINT container and related elements,
            dictionary containing the SAINT input data,
            JSON string containing the processed crapome data
        )
    """
    if '["No data"]' in input_data_dict['data tables']['spc']:
        return (html.Div(['No spectral count data in input, cannot run SAINT.']),{},'')
    logger.warning(
        f'generate_saint_container: preparations started: {datetime.now()}')
    db_conn = db_functions.create_connection(db_file)
    additional_controls = [
        f'control_{ctrl_name[0].lower().replace(" ","_")}' for ctrl_name in additional_controls]
    crapomes = [
        f'crapome_{crap_name[0].lower().replace(" ","_")}' for crap_name in crapomes]
    logger.warning(f'generate_saint_container: DB connected')
    spc_table: pd.DataFrame
    control_table: pd.DataFrame
    spc_table, control_table = prepare_controls(
        input_data_dict, uploaded_controls, additional_controls, db_conn, do_proximity_filtering, n_controls)
    logger.warning(f'generate_saint_container: Controls prepared')
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
    logger.warning(f'generate_saint_container: Protein table retrieved')
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
    logger.warning(
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
    """
    Filters SAINT output data based on BFDR threshold and crapome percentage/crapome fold change.
    Crapome percentage determines, how frequently a given bait should be seen in crapome runs to be considered a contaminant. These will be dropped, unless their spectral count is higher than crapome average*crapome_fc

    Args:
        saint_output_json: JSON string containing a dataframe with SAINT output data
        bfdr_threshold: BFDR threshold for filtering
        crapome_percentage: Crapome percentage for filtering
        crapome_fc: Crapome fold change for filtering
        do_rescue: If True, uses preys that pass the filter in one bait should be rescued in the others, regardless of their spectral count.

    Returns:
        str: JSON string containing the filtered SAINT output
    """
    saint_output: pd.DataFrame = pd.read_json(
        StringIO(saint_output_json), orient='split')
    logger.warning(f'saint filtering - beginning: {saint_output.shape}')
    logger.warning(
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

    logger.warning(
        f'saint filtering - Preys pass filter: {len(keep_preys)}')
    saint_output['Passes filter'] = keep_col
    logger.warning(
        f'saint filtering - Saint output pass filter: {saint_output["Passes filter"].value_counts()}')
    saint_output['Passes filter with rescue'] = saint_output['Prey'].isin(
        keep_preys)
    logger.warning(
        f'saint filtering - Saint output pass filter with rescue: {saint_output["Passes filter with rescue"].value_counts()}')
    if do_rescue:
        use_col: str = 'Passes filter with rescue'
    else:
        use_col = 'Passes filter'
    filtered_saint_output: pd.DataFrame = saint_output[
        saint_output[use_col]
    ].copy()

    logger.warning(
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
    logger.warning(
        f'saint filtering - bait removed filtered size: {filtered_saint_output.shape}')
    logger.warning(
        f'saint filtering - bait removed filtered size nodupes: {filtered_saint_output.drop_duplicates().shape}')
    return filtered_saint_output.reset_index().drop(columns=['index']).to_json(orient='split')

def get_saint_matrix(saint_data_json: str):
    """
    Retrieves the SAINT matrix from SAINT output data.

    Args:
        saint_data_json: JSON string containing SAINT output data

    Returns:
        DataFrame: DataFrame containing the SAINT matrix
    """
    df = pd.read_json(StringIO(saint_data_json),orient='split')
    return df.pivot_table(index='Prey',columns='Bait',values='AvgSpec')

def saint_counts(filtered_output_json, figure_defaults, replicate_colors):
    """
    Counts the number of preys for each bait protein.

    Args:
        filtered_output_json: JSON string containing filtered SAINT output data
        figure_defaults: Dictionary containing default figure parameters
        replicate_colors: Dictionary mapping sample groups to their display colors

    Returns:
        tuple: (
            bar_graph.bar_plot containing the counts,
            JSON string containing the counts
        )
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
