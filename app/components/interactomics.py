from dash import html

import pandas as pd
from components import db_functions
import numpy as np
import os
import subprocess
from components.figures import histogram, bar_graph

def saint_histogram(saint_output_json:str, figure_defaults):
    saint_output:pd.DataFrame = pd.read_json(saint_output_json)
    return histogram(saint_output,'BFDR','BFDR distribution',figure_defaults)

def run_saint(saint_input: dict, saint_path:str, error_log_file: str, session_uid:str, cleanup: bool = True) -> str:
    temp_dir: str = os.path.join(saint_path[:-1])
    saint_cmd: str = os.path.join(temp_dir, saint_path[-1])
    temp_dir = os.path.join(temp_dir, session_uid)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    paths: list = [ os.path.join(temp_dir, x) for x in 'inter.dat,prey.dat,bait.dat'.split(',')]
    with open(paths[0],'w',encoding='utf-8') as fil:
        fil.write('\n'.join([
            '\t'.join(x) for x in saint_input['int']
        ]))
    with open(paths[1],'w',encoding='utf-8') as fil:
        fil.write('\n'.join([
            '\t'.join(x) for x in saint_input['prey']
        ]))
    with open(paths[2],'w',encoding='utf-8') as fil:
        fil.write('\n'.join([
            '\t'.join(x) for x in saint_input['bait']
        ]))
    try:
        subprocess.check_output([saint_cmd], stderr=subprocess.STDOUT, cwd=temp_dir, text=True)
        failed: bool = os.path.isfile(os.path.join(temp_dir, 'list.txt'))
    except subprocess.CalledProcessError:
        failed = True
    if failed:
        with open(error_log_file,'a',encoding='utf-8') as fil:
            fil.write(f'SAINT run failed: {session_uid}. Cleanup not performed.')
        ret: str = 'SAINT failed. Can not proceed.'
    else:
        ret = pd.read_csv(os.path.join(temp_dir, 'list.txt'), sep='\t').to_json(orient='split')
    
    return ret

def prepare_crapome(db_conn, crapomes: list) -> pd.DataFrame:
    crapome_tables: list = [
        db_functions.get_full_table_as_pd(db_conn, tablename,index_col='protein_id') for tablename in crapomes
    ]
    crapome_table:pd.DataFrame = pd.concat(
        [
            crapome_tables[i][['frequency','spc_avg']].\
                rename(columns={
                    'frequency': f'{table_name}_frequency',
                    'spc_avg': f'{table_name}_spc_avg'
                    })
            for i, table_name in enumerate(crapomes)
        ]
    )
    crapome_freq_cols: list = [c for c in crapome_table.columns if '_frequency' in c]
    crapome_table['Max crapome frequency'] = crapome_table[crapome_freq_cols].max(axis=1)
    return crapome_table


def prepare_controls(input_data_dict, uploaded_controls, additional_controls, db_conn) -> tuple:
    sample_groups: dict = input_data_dict['sample groups']['norm']
    spc_table: pd.DataFrame = pd.read_json(
        input_data_dict['data tables']['spc'], orient='split')
    controls: list = [
        db_functions.get_full_table_as_pd(db_conn, tablename) for tablename in additional_controls
    ]
    control_cols: list = []
    for cg in uploaded_controls: 
        control_cols.extend(sample_groups[cg])
    controls.append(spc_table[control_cols])
    spc_table = spc_table[[c for c in spc_table.columns if c not in control_cols]]
    control_table: pd.DataFrame = pd.concat(controls)

    return (spc_table, control_table)

def add_crapome(saint_output_json, crapome_json) -> str:
    if 'Saint failed.' in saint_output_json:
        return saint_output_json
    saint_output: pd.DataFrame = pd.read_json(saint_output_json,orient='split')
    return pd.merge(
        saint_output,
        pd.read_json(crapome_json, orient='split'),
        left_on='Prey',
        right_index=True,
        how='left'
    ).to_json(orient='split')

def make_saint_dict(spc_table, rev_sample_groups, control_table,protein_table) -> dict:
    protein_lenghts_and_names = {}
    for _,row in protein_table.iterrows():
        protein_lenghts_and_names[row['uniprot_id']] = {'length': row['length'], 'gene name': row['gene_name']}
    
    bait:list = []
    prey:list = []
    inter:list = []
    for col in spc_table.columns:
        bait.append([col, rev_sample_groups[col], 'T'])
    for col in control_table.columns:
        if col in rev_sample_groups:
            bait.append([col, rev_sample_groups[col], 'C'])
        else:
            bait.append([col, 'inbuilt_ctrl', 'C'])
    for uniprot, srow in pd.melt(control_table,ignore_index=False).replace(0,np.nan).dropna():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in rev_sample_groups:
            sgroup = rev_sample_groups[srow['variable']]
        inter.append([srow['variable'], sgroup, uniprot, str(srow['value'])])
    for uniprot, srow in pd.melt(control_table,ignore_index=False).replace(0,np.nan).dropna():
        sgroup: str = 'inbuilt_ctrl'
        if srow['variable'] in spc_table:
            sgroup = rev_sample_groups[srow['variable']]
        inter.append([srow['variable'], sgroup, uniprot, str(srow['value'])])
    for uniprotid in (set(control_table.index.values) | set(spc_table.index.values)):
        try:
            plen: str = str(protein_lenghts_and_names[uniprotid]['length'])
        except KeyError:
            print('NO LENGTH FOUND', uniprotid)
            plen = '200'
        prey.append([uniprotid, plen, protein_lenghts_and_names[uniprot]['gene name']])
    
    return {'bait': bait, 'prey': prey, 'int': inter}

def generate_saint_container(input_data_dict, uploaded_controls, additional_controls: list, crapomes: list, db_file) -> html.Div:
    if '["No data"]' in input_data_dict['data tables']['spc']:
        return html.Div(['No spectral count data in input, cannot run SAINT.'])
    db_conn = db_functions.create_connection(db_file)
    additional_controls = [f'control_{ctrl_name.lower()}' for ctrl_name in additional_controls]
    crapomes = [f'crapome_{crap_name.lower()}' for crap_name in crapomes]
    spc_table, control_table = prepare_controls(input_data_dict, uploaded_controls, additional_controls, db_conn)
    protein_list: list = list(set(spc_table.index.values) | set(control_table.index))

    protein_table: pd.DataFrame = db_functions.get_from_table(
        db_conn,
        'proteins',
        select_col = [
            'uniprot_id',
            'length',
            'gene_name'
        ],
        as_pandas = True
    )
    protein_table = protein_table[protein_table['uniprot_id'].isin(protein_list)]
    crapome: pd.DataFrame = prepare_crapome(db_conn, crapomes)
    db_conn.close()

    saint_dict: dict = make_saint_dict(spc_table, input_data_dict['sample groups']['rev'], control_table, protein_table)

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
    saint_output:pd.DataFrame = pd.read_json(saint_output_json)
    crapome_columns: list = []
    for column in saint_output.columns:
        if '_frequency' in column:
            crapome_columns.append((column, column.replace('_frequency','_spc_avg')))
    keep_col: list = []
    for _, row in saint_output.iterrows():
        keep: bool = True
        if row['BFDR']>= bfdr_threshold:
            keep = False
        else:
            if row['Max crapome frequency'] > crapome_percentage:
                for freq_col, fc_col in crapome_columns:
                    if row[freq_col] >= crapome_percentage:
                        if row[fc_col] <= crapome_fc:
                            keep = False
                            break
        keep_col.append(keep)
    filtered_saint_output: pd.DataFrame = saint_output[keep_col]
    if 'Bait uniprot' in filtered_saint_output.columns:
        filtered_saint_output = filtered_saint_output[
            filtered_saint_output['Prey'] != filtered_saint_output['Bait uniprot']
        ]
    return filtered_saint_output.reset_index().drop(columns=['index'])

def saint_counts(filtered_output_json, figure_defaults, replicate_colors): 
    count_df: pd.DataFrame = pd.read_json(filtered_output_json)['Bait'].\
        value_counts().\
        to_frame(name='Prey count')
    count_df['Color'] = [
        replicate_colors['non-contaminant']['samples'][index] for index in count_df.index.values
    ]
    return bar_graph.bar_plot(
        figure_defaults, 
        count_df,
        title='Filtered Prey counts per bait'
    )