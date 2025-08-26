import sqlite3
import os
from components.tools import utils


def control_or_crapome_table(control_or_crapome: str = 'crapome') -> str:
    overall_columns = [f'{control_or_crapome}_set',f'{control_or_crapome}_set_name','runs','is_disabled','is_default',f'{control_or_crapome}_table_name','version_update_time','prev_version']
    overall_types = ['TEXT PRIMARY KEY','TEXT NOT NULL','INTEGER NOT NULL','INTEGER NOT NULL','INTEGER NOT NULL','TEXT NOT NULL','TEXT NOT NULL','TEXT']

    table_create_str =  [
            f'CREATE TABLE IF NOT EXISTS {control_or_crapome}_sets (',
        ]
    for i, c in enumerate(overall_columns):
        table_create_str.append(f'    {c} {overall_types[i]},',)
    table_create_str = '\n'.join(table_create_str).strip(',')
    table_create_str += '\n);'

    return table_create_str
    
def prot_table() -> str:
    prot_cols = [
        'uniprot_id',
        'is_reviewed',
        'gene_name',
        'entry_name',
        'all_gene_names',
        'organism',
        'length',
        'sequence',
        'is_latest',
        'entry_source',
        'version_update_time',
        'prev_version'
    ]
    prot_exts = [
        'TEXT PRIMARY KEY',
        'INTEGER NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'INTEGER NOT NULL',
        'TEXT NOT NULL',
        'INTEGER NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT'
    ]

    prot_table_str =  [
            f'CREATE TABLE IF NOT EXISTS  proteins (',
        ]
    for i, c in enumerate(prot_cols):
        prot_table_str.append(f'    {c} {prot_exts[i]},',)
    prot_table_str = '\n'.join(prot_table_str).strip(',')
    prot_table_str += '\n);'
    return prot_table_str

def contaminant_table() -> str:
    cont_cols = [
        'uniprot_id',
        'is_reviewed',
        'gene_name',
        'entry_name',
        'all_gene_names',
        'organism',
        'length',
        'sequence',
        'entry_source',
        'contamination_source',
        'version_update_time',
        'prev_version'
    ]
    cont_exts = [
        'TEXT PRIMARY KEY',
        'INTEGER NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'INTEGER NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
        'TEXT NOT NULL',
    ]
    cont_table_str =  [
        f'CREATE TABLE IF NOT EXISTS  contaminants (',
    ]
    for i, c in enumerate(cont_cols):
        cont_table_str.append(f'    {c} {cont_exts[i]},',)
    cont_table_str = '\n'.join(cont_table_str).strip(',')
    cont_table_str += '\n);'
    return cont_table_str

def int_table() -> str:
    inttable_cols = [
        'interaction TEXT PRIMARY KEY',
        'uniprot_id_a TEXT NOT NULL',
        'uniprot_id_b TEXT NOT NULL',
        'uniprot_id_a_noiso TEXT NOT NULL',
        'uniprot_id_b_noiso TEXT NOT NULL',
        'source_database TEXT NOT NULL',
        'isoform_a TEXT',
        'isoform_b TEXT',
        'organism_interactor_a TEXT',
        'organism_interactor_b TEXT',
        'experimental_role_interactor_a TEXT',
        'interaction_detection_method TEXT',
        'publication_identifier TEXT',
        'biological_role_interactor_b TEXT',
        'annotation_interactor_a TEXT',
        'confidence_value TEXT',
        'interaction_type TEXT',
        'experimental_role_interactor_b TEXT',
        'annotation_interactor_b TEXT',
        'biological_role_interactor_a TEXT',
        'publication_count TEXT',
        'notes TEXT',
        'version_update_time TEXT',
        'prev_version TEXT'
    ]
    inttable_create = ['CREATE TABLE IF NOT EXISTS known_interactions (']
    for col in inttable_cols:
        inttable_create.append(f'    {col},')
    inttable_create = '\n'.join(inttable_create).strip(',')
    inttable_create += '\n);'
    return inttable_create

def ms_runs_table() -> str:
    mstable_create = ['CREATE TABLE IF NOT EXISTS ms_runs (']
    ms_cols = [
        'run_id TEXT PRIMARY KEY',
        'run_name TEXT NOT NULL',
        'sample_name TEXT NOT NULL',
        'file_name TEXT NOT NULL',
        'run_time TEXT NOT NULL',
        'run_date TEXT NOT NULL',
        'instrument TEXT NOT NULL',
        'author TEXT NOT NULL',
        'sample_type TEXT NOT NULL',
        'run_type TEXT NOT NULL',
        'lc_method TEXT NOT NULL',
        'ms_method TEXT NOT NULL',
        'num_precursors INTEGER NOT NULL',
        'bait TEXT',
        'bait_uniprot TEXT',
        'bait_mutation TEXT',
        'chromatogram_max_time INTEGER NOT NULL',
        'cell_line_or_material TEXT',
        'project TEXT',
        'author_notes TEXT',
        'bait_tag TEXT',
        'version_update_time TEXT',
        'prev_version TEXT'
    ]
    keytypes = {
        'auc': 'REAL NOT NULL',
        'intercepts': 'INTEGER NOT NULL',
        'mean_intensity': 'INTEGER NOT NULL',
        'max_intensity': 'INTEGER NOT NULL',
        'json': 'TEXT NOT NULL',
        'trace': 'TEXT NOT NULL', 
        'intercept_json': 'TEXT NOT NULL',
        'json_smooth': 'TEXT NOT NULL',
        'trace_smooth': 'TEXT NOT NULL', 
    }
    for typ in ['MSn_filtered','TIC','MSn_unfiltered']:
        for key in ['auc','intercepts','mean_intensity','max_intensity', 'json','trace', 'intercept_json', 'json_smooth', 'trace_smooth']:
            ms_cols.append(f'{typ.lower()}_{key.lower()} {keytypes[key]}')
            
    for col in ms_cols:
        mstable_create.append(f'    {col},')
    mstable_create = '\n'.join(mstable_create).strip(',')
    mstable_create += '\n);'
    return mstable_create

def msmicroscopy_table() -> str:
    msmictable_create = ['CREATE TABLE IF NOT EXISTS msmicroscopy (']
    msmictable_cols = [
        'Interaction TEXT PRIMARY KEY',
        'Bait TEXT NOT NULL',
        'Prey TEXT NOT NULL',
        'Bait_norm REAL NOT NULL',
        'Bait_sumnorm REAL NOT NULL',
        'Loc TEXT NOT NULL',
        'Unique_to_loc REAL NOT NULL',
        'Loc_norm REAL NOT NULL',
        'Loc_sumnorm REAL NOT NULL',
        'MSMIC_version TEXT NOT NULL',
        'Version_update_time TEXT NOT NULL',
        'Prev_version TEXT',
    ]
    for col in msmictable_cols:
        msmictable_create.append(f'    {col},')
    msmictable_create = '\n'.join(msmictable_create).strip(',')
    msmictable_create += '\n);'
    return msmictable_create

def common_proteins_table() -> str:
    comtable_create = ['CREATE TABLE IF NOT EXISTS common_proteins (']
    com_cols = [
        'uniprot_id TEXT PRIMARY KEY',
        'gene_name TEXT',
        'entry_name TEXT',
        'all_gene_names TEXT',
        'organism TEXT',
        'protein_type TEXT NOT NULL',
        'version_update_time TEXT NOT NULL',
        'prev_version TEXT'
    ]
    for col in com_cols:
        comtable_create.append(f'    {col},')
    comtable_create = '\n'.join(comtable_create).strip(',')
    comtable_create += '\n);'
    return comtable_create

def run_table_generation(func, parameters, timestamp, time_format=None):
    if time_format:
        return func(parameters, timestamp, time_format)
    return func(parameters, timestamp)

def generate_database(database_filename: str) -> None:
    table_creations = [
        control_or_crapome_table('crapome'),
        control_or_crapome_table('control'),
        prot_table(),
        contaminant_table(),
        int_table(),
        ms_runs_table(),
        msmicroscopy_table(),
        common_proteins_table()
    ]
    conn = sqlite3.connect(database_filename)
    cursor = conn.cursor()
    for create_table_str in table_creations:
        cursor.execute(create_table_str)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parameters = utils.read_toml('parameters.toml')
    dbfile = os.path.join(*parameters['Data paths']['Database file'])
    generate_database(dbfile) 