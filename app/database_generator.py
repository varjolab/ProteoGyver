import re
import shutil
import sqlite3
import os
import pandas as pd
import json
import numpy as np
from components import text_handling
from components.api_tools.annotation import intact
from components.api_tools.annotation import biogrid
from components.api_tools.annotation import uniprot
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator
import pandas as pd
import numpy as np
from datetime import datetime

def do_crapome_or_controls(parameters: dict, sets: dict, crapome:pd.DataFrame, col_map: dict, timestamp: str, control_or_crapome: str = 'crapome') -> tuple[list, list]:
    overall_columns = [f'{control_or_crapome}_set',f'{control_or_crapome}_set_name','runs','is_disabled','is_default',f'{control_or_crapome}_table_name','version_update_time','prev_version']
    overall_types = ['TEXT PRIMARY KEY','TEXT NOT NULL','INTEGER NOT NULL','INTEGER NOT NULL','INTEGER NOT NULL','TEXT NOT NULL','TEXT NOT NULL','TEXT']
    detailed_columns = parameters['Control and crapome db detailed columns']
    detailed_types = parameters['Control and crapome db detailed types']
    entries = []
    tables = {}
    for setname, setcols in sets.items():
        if control_or_crapome == 'control':
            if setname in parameters['Sets not included in controls']:
                continue
        all_cols = ['PROTID']
        defa = 1
        for regex in parameters['Default control disabled regex list']:
            if re.search(regex, setname):
                defa = 0
                break
        tablename = f'{control_or_crapome}_{setname}'.lower().replace(' ','_')
        for sc in setcols:
            all_cols.extend(col_map[sc])
        all_cols = sorted(list(set(all_cols)))
        set_df = crapome[all_cols]
        setname = f'{setname} ({len(all_cols)} runs)'
        set_df.index = set_df['PROTID'] # type: ignore
        set_df = set_df.drop(columns=['PROTID']).replace(0,np.nan).dropna(how='all',axis=0).dropna(how='all',axis=1)
        nruns = set_df.shape[1]
        set_data = []
        for protid, row in set_df.iterrows():
            stdval = row.std()
            if pd.isna(stdval):
                stdval = -1
            set_data.append([protid, row.notna().sum(), row.notna().sum()/nruns,row.sum(), row.mean(), row.min(), row.max(), stdval])
        tables[tablename] = (set_df, pd.DataFrame(columns=detailed_columns, data=set_data))
        entries.append([tablename, setname, nruns, 0, defa, tablename, timestamp, -1])
    
    table_create_str =  [
            f'CREATE TABLE IF NOT EXISTS {control_or_crapome}_sets (',
        ]
    for i, c in enumerate(overall_columns):
        table_create_str.append(f'    {c} {overall_types[i]},',)
    table_create_str = '\n'.join(table_create_str).strip(',')
    table_create_str += '\n);'

    table_create_sql = [table_create_str]
    insert_sql = []
    for vals in entries:
        tablename = vals[0]
        detailed, overall = tables[tablename]

        detailed_tablename = tablename

        if control_or_crapome == 'control':
            tablename = f'{tablename}_overall'
        create_str = [
            f'CREATE TABLE IF NOT EXISTS  {tablename} (',
        ]
        for i, c in enumerate(overall.columns):
            create_str.append(f'    {c} {detailed_types[i]},',)
        create_str = '\n'.join(create_str).strip(',')
        create_str += '\n);'
        table_create_sql.append(create_str)
        add_str = [f'INSERT INTO {control_or_crapome}_sets ({", ".join(overall_columns)}) VALUES ({", ".join(["?" for _ in overall_columns])})', vals]
        insert_sql.append(add_str)
        for _, row in overall.iterrows():
            add_str = [f'INSERT INTO {tablename} ({", ".join(overall.columns)}) VALUES ({", ".join(["?" for _ in overall.columns])})', tuple(row.values)]
            insert_sql.append(add_str)

        if control_or_crapome == 'control':
            tablename = detailed_tablename
            detailed.rename(
                columns={
                    c: 'CS_'+text_handling.replace_accent_and_special_characters(c,'_')
                    for c in detailed.columns
                },
                inplace=True
            )
            create_str = [
                f'CREATE TABLE IF NOT EXISTS  {tablename} (',
            ]
            detailed = detailed.reset_index()
            detailed_control_types = ['TEXT PRIMARY KEY']
            for c in detailed.columns[1:]:
                detailed_control_types.append('REAL')
            for i, c in enumerate(detailed.columns):
                create_str.append(f'    {c} {detailed_control_types[i]},',)
            create_str = '\n'.join(create_str).strip(',')
            create_str += '\n);'
            table_create_sql.append(create_str)
            for _, row in detailed.iterrows():
                add_str = [f'INSERT INTO {tablename} ({", ".join(detailed.columns)}) VALUES ({", ".join(["?" for _ in detailed.columns])})', tuple(row.values)]
                insert_sql.append(add_str)
    return table_create_sql, insert_sql

def crapome_and_controls(parameters: dict, timestamp: str) -> tuple[list, list]:
    sets = parameters['Control and crapome sets']
    datadir = os.path.join(*parameters['Database build files directory'])
    crapome = pd.read_csv(os.path.join(datadir,parameters['Crapome table']),sep='\t')
    controls = pd.read_csv(os.path.join(datadir,parameters['Control table']),sep='\t')

    additional_controls_dir = os.path.join(datadir,'gfp control')
    data_cols= {}
    for _, file_path in parameters['Control and crapome column mapping'].items():
        with open(os.path.join(*file_path),'r') as f:
            for key, value in json.load(f).items():
                data_cols[key] = value
    for setdir in os.listdir(additional_controls_dir):
        sampleinfo = pd.read_excel(os.path.join(additional_controls_dir, setdir, 'Sample_Information.xlsx'))
        data = pd.read_csv(os.path.join(additional_controls_dir, setdir, 'reprint.spc.tsv'),sep='\t', index_col='PROTID').drop(columns=['GENEID','PROTLEN'])
        data = data[(data.index.notna()) & (~data.index.isin({'na','NA'}))].astype(int).replace('na',np.nan).replace(0, np.nan)
        data = data.reset_index()
        namecol, runidcol, samplename, _, expcol, pgsetnamecol = sampleinfo.columns
        for exp in sampleinfo[expcol].unique():
            pgsetname = sampleinfo[sampleinfo[expcol]==exp][pgsetnamecol].iloc[0]
            setname = f'{setdir} {exp}'
            sets.setdefault(pgsetname, []).append(setname)
            data_cols[setname] = list(data.columns)
        crapome = crapome.merge(data,left_on='PROTID',right_on='PROTID',how='outer')
        controls = controls.merge(data,left_on='PROTID',right_on='PROTID',how='outer')
    crapome_sql, crapome_insert_sql = do_crapome_or_controls(parameters, sets, crapome, data_cols, timestamp, 'crapome')
    control_sql, control_insert_sql = do_crapome_or_controls(parameters, sets, controls, data_cols, timestamp, 'control')
    print('crapome_and_controls done')
    return (crapome_sql+control_sql, crapome_insert_sql+control_insert_sql)


def stream_flattened_rows(df: pd.DataFrame) -> Iterator[dict]:
    if 'interaction' in df.columns:
        df.set_index('interaction', inplace=True)

    for interaction, row in df.iterrows():
        out_row: dict = {'interaction': interaction}
        for col in df.columns:
            if pd.isna(row[col]):
                continue
            values = [v.strip(';') for v in str(row[col]).split(';') if v]
            key = col
            if key in out_row:
                out_row[key].update(values)
            else:
                out_row[key] = set(values)
        yield out_row

def get_merg_chunk(organisms: set | None, timestamp: str, L: str, inttable_cols: list, tmpdir: str) -> None:
    
    def stream_cleaned_rows():
        sources = [
            intact.get_latest(organisms, subset_letter=L),   # type: ignore
            biogrid.get_latest(organisms, subset_letter=L),  # type: ignore
        ]
        merged = {}
        for df in sources:
            for row in stream_flattened_rows(df):
                interaction = row.pop('interaction')
                merged.setdefault(interaction, {})
                for k, v in row.items():
                    merged[interaction].setdefault(k, set()).update(v)

        for interaction, fields in merged.items():
            row_data = {
                'interaction': interaction,
                **{k: ';'.join(sorted(v)).strip(';') for k, v in fields.items()}
            }

            for badval in ['nan', '', '-', 'None']:
                for k, v in row_data.items():
                    if v == badval:
                        row_data[k] = None

            pubid = row_data.get('publication_identifier') or ''
            row_data['publication_count'] = len(pubid.strip(';').split(';')) if pubid else 0

            row_data['version_update_time'] = timestamp
            row_data['prev_version'] = -1

            required = [
                'uniprot_id_a', 'uniprot_id_b',
                'uniprot_id_a_noiso', 'uniprot_id_b_noiso',
                'isoform_a', 'isoform_b'
            ]
            if any(row_data.get(k) in [None, '', np.nan] for k in required):
                continue

            yield row_data

    add_str = f'INSERT INTO known_interactions ({", ".join([c.split()[0] for c in inttable_cols])}) VALUES ({", ".join(["?" for _ in inttable_cols])})'

    data = []
    for row in stream_cleaned_rows():
        out = []
        for c in inttable_cols:
            val = row.get(c.split()[0], '')
            out.append(val)
        data.append(out)

    if data:
        with open(os.path.join(tmpdir, f'{L}.json'), 'w') as fil:
            json.dump({'add_str': add_str, 'data': data}, fil)

def write_int_insert_sql(inttable_cols: list, organisms: set|None, timestamp: str, tmpdir: str, ncpu: int) -> None:
    get_chunks = sorted(list(set(intact.get_available()) | set(biogrid.get_available())))
    num_cpus = ncpu
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        for L in get_chunks:
            futures.append(executor.submit(get_merg_chunk, organisms, timestamp, L, inttable_cols, tmpdir))
        for future in as_completed(futures):
            future.result()
    
def prot_knownint_and_contaminant_table(parameters: dict, timestamp: str, tmpdir:str, ncpu: int, organisms: set|None = None) -> tuple[list, list]:
    table_create_sql = []
    insert_sql = []
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
    table_create_sql.append(prot_table_str)
    

    uniprot_df = uniprot.download_uniprot_for_database(organisms = organisms)
    #uniprot_df.to_csv('uniprot_df_full.tsv',sep='\t',index=True)
    #uniprot_df = pd.read_csv('uniprot_df_full.tsv',sep='\t',index_col='Entry')
    uniprots = set(uniprot_df.index.values)
    for protid, row in uniprot_df.iterrows():
        gn = row['Gene names (primary)']
        if pd.isna(gn):
            gn = row['Entry name']
        gns = row['Gene names']
        if pd.isna(gns):
            gns = row['Entry name']
        row = row.fillna('')
        data = [
            protid,
            int(row['Reviewed']=='reviewed'),
            gn,
            row['Entry name'],
            gns,
            row['Organism'],
            row['Length'],
            row['Sequence'],
            1,
            'uniprot_initial_download',
            timestamp,
            -1
        ]
        add_str = f'INSERT INTO proteins ({", ".join(prot_cols)}) VALUES ({", ".join(["?" for _ in prot_cols])})'
        insert_sql.append([add_str, data])

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

    datadir = os.path.join(*parameters['Database build files directory'])
    conts = pd.read_csv(os.path.join(datadir,parameters['Contaminants table']),sep='\t')
    conts = conts[~conts['Uniprot ID'].isin(['P0C1U8','Q2FZL2'])]
    old_ids = {
        'Q32MB2':'Q86Y46',
        'Q7RTT2':'Q8N1N4',
        'Q9R4J5':'Q9R4J4',
        'Q6IFU5':'Q6A162',
        'Q6IFU6':'Q6A163',
        'A3EZ79':'Q6E0U4',
        'A3EZ82':'Q6E0U4',
        'Q3SX28':'Q5KR48',
        'Q0V8M9':'P56652',
        'Q3SX09':'P02081',
        'Q2KJ62':'P01044',
        'Q0IIK2':'Q29443',
        'A2VCT4':'Q6IFX4',
        'A2A5Y0':'Q61765'
    }

    for index,row in conts.iterrows():
        if row['Uniprot ID'] in old_ids:
            conts.at[index,'Uniprot ID'] = old_ids[row['Uniprot ID']]
        if row['Uniprot ID'] in uniprots:
            uprow = uniprot_df.loc[row['Uniprot ID']]
            conts.at[index,'Length'] = float(uprow['Length'])
            conts.at[index,'Sequence'] = uprow['Sequence']
            conts.at[index,'Gene names'] = uprow['Gene names']
            conts.at[index,'Status'] = uprow['Reviewed']
    conts = conts.drop_duplicates(subset='Uniprot ID', keep='first')

    seqs = {entry: row['Sequence'] for entry, row in uniprot_df.iterrows()}
    seq_col = []
    for _,row in conts.iterrows():
        if row['Uniprot ID'] not in seqs:
            seq_col.append('')
        else:
            seq_col.append(seqs[row['Uniprot ID']])
    conts['Sequence'] = seq_col
    conts['Length'] = conts['Length'].fillna(1).astype(int)
    for i, row in conts[conts['Gene names'].isna()].iterrows():
        conts.at[i, 'Gene names'] = f'{row["Protein names"]}({row["Uniprot ID"]})'
    conts['Organism'] = conts['Organism'].fillna('None')
    conts['Sequence'] = conts['Sequence'].fillna('Unknown')
    conts['Sequence'] = conts['Sequence'].fillna('Unknown')
    conts['Source of Contamination'] = conts['Source of Contamination'].fillna('Unspecified')
    cont_table_str =  [
        f'CREATE TABLE IF NOT EXISTS  contaminants (',
    ]
    for i, c in enumerate(cont_cols):
        cont_table_str.append(f'    {c} {cont_exts[i]},',)
    cont_table_str = '\n'.join(cont_table_str).strip(',')
    cont_table_str += '\n);'
    for _, row in conts.iterrows():
        gn = row['Gene names']
        if not 'Uncharac' in gn:
            gn = gn.split()[0]
        gns = row['Gene names']
        data = [
            row['Uniprot ID'],
            int(row['Status']=='reviewed'),
            gn,
            row['Entry name'],
            gns,
            row['Organism'],
            row['Length'],
            row['Sequence'],
            row['DataBase'],
            row['Source of Contamination'],
            timestamp,
            -1
        ]
        add_str = f'INSERT INTO contaminants ({", ".join(cont_cols)}) VALUES ({", ".join(["?" for _ in cont_cols])})'
        insert_sql.append([add_str, data])
    table_create_sql.append(cont_table_str)

    inttable_cols = [
        'interaction TEXT PRIMARY KEY',
        'uniprot_id_a TEXT NOT NULL',
        'uniprot_id_b TEXT NOT NULL',
        'uniprot_id_a_noiso TEXT NOT NULL',
        'uniprot_id_b_noiso TEXT NOT NULL',
        'isoform_a TEXT',
        'isoform_b TEXT',
        'organism_interactor_a TEXT',
        'organism_interactor_b TEXT',
        'annotation_interactor_a TEXT',
        'annotation_interactor_b TEXT',
        'biological_role_interactor_a TEXT',
        'biological_role_interactor_b TEXT',
        'interaction_detection_method TEXT',
        'publication_identifier TEXT',
        'confidence_value TEXT',
        'interaction_type TEXT',
        'source_database TEXT NOT NULL',
        'publication_count TEXT',
        'throughput TEXT',
        'notes TEXT',
        'version_update_time TEXT',
        'update_time TEXT',
        'prev_version TEXT'
    ]

    inttable_create = ['CREATE TABLE IF NOT EXISTS known_interactions (']
    for col in inttable_cols:
        inttable_create.append(f'    {col},')
    inttable_create = '\n'.join(inttable_create).strip(',')
    inttable_create += '\n);'
    table_create_sql.append(inttable_create)

    intact.update(organisms=organisms)
    biogrid.update(organisms=organisms)
    write_int_insert_sql(inttable_cols, organisms, timestamp, tmpdir, ncpu)
    print('prot_knownint_and_contaminant_table done')
    return table_create_sql, insert_sql

def ms_runs_table(parameters: dict, timestamp: str, time_format: str) -> tuple[list, list]:
    parameters = parameters['MS runs information']
    table_create_sql = []
    insert_sql = []

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
        'avg_peaks_per_timepoint': 'REAL NOT NULL',
        'mean_intensity': 'INTEGER NOT NULL',
        'max_intensity': 'INTEGER NOT NULL',
        'json': 'TEXT NOT NULL',
        'trace': 'TEXT NOT NULL', 
        'intercept_json': 'TEXT NOT NULL'
    }
    for typ in ['MSn_filtered','TIC','MSn_unfiltered']:
        for key in ['auc','intercepts','avg_peaks_per_timepoint','mean_intensity','max_intensity', 'json','trace', 'intercept_json']:
            ms_cols.append(f'{typ.lower()}_{key.lower()} {keytypes[key]}')
            
    for col in ms_cols:
        mstable_create.append(f'    {col},')
    mstable_create = '\n'.join(mstable_create).strip(',')
    mstable_create += '\n);'
    table_create_sql.append(mstable_create)
    data_to_enter = []
    failed_json_files = []
    runs_done = set()
    banned_run_dirs = parameters['Ignore runs from dirs']
    run_id_regex = parameters['MS run ID regex']

    if parameters['Additional info excel'] != '':
        runlist = pd.read_excel(os.path.join(*parameters['Additional info excel']))
    else:
        runlist = pd.DataFrame(columns=['Sample name','Who','Sample type','Bait name','Bait / other uniprot or ID','Bait mutation','Cell line / material','Project','Notes','tag'])
    ms_run_datadir = parameters['MS data import dir']
    for i, datafilename in enumerate(os.listdir(ms_run_datadir)):
        with open(os.path.join(ms_run_datadir, datafilename)) as fil:
            try:
                dat = json.load(fil)
            except json.JSONDecodeError:
                failed_json_files.append(['json decode error', datafilename, ''])
                continue
        if dat['SampleID'] in runs_done: continue
        if dat['SampleInfo'] == ['']: continue
        banned = False
        for b in banned_run_dirs:
            if b in dat['SampleInfo']['SampleTable']['AnalysisHeader']['@FileName']:
                banned = True
        if banned:
            continue
        runs_done.add(dat['SampleID'])
        lc_method = None
        ms_method = None
        if isinstance(dat['SampleInfo'], list):
            failed_json_files.append(['no sample info',datafilename, dat])
            continue
        if not 'polarity_1' in dat:
            failed_json_files.append(['no polarity',datafilename, dat])
            continue
        if not re.match(run_id_regex, dat['SampleID']):
            failed_json_files.append(['run ID mismatch',datafilename, dat])
            continue
        for propdic in dat['SampleInfo']['SampleTable']['SampleTableProperties']['Property']:
            if propdic['@Name'] == 'HyStar_LC_Method_Name':
                lc_method = propdic['@Value']
            if propdic['@Name'] == 'HyStar_MS_Method_Name':
                ms_method = propdic['@Value']
        sample_names = {
            dat['SampleInfo']['SampleTable']['Sample']['@SampleID'],
            dat['SampleInfo']['SampleTable']['Sample']['@SampleID']+'.d',
            dat['SampleInfo']['SampleTable']['Sample']['@DataPath'],
        }
        samplerow = runlist[runlist['Raw file'].isin(sample_names)]
        if 'polarity_1_sers' in dat.keys():
            del dat['polarity_1_sers']
        if (lc_method is None) or (ms_method is None):
            print('LC or MS method not found')
            continue
        if len([k for k in dat.keys() if 'polarity' in k]) > 1:
            print('Multiple polarities found!')
            continue
        if samplerow.shape[0] == 0:
            samplerow = pd.Series(index = samplerow.columns, data = ['No data' for c in samplerow.columns])
        else:
            samplerow = samplerow.iloc[0]
        instrument = 'TimsTOF 1'
        runtime = datetime.strftime(
            datetime.strptime(
                dat['SampleInfo']['SampleTable']['AnalysisHeader']['@CreationDateTime'].split('+')[0],
                '%Y-%m-%dT%H:%M:%S'
            ),
            time_format
        )
        
        samplename = samplerow['Sample name']
        author = samplerow['Who']
        sample_type = samplerow['Sample type']
        bait = samplerow['Bait name']
        bait_uniprot = samplerow['Bait / other uniprot or ID']
        bait_mut = samplerow['Bait mutation']
        cell_line = samplerow['Cell line / material']
        project = samplerow['Project']
        author_notes = samplerow['Notes']
        bait_tag = samplerow['tag']
        try:
            precur = dat['NumPrecursors']
        except KeyError:
            precur = 'No precursor data'
        ms_run_row = [
            dat['SampleID'],
            dat['SampleInfo']['SampleTable']['AnalysisHeader']['@SampleID'],
            samplename,
            dat['SampleInfo']['SampleTable']['AnalysisHeader']['@FileName'],
            runtime,
            runtime.split()[0],
            instrument,
            author,
            sample_type,
            dat['DataType'],
            lc_method,
            ms_method,
            precur,
            bait,
            bait_uniprot,
            bait_mut,
            len(pd.Series(dat['polarity_1']['tic df']['Series'])),
            cell_line,
            project,
            author_notes,
            bait_tag,
            timestamp,
            -1
        ]
        for dataname in ['bpc filtered df', 'tic df', 'bpc unfiltered df']:
            ms_run_row.extend([
                dat['polarity_1'][dataname]['auc'],
                dat['polarity_1'][dataname]['intercepts'],
                dat['polarity_1'][dataname]['peaks_per_timepoint'],
                dat['polarity_1'][dataname]['mean_intensity'],
                dat['polarity_1'][dataname]['max_intensity'],
                json.dumps(dat['polarity_1'][dataname]['Series']),
                dat['polarity_1'][dataname]['trace'],
                json.dumps(dat['polarity_1'][dataname]['intercept_dict']),
            ])   
        
        data_to_enter.append(ms_run_row)
    for data in data_to_enter:
        add_str = f'INSERT INTO ms_runs ({", ".join([c.split()[0] for c in ms_cols])}) VALUES ({", ".join(["?" for _ in ms_cols])})'
        insert_sql.append([add_str, data])
    print('ms_runs_table done')
    return table_create_sql, insert_sql

def msmicroscopy_table(parameters: dict, timestamp: str) -> tuple[list, list]:
    table_create_sql = []
    insert_sql = []
    
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
    table_create_sql.append(msmictable_create)
    datadir = os.path.join(*parameters['Database build files directory'])
    for dirname in os.listdir(os.path.join(datadir,'msmic')):
        if not os.path.isdir(os.path.join(datadir, 'msmic',dirname)):
            continue
        version = dirname
        ref_data = pd.read_csv(os.path.join(datadir, 'msmic', version, 'msmic_ref_table.txt'),sep='\t')
        loc_data = pd.read_csv(os.path.join(datadir, 'msmic', version, 'msmic_localizations.txt'),sep='\t')
        loc_col = 'Organelle'

        loc_data[loc_col] = [s.capitalize().strip() for s in loc_data[loc_col].values]
        baitnorm = []
        baitsumnorm = []
        preys_in_baits = {}
        preys_in_localizations = {}
        db_bait_max = {}
        db_bait_sum= {}
        for b in ref_data['Bait'].unique():
            db_bait_max[b] = max(ref_data[ref_data['Bait']==b]['AvgSpec'].values)
            db_bait_sum[b] = sum(ref_data[ref_data['Bait']==b]['AvgSpec'].values)
        for _,row in ref_data.iterrows():
            if row['Prey'] not in preys_in_baits:
                preys_in_baits[row['Prey']] = {}
                preys_in_localizations[row['Prey']] = {}
            preys_in_baits[row['Prey']][row['Bait']] = row['AvgSpec']
            baitnorm.append(row['AvgSpec']/db_bait_max[row['Bait']])
            baitsumnorm.append(row['AvgSpec']/db_bait_sum[row['Bait']])
            localization = loc_data[loc_data['Bait']==row['Bait']].iloc[0][loc_col]
            if localization not in preys_in_localizations:
                preys_in_localizations[row['Prey']][localization] = []
            preys_in_localizations[row['Prey']][localization].append(row['AvgSpec'])
        ref_data['Bait_norm'] = baitnorm    
        ref_data['Bait_sumnorm'] = baitsumnorm
        unique_preys = [p for p, v in preys_in_localizations.items() if len(v) == 1]
        ref_data['Loc'] = [loc_data[loc_data['Bait']==bait].iloc[0][loc_col] for bait in ref_data['Bait'].values]
        ref_data['Unique_to_loc'] = [prey in unique_preys for prey in ref_data['Prey'].values]

        uref = ref_data[ref_data['Unique_to_loc']].copy()
        locnorm = []
        locsumnorm = []
        loc_max = {}
        loc_sum = {}
        for l in uref['Loc'].unique():
            loc_max[l] = uref[uref['Loc']==l]['AvgSpec'].max()
            loc_sum[l] = uref[uref['Loc']==l]['AvgSpec'].sum()
        for _,row in uref.iterrows():
            locnorm.append(row['AvgSpec']/loc_max[row['Loc']])
            locsumnorm.append(row['AvgSpec']/loc_sum[row['Loc']])
        uref['Loc_norm'] = locnorm
        uref['Loc_sumnorm'] = locsumnorm
        uref['MSMIC_version'] = version
        uref['Interaction'] = uref['Bait']+uref['Prey']
        uref['Version_update_time'] = timestamp
        uref['Prev_version'] = -1

        for _,row in uref.iterrows():
            data = [
                row[c.split()[0]] for c in msmictable_cols
            ]
            add_str = f'INSERT INTO msmicroscopy ({", ".join([c.split()[0] for c in msmictable_cols])}) VALUES ({", ".join(["?" for _ in msmictable_cols])})'
            insert_sql.append([add_str, data])
    print('msmicroscopy_table done')
    
    return table_create_sql, insert_sql

def common_proteins_table(parameters: dict, timestamp: str) -> tuple[list, list]:
    table_create_sql = []
    insert_sql = []
        
    comtable_create = ['CREATE TABLE IF NOT EXISTS common_proteins (']
    datafiledir = os.path.join(*parameters['Database build files directory'])
    comdir = os.path.join(datafiledir,'Potential contaminant protein groups')
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
    table_create_sql.append(comtable_create)
    common_proteins = {}

    for root, _, files in os.walk(comdir):
        if 'ipynb' in root: continue
        for f in files:
            comdf = pd.read_csv(os.path.join(root,f),sep='\t')
            name = root.rsplit(os.sep,maxsplit=1)[-1]
            for _,row in comdf.iterrows():
                if row['Entry'] not in common_proteins:
                    try:
                        common_proteins[row['Entry']] = [
                            row['Entry'],
                            row['Gene Names (primary)'],
                            row['Entry Name'],
                            row['Gene Names'],
                            row['Organism'],
                            [name],
                            timestamp,
                            -1
                        ]
                    except KeyError:

                        common_proteins[row['Entry']] = [
                            row['Entry'],
                            row['Gene names (primary)'],
                            row['Entry name'],
                            row['Gene names'],
                            row['Organism'],
                            [name],
                            timestamp,
                            -1
                        ]
                else:
                    common_proteins[row['Entry']][5].append(name)
    for _, data in common_proteins.items():
        data[5] = ', '.join(sorted(list(set(data[5]))))
        add_str = f'INSERT INTO common_proteins ({", ".join([c.split()[0] for c in com_cols])}) VALUES ({", ".join(["?" for _ in com_cols])})'
        insert_sql.append([add_str, data])
    print('common_proteins_table done')
    return table_create_sql, insert_sql

def run_table_generation(func, parameters, timestamp, time_format=None):
    if time_format:
        return func(parameters, timestamp, time_format)
    return func(parameters, timestamp)

def generate_database(parameters: dict, database_filename: str, time_format, timestamp: str,tmpdir:str, num_cpus: int,  organisms: set|None = None) -> None:
    full_table_create_sql = []
    num_cpus = num_cpus
    ms_create_sql, ms_insert_sql = ms_runs_table(parameters, timestamp, time_format)
    pkic_create_sql, pkic_insert_sql = prot_knownint_and_contaminant_table(parameters, timestamp, tmpdir, num_cpus, organisms)
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        cc_future = executor.submit(run_table_generation, crapome_and_controls, parameters, timestamp)
        #ms_future = executor.submit(run_table_generation, ms_runs_table, parameters, timestamp, time_format)
        msmic_future = executor.submit(run_table_generation, msmicroscopy_table, parameters, timestamp)
        com_future = executor.submit(run_table_generation, common_proteins_table, parameters, timestamp)

        cc_create_sql, cc_insert_sql = cc_future.result()
        #ms_create_sql, ms_insert_sql = ms_future.result()
        msmic_create_sql, msmic_insert_sql = msmic_future.result()
        com_create_sql, com_insert_sql = com_future.result()

    full_table_create_sql.extend(cc_create_sql)
    full_table_create_sql.extend(pkic_create_sql)
    full_table_create_sql.extend(ms_create_sql)
    full_table_create_sql.extend(msmic_create_sql)
    full_table_create_sql.extend(com_create_sql)

    insert_sql_list = [
        ('crapome+controls', cc_insert_sql),
        ('proteins+knowns+contaminants', pkic_insert_sql),
        ('ms_runs', ms_insert_sql),
        ('msmicroscopy', msmic_insert_sql),
        ('common_proteins', com_insert_sql),
    ]
    conn = sqlite3.connect(database_filename)
    cursor = conn.cursor()
    start = datetime.now()
    total = 0
    for create_table_str in full_table_create_sql:
        cursor.execute(create_table_str)
    for table_name, table_insert_sql in insert_sql_list:
        table_sql_len = len(table_insert_sql)
        print(table_name, table_sql_len, 'rows')
        for insert_str, insert_data in table_insert_sql:
            cursor.execute(insert_str, insert_data)
        total += table_sql_len
    int_start = total
    for filename in os.listdir(tmpdir):
        with open(os.path.join(tmpdir, filename),'r') as fil:
            data = json.load(fil)
            for dline in data['data']:
                cursor.execute(data['add_str'], dline)
                total += 1
    print('interactions', total-int_start, 'rows')
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    print('Table creation and data insertion took', (datetime.now() - start).seconds, 'seconds')
    print('Total number of rows inserted:', total)
    
    conn.commit()
    conn.close()
