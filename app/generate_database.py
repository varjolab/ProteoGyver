import sqlite3
import os
import pandas as pd
import json
import numpy as np
from datetime import datetime
from components import text_handling
from components.api_tools.annotation import intact
from components.api_tools.annotation import biogrid
from components.api_tools.annotation import uniprot

with open('parameters.json') as fil:
    parameters = json.load(fil)['Database creation']

dbdir = os.path.join('data','db')
datadir = os.path.join(dbdir,'db build files')
crapome = pd.read_csv(os.path.join(datadir,parameters['Crapome table']),sep='\t')
controls = pd.read_csv(os.path.join(datadir,parameters['Control table']),sep='\t')
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
jsons = {}
for f in os.listdir(datadir):
    if f.split('.')[-1]=='json':
        with open(os.path.join(datadir,f)) as fil:
            jsons[f'data_{f}'] = json.load(fil)
runlist = pd.read_excel(os.path.join('..','..','..','combined runlist.xlsx'))

ms_run_datadir = parameters['MS data import dir']
sets = {
    'VL GFP MAC3 10min AP': [
        'VL GFP MAC3-N AP-MS',
    ],
    'VL GFP MAC3 10min BioID': [
        'VL GFP MAC3-N BioID',
    ],
    'VL GFP MAC2 18h AP': [
        'VL GFP MAC2-C AP-MS',
        'VL GFP MAC2-N AP-MS',
    ],
    'VL GFP MAC2 18h BioID': [
        'VL GFP MAC2-C BioID',
        'VL GFP MAC2-N BioID'
    ],
    'VL GFP MAC 24h AP': [
        'VL GFP MAC-C AP-MS',
        'VL GFP MAC-N AP-MS',
    ],
    'VL GFP MAC 24h AP NLS': [
        'VL GFP MAC-MED-NLS AP-MS',
        'VL GFP MAC-MYC-NLS AP-MS',
        'VL GFP MAC-NLS AP-MS',
    ],
    'VL GFP MAC 24h BioID': [
        'VL GFP MAC-C BioID',
        'VL GFP MAC-N BioID'
    ],
    'VL GFP MAC 24h BioID NLS': [
        'VL GFP MAC-MED-NLS BioID',
        'VL GFP MAC-MYC-NLS BioID',
        'VL GFP MAC-NLS BioID'
    ],
    'Nesvilab': [
        'nesvilab'
    ]
}

additional_controls_dir = os.path.join(datadir,'gfp control')
new_control_sets = {}
overall_setnames = {
    '202401 Liu AP-MS': 'VL GFP MAC3 10min AP',
    '202401 Liu BioID': 'VL GFP MAC3 10min BioID',
    '202411 GD BioID': 'VL GFP MAC3 6h BioID',
    '202411 GD_loc BioID': 'VL GFP MAC3 6h BioID',
}
for setdir in os.listdir(additional_controls_dir):
    sampleinfo = pd.read_excel(os.path.join(additional_controls_dir, setdir, 'Sample_Information.xlsx'))
    data = pd.read_csv(os.path.join(additional_controls_dir, setdir, 'reprint.spc.tsv'),sep='\t', index_col='PROTID').drop(columns=['GENEID','PROTLEN'])
    data = data[(data.index.notna()) & (~data.index.isin({'na','NA'}))].astype(int).replace('na',np.nan).replace(0, np.nan)
    data = data.reset_index()
    namecol, runidcol, samplename, _, expcol = sampleinfo.columns
    for exp in sampleinfo[expcol].unique():
        setname = f'{setdir} {exp}'
        if overall_setnames[setname] not in sets:
            sets[overall_setnames[setname]] = []
        sets[overall_setnames[setname]].append(setname)
        jsons['data_control sets.json'][setname] = list(data.columns)
        jsons['data_crapome sets.json'][setname] = list(data.columns)
        print(f'added {setname}')
    crapome = crapome.merge(data,left_on='PROTID',right_on='PROTID',how='outer')
    controls = controls.merge(data,left_on='PROTID',right_on='PROTID',how='outer')

crapome_tables = {}
columns = [
    'protein_id',
    'identified_in',
    'frequency',
    'spc_sum',
    'spc_avg',
    'spc_min',
    'spc_max',
    'spc_stdev']
types = [
    'TEXT PRIMARY KEY',
    'INTEGER NOT NULL',
    'REAL NOT NULL',
    'INTEGER NOT NULL',
    'REAL NOT NULL',
    'INTEGER NOT NULL',
    'INTEGER NOT NULL',
    'REAL NOT NULL'
]
crapome_entries = []
for setname, setcols in sets.items():
    all_cols = ['PROTID']
    defa = 1
    if 'MAC2' in setname: defa = 0 # default enabled value
    tablename = f'crapome_{setname}'.lower().replace(' ','_')
    for sc in setcols:
        all_cols.extend(jsons['data_crapome sets.json'][sc])
    all_cols = sorted(list(set(all_cols)))
    set_df = crapome[all_cols]
    setname = f'{setname} ({len(all_cols)} runs)'
    set_df.index = set_df['PROTID']
    set_df = set_df.drop(columns=['PROTID']).replace(0,np.nan).dropna(how='all',axis=0).dropna(how='all',axis=1)
    nruns = set_df.shape[1]
    set_data = []
    for protid, row in set_df.iterrows():
        stdval = row.std()
        if pd.isna(stdval):
            stdval = -1
        set_data.append([protid, row.notna().sum(), row.notna().sum()/nruns,row.sum(), row.mean(), row.min(), row.max(), stdval])
    crapome_tables[tablename] = pd.DataFrame(columns=columns, data=set_data)
    crapome_entries.append([tablename, setname, nruns, 0, defa, tablename, timestamp, -1])

    control_tables = {}
control_entries = []
for setname, setcols in sets.items():
    if setname == 'Nesvilab': continue
    all_cols = ['PROTID']
    defa = 1
    if 'MAC2' in setname: defa = 0
    tablename = f'control_{setname}'.lower().replace(' ','_')
    for sc in setcols:
        all_cols.extend(jsons['data_control sets.json'][sc])
    all_cols = sorted(list(set(all_cols)))
    setname = f'{setname} ({len(all_cols)} runs)'
    set_df = controls[all_cols]
    set_df.index = set_df['PROTID']
    set_df = set_df.drop(columns=['PROTID']).replace(0,np.nan).dropna(how='all',axis=0).dropna(how='all',axis=1)
    nruns = set_df.shape[1]
    set_data = []
    for protid, row in set_df.iterrows():
        stdval = row.std()
        if pd.isna(stdval):
            stdval = -1
        set_data.append([protid, row.notna().sum(), row.notna().sum()/nruns,row.sum(), row.mean(), row.min(), row.max(), stdval])
    control_tables[tablename] = (set_df, pd.DataFrame(columns=columns, data=set_data))
    control_entries.append([tablename, setname, nruns, 0, defa, tablename, timestamp, -1])

control_cols = ['control_set','control_set_name','runs','is_disabled','is_default','control_table_name','version_update_time','prev_version']
crapome_cols = ['crapome_set','crapome_set_name','runs','is_disabled','is_default','crapome_table_name','version_update_time','prev_version']
exts = ['TEXT PRIMARY KEY','TEXT NOT NULL','INTEGER NOT NULL','INTEGER NOT NULL','INTEGER NOT NULL','TEXT NOT NULL','TEXT NOT NULL','TEXT']

control_table_str =  [
        f'CREATE TABLE IF NOT EXISTS control_sets (',
    ]
for i, c in enumerate(control_cols):
    control_table_str.append(f'    {c} {exts[i]},',)
control_table_str = '\n'.join(control_table_str).strip(',')
control_table_str += '\n);'

crapome_table_str =  [
        f'CREATE TABLE IF NOT EXISTS  crapome_sets (',
    ]
for i, c in enumerate(crapome_cols):
    crapome_table_str.append(f'    {c} {exts[i]},',)
crapome_table_str = '\n'.join(crapome_table_str).strip(',')
crapome_table_str += '\n);'

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

table_create_sql = [control_table_str, crapome_table_str, prot_table_str]

control_insert_sql = []
for vals in control_entries:
    tablename = vals[0]
    detailed, overall = control_tables[tablename]
    detailed.rename(
        columns={
            c: 'CS_'+text_handling.replace_accent_and_special_characters(c,'_')
            for c in detailed.columns
        },
        inplace=True
    )
    create_str = [
        f'CREATE TABLE IF NOT EXISTS  {tablename}_overall (',
    ]
    for i, c in enumerate(overall.columns):
        create_str.append(f'    {c} {types[i]},',)
    create_str = '\n'.join(create_str).strip(',')
    create_str += '\n);'
    table_create_sql.append(create_str)
    add_str = [f'INSERT INTO control_sets ({", ".join(control_cols)}) VALUES ({", ".join(["?" for _ in control_cols])})', vals]
    control_insert_sql.append(add_str)
    for _, row in overall.iterrows():
        add_str = [f'INSERT INTO {tablename}_overall ({", ".join(overall.columns)}) VALUES ({", ".join(["?" for _ in overall.columns])})', tuple(row.values)]
        control_insert_sql.append(add_str)
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
        control_insert_sql.append(add_str)
print('control:', len(control_insert_sql))

crapome_insert_sql = []
for vals in crapome_entries:
    tablename = vals[0]
    create_str = [
        f'CREATE TABLE IF NOT EXISTS  {tablename} (',
    ]
    for i, c in enumerate(columns):
        create_str.append(f'    {c} {types[i]},',)
    create_str = '\n'.join(create_str).strip(',')
    create_str += '\n);'
    table_create_sql.append(create_str)
    add_str = [f'INSERT INTO crapome_sets ({", ".join(crapome_cols)}) VALUES({", ".join(["?" for _ in crapome_cols])})', vals]
    crapome_insert_sql.append(add_str)
    for _, row in crapome_tables[tablename].iterrows():
        add_str = [f'INSERT INTO {tablename} ({", ".join(columns)}) VALUES ({", ".join(["?" for _ in columns])})', tuple(row.values)]
        crapome_insert_sql.append(add_str)
print('crapome:',len(crapome_insert_sql))

#TODO: Download uniprot data here
uniprot_df = uniprot.download_uniprot_chunks(reviewed_only=True,fields=parameters['Uniprot fields'])
uniprots = set(uniprot_df.index.values)
proteins_insert_sql = []
for protid, row in uniprot_df.iterrows():
    gn = row['Gene Names (primary)']
    if pd.isna(gn):
        gn = row['Entry Name']
    gns = row['Gene Names']
    if pd.isna(gns):
        gns = row['Entry Name']
    row = row.fillna('')
    data = [
        protid,
        int(row['Reviewed']=='reviewed'),
        gn,
        row['Entry Name'],
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
    proteins_insert_sql.append([add_str, data])
print('protein:', len(proteins_insert_sql))


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

contaminants_insert_sql = []
conts = pd.read_csv(os.path.join(datadir,parameters['Contaminants table']),sep='\t')
conts = conts[~conts['Uniprot ID'].isin(['P0C1U8','Q2FZL2'])]
#TODO: Download idmapping data here
dd = pd.read_csv(os.path.join(datadir,'idmapping_2023_09_11.tsv'),sep='\t')
for _,row in dd.iterrows():
    conts.loc[conts[conts['Uniprot ID']==row['Entry']].index,'Length'] = row['Length']
dd2 = pd.read_csv(os.path.join(datadir,'idmapping_2023_09_121.tsv'),sep='\t')
for _, row in dd2.iterrows():
    ctloc = conts[conts['Uniprot ID']==row['From']]
    conts.loc[ctloc.index, 'Sequence'] = row['Sequence']
    conts.loc[ctloc.index, 'Gene names'] = row['Gene Names']
    conts.loc[ctloc.index, 'Length'] = row['Length']
    conts.loc[ctloc.index, 'Status'] = row['Reviewed']

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
    conts.loc[i, 'Gene names'] = f'{row["Protein names"]}({row["Uniprot ID"]})'
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
    contaminants_insert_sql.append([add_str, data])
table_create_sql.append(cont_table_str)
print('contaminants:',len(contaminants_insert_sql))

from io import StringIO
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
        
ms_runs_insert_sql = []
for col in ms_cols:
    mstable_create.append(f'    {col},')
mstable_create = '\n'.join(mstable_create).strip(',')
mstable_create += '\n);'
table_create_sql.append(mstable_create)
data_to_enter = []
failed_json_files = []
runs_done = set()
banned_run_dirs = [
    'BRE_20_xxxxx_Helsinki',
    'TrapTrouble_3'
]
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
    frame_df_name = f'{instrument} {dat["SampleID"]}'
    frame_df = pd.read_json(StringIO(json.dumps(dat['Frames'])),orient='split')
    runtime = datetime.strftime(
        datetime.strptime(
            dat['SampleInfo']['SampleTable']['AnalysisHeader']['@CreationDateTime'].split('+')[0],
            '%Y-%m-%dT%H:%M:%S'
        ),
        parameters['Config']['Time format']
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
    ms_runs_insert_sql.append([add_str, data])
print(len(ms_runs_insert_sql))

inttable_create = ['CREATE TABLE IF NOT EXISTS known_interactions (']
inttable_cols = [
    'interaction TEXT PRIMARY KEY',
    'uniprot_id_a TEXT NOT NULL',
    'uniprot_id_b TEXT NOT NULL',
    'uniprot_id_a_noiso TEXT NOT NULL',
    'uniprot_id_b_noiso TEXT NOT NULL',
    'source_database TEXT NOT NULL',
    'isoform_a TEXT',
    'isoform_b TEXT',
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
for col in inttable_cols:
    inttable_create.append(f'    {col},')
inttable_create = '\n'.join(inttable_create).strip(',')
inttable_create += '\n);'
table_create_sql.append(inttable_create)
known_interactions_insert_sql = []

biogrid.update(uniprots)
intact.update(uniprots)
dbtables = [intact.get_latest(), biogrid.get_latest()]
for d in dbtables:
    if 'Unnamed: 0' in d.columns:
        d.drop(columns=['Unnamed: 0'],inplace = True)
shared = set()
for i, d in enumerate(dbtables):
    for d2 in dbtables[i+1:]:
        shared |= (set(d2.index) & set(d.index))

shared = sorted(list(shared))
ind = 0
if len(shared) > 0:
    mtables = [d.loc[shared].sort_index() for d in dbtables]
    dbtables = [d.drop(index=shared) for d in dbtables]
new_data = []
no_set = [c for c in mtables[0].columns if (('uniprot' not in c))]
for c in mtables[0].columns:
    if c == 'source_database':
        jst = ';'
    else:
        jst = '__'
    if c in no_set:
        nc = mtables[0][c].astype(str)
        for d in mtables[1:]:
            nc = nc + jst + d[c].astype(str)
        new_data.append(nc)
    else:
        new_data.append(mtables[0][c])
newer_data = [
    c.str.replace('nan','').str.strip('_') for c in new_data
]
shared_df = pd.DataFrame.from_dict({c: newer_data[i] for i, c in enumerate(mtables[0].columns)}).replace('__',np.nan)
shared_df.index = mtables[0].index
dbtables.append(shared_df)
merg = pd.concat(dbtables)
pm = []
npubs = []
for _,row in merg.iterrows():
    pmids = set()
    for p in row['publication_identifier'].split('__'):
        for pp in p.split(';'):
            if 'pubmed' in pp.lower():
                pmids.add(pp)
    pm.append(len(pmids))
    npubs.append(';'.join(sorted(list(pmids))))
merg['publication_count'] = pm
merg['publication_identifier'] = npubs
merg['version_update_time'] = timestamp
merg['prev_version'] = -1

int_df_slim = merg[merg['uniprot_id_a'].isin(uniprots) & merg['uniprot_id_b'].isin(uniprots)]
int_df_slim = int_df_slim.reset_index()
for _,row in int_df_slim.iterrows():
    data = [
        row[c.split()[0]] for c in inttable_cols
    ]
    add_str = f'INSERT INTO known_interactions ({", ".join([c.split()[0] for c in inttable_cols])}) VALUES ({", ".join(["?" for _ in inttable_cols])})'
    known_interactions_insert_sql.append([add_str, data])
print('Knowns', len(known_interactions_insert_sql))


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
msmicroscopy_insert_sql = []
table_create_sql.append(msmictable_create)
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
        msmicroscopy_insert_sql.append([add_str, data])
    print(version, len(msmicroscopy_insert_sql))


common_proteins_insert_sql = []
comtable_create = ['CREATE TABLE IF NOT EXISTS common_proteins (']
comdir = os.path.join(datadir,'Potential contaminant protein groups')
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

for root, dirs, files in os.walk(comdir):
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
                        datetime.today().strftime('%Y-%m-%d'),
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
                        datetime.today().strftime('%Y-%m-%d'),
                        -1
                    ]
            else:
                common_proteins[row['Entry']][5].append(name)
for common_protein, data in common_proteins.items():
    data[5] = ', '.join(sorted(list(set(data[5]))))
    add_str = f'INSERT INTO common_proteins ({", ".join([c.split()[0] for c in com_cols])}) VALUES ({", ".join(["?" for _ in com_cols])})'
    common_proteins_insert_sql.append([add_str, data])
print(len(common_proteins_insert_sql))

# # Connect to the database (create it if it doesn't exist)
conn = sqlite3.connect(os.path.join(dbdir,'proteogyver2.db'))
# Create a cursor object
cursor = conn.cursor()
start = datetime.now()
for create_table_str in table_create_sql:
    if len(create_table_str) == 39:
        print(create_table_str)
        continue
    cursor.execute(create_table_str)
for insert_str, insert_data in control_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in crapome_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in proteins_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in contaminants_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in ms_runs_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in known_interactions_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in msmicroscopy_insert_sql:
    cursor.execute(insert_str, insert_data)
for insert_str, insert_data in common_proteins_insert_sql:
    cursor.execute(insert_str, insert_data)
print('Table creation and data insertion took', (datetime.now() - start).seconds, 'seconds')
# Commit changes and close the connection
conn.commit()
conn.close()


su = 0
for iq in [
    control_insert_sql,
    crapome_insert_sql,
    proteins_insert_sql,
    contaminants_insert_sql,
    ms_runs_insert_sql,
    known_interactions_insert_sql,
    msmicroscopy_insert_sql,
    common_proteins_insert_sql
]:
    su += len(iq)
    print(len(iq))
print('total',su)