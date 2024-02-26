import sys
import os
sys.path.append('..')
from components import db_functions as dbfun
import json
from datetime import datetime
import pandas as pd

def add_runs(file_paths: list, database_file: str, _, runlist: pd.DataFrame, time_format:str):
    db_conn = dbfun.create_connection(database_file)

    current_table = dbfun.get_full_table_as_pd(db_conn, 'ms_runs')
    db_conn.close()

    additions = []
    modifications = []
    failed_json_files = []
    not_these =  ['BRE_20_xxxxx_Helsinki','TrapTrouble_3']
    for datafile_path in file_paths:
        with open(os.path.join(*datafile_path)) as fil:
            try:
                dat = json.load(fil)
            except json.JSONDecodeError:
                failed_json_files.append(['json decode error', str(datafile_path), ''])
                continue
        lc_method = None
        ms_method = None

        if isinstance(dat['SampleInfo'], list):
            failed_json_files.append(['no sample info',str(datafile_path), dat])
            continue
        if not 'polarity_1' in dat:
            failed_json_files.append(['no polarity',str(datafile_path), dat])
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
        if (lc_method is None) or (ms_method is None):
            continue
        if len([k for k in dat.keys() if 'polarity' in k]) > 1:
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

        do = True
        for exclude_str in not_these:
            if exclude_str in dat['SampleInfo']['SampleTable']['AnalysisHeader']['@FileName']:
                do = False
        if not do: continue
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
            bait_tag
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
        
        if dat['SampleID'] in current_table['run_id'].values:
            diff_vals = []
            diff_cols = []
            oldvals = current_table[current_table['run_id']==ms_run_row[0]].iloc[0]
            for i, v in enumerate(ms_run_row):
                if pd.isna(v):continue
                colname = current_table.columns[i]
                if oldvals[colname] != v:
                    if str(v) != str(oldvals[colname]):
                        diff_vals.append(v)
                        diff_cols.append(colname)
            modifications.append('ms_runs', 'run_id', ms_run_row[0], diff_cols, diff_vals)
        else:
            additions.append('ms_runs', current_table.columns, ms_run_row)
    return (additions, modifications)

def add_common(file_paths, database_file, _):
    db_conn = dbfun.create_connection(database_file)
    current = dbfun.get_full_table_as_pd(db_conn, 'common_proteins')
    have = set(current['uniprot_id'].values)
    db_conn.close()
    common_proteins = {}
    for root, f in file_paths:
        comdf = pd.read_csv(os.path.join(root,f),sep='\t')
        if not 'Gene Names (primary)' in comdf.columns:
            gns = []
            for _,row in comdf.iterrows():
                g = str(row['Gene Names'])
                if len(g) > 0:
                    g = g.split()[0]
                gns.append(g)
            comdf['Gene Names (primary)'] = gns
        name = root.rsplit(os.sep,maxsplit=1)[-1]
        for _,row in comdf.iterrows():
            if row['Entry'] not in common_proteins:
                common_proteins[row['Entry']] = [
                    row['Entry'],
                    row['Gene Names (primary)'],
                    row['Entry Name'],
                    row['Gene Names'],
                    row['Organism'],
                    [name],
                    datetime.today().strftime('%Y-%m-%d')
                ]
            else:
                common_proteins[row['Entry']][5].append(name)
    modifications = []
    additions = []
    for common_protein, data in common_proteins.items():
        if common_protein in have:
            cur_ver = current[current['uniprot_id']==common_protein]
            if cur_ver.shape[0] != 1:
                print(cur_ver)
            cur_ver = cur_ver.iloc[0]
            mods = cur_ver['protein_type'].split(', ')
            mods.extend(data[5])
            modifications.append([
                'common_proteins',
                'uniprot_id',
                common_protein,
                ['protein_type'],
                [', '.join(sorted(list(set(mods))))]
            ])
        else:
            data[5] = ', '.join(data[5])
            additions.append(['common_proteins', current.columns, data])
    return (additions, modifications)

def commit_modifications(additions, modifications, database_file):
    print(modifications)
    return []
    db_conn = dbfun.create_connection(database_file)
    fails = []
    for m in modifications:
        try:
            dbfun.modify_record(db_conn, *m)
        except Exception as e:
            fails.append(['modify', m, f'{e}'])
    for a in additions:
        try:
            dbfun.add_record(db_conn, *a)
        except Exception as e:
            fails.append(['addition', a, f'{e}'])
    db_conn.commit()
    db_conn.close()
    return fails

with open(os.path.join('..','parameters.json'),encoding='utf-8') as fil:
    parameters = json.load(fil)
    time_format_from_config = parameters['Config']['Time format']
    parameters = parameters['Data paths']
db_file = os.path.join('..',os.path.join(*parameters['Database file']).replace('.db','testing.db'))
runlist = pd.read_excel(os.path.join('..',os.path.join(*parameters['Run list'])))
import_dir = os.path.join(*parameters['Data import and export']['Data import dir'])
import_dir = '/home/kmsaloka/Downloads/Database additions'
#import_dir = '/run/user/1237916/gvfs/smb-share:server=biotek-filesrv1.ad.helsinki.fi,share=data1/varjosalo/Kari/202401 tims chroms/20240123 tics3'
#print(import_dir, db_file, os.listdir(import_dir))
additions = {}
for root, dirs, fils in os.walk(import_dir):
    if len(fils) > 0:
        destination = tuple(root.replace(import_dir,'').strip(os.sep).split(os.sep))
        additions[destination] = []
        for f in fils:
            additions[destination].append((root,f))
fun_dict = {
    'ms_runs': {
        'func': add_runs,
        'params': {
            'runlist': runlist,
            'time_format': time_format_from_config
        }
    },
    'common_proteins': {
        'func': add_common,
        'params': {

        }
    }
}

db_operations = []
for table_info, add_files in additions.items():
    table_name = table_info[0]
    if table_name not in fun_dict:
        continue
    else:
        db_operations.append(
            fun_dict[table_name]['func'](
                add_files,
                db_file,
                table_info[1],
                **fun_dict[table_name]['params']
            )
        )
commit_modifications(
    [op[0] for op in db_operations],
    [op[1] for op in db_operations],
    db_file
    )