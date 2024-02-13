import sys
import os
sys.path.append('..')
from components import db_functions as dbfun
import json
from datetime import datetime
import pandas as pd

def add_runs(file_paths: list, database_file: str, runlist: pd.DataFrame, time_format:str):
    db_conn = dbfun.create_connection(database_file)

    current_table = dbfun.get_full_table_as_pd(db_conn, 'ms_runs')
    db_conn.close()

    data_to_enter = []
    data_to_modify = []
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
            data_to_modify.append(ms_run_row)
        else:
            data_to_enter.append(ms_run_row)
            
    db_conn = dbfun.create_connection(database_file)
    fails = []
    for data in data_to_enter:
        try:
            debu = dbfun.add_record(
                db_conn, 'ms_runs',current_table.columns, data
            )
        except:
            fails.append(['add',data])
    dodi = {}
    for data in data_to_modify:
        diff_vals = []
        diff_cols = []
        oldvals = current_table[current_table['run_id']==data[0]].iloc[0]
        for i, v in enumerate(data):
            if pd.isna(v):continue
            colname = current_table.columns[i]
            if oldvals[colname] != v:
                if str(v) != str(oldvals[colname]):
                    diff_vals.append(v)
                    diff_cols.append(colname)
        if len(diff_vals) > 0:
            dodi[data[0]] = {}
            for c, v in enumerate(diff_vals):
                oldval = oldvals[diff_cols[c]]
                dodi[data[0]][diff_cols[c]] = [str(type(v)), str(type(oldval)), str(v), str(oldval)]
            try:
                dbfun.modify_record(db_conn, 'ms_runs','run_id',data[0], diff_cols, diff_vals)
            except:
                fails.append(['modify',data])
    db_conn.commit()
    db_conn.close()
    with open('debug_differences.json','w') as fil:
        json.dump(dodi, fil,indent=2)
    with open('debug_mods.json','w') as fil:
        json.dump({'data': data_to_modify}, fil,indent=2)
    with open('enter_data.json','w') as fil:
        json.dump({'data': data_to_enter},fil,indent=2)
    with open('fails.json','w') as fil:
        json.dump({'failed json files': failed_json_files, 'fails': fails},fil,indent=2)

with open(os.path.join('..','parameters.json'),encoding='utf-8') as fil:
    parameters = json.load(fil)
    time_format_from_config = parameters['Config']['Time format']
    parameters = parameters['Data paths']
db_file = os.path.join('..',os.path.join(*parameters['Database file']).replace('.db','testing.db'))
runlist = pd.read_excel(os.path.join('..',os.path.join(*parameters['Run list'])))
import_dir = os.path.join(*parameters['Data import and export']['Data import dir'])
import_dir = '/run/user/1237916/gvfs/smb-share:server=biotek-filesrv1.ad.helsinki.fi,share=data1/varjosalo/Kari/202401 tims chroms/20240123 tics3'
#print(import_dir, db_file, os.listdir(import_dir))
additions = {}
for root, dirs, fils in os.walk(import_dir):
    if len(fils) > 0:
        destination = tuple(root.replace(import_dir,'').strip(os.sep).split(os.sep))
        additions[destination] = []
        for f in fils:
            additions[destination].append((root,f))
fun_dict = {
    'ms_runs': {'func': add_runs, 'params': {'runlist': runlist, 'time_format': time_format_from_config}}
}
for table_info, add_files in additions.items():
    table_name = table_info[0]
    if table_name not in fun_dict:
        continue
    else:
        fun_dict[table_name]['func'](add_files, db_file, **fun_dict[table_name]['params'])
 