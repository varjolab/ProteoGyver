import os
import pandas as pd
import json
from dateutil.parser import parse
from typing import Any
destdir = os.path.join('..','app','data','Run information')
destdirforjson = os.path.join(destdir,'info_files')
destdirfortic = os.path.join(destdir,'TIC_files')
if not os.path.isdir(destdirforjson):
    os.makedirs(destdirforjson)
    os.makedirs(destdirfortic)
new_cols = [
    'run_id',
    'run_name',
    'run_time',
    'MS',
    'sample_type',
    'AUC',
    'intercepts',
    'list_of_intercepts',
    'mean_intensity',
    'max_intensity',
    'tic_max_intensity',
    'monoisotopic_mz_mean',
    'monoisotopic_mz_max',
    'charge_mean',
    'no_of_precursors',
    'no_of_parents',
    'max_time',
    'runid_prefixed'
]
new_data = []
id_format_switch_date = parse('2022-11-25T00:00:00+02:00')
mt = 0
mi = 0
for curdir in [x for x in os.listdir('.') if 'ticdata_' in x]:
    for f in os.listdir(curdir):
        try:
            if f.endswith('Precursors.tsv'):
                run_name = f.replace('Precursors.tsv','')
            else:
                continue
            inf = os.path.join(curdir,run_name)
            with open(f'{inf} info.txt') as fil:
                info1 = [s.strip() for s in fil.readlines()]
            with open(f'{inf} info2.txt') as fil:
                info2 = [s.strip() for s in fil.readlines()]
            with open(f'{inf} info3.json') as fil:
                info3 = json.load(fil)
            tsv = pd.read_csv(f'{inf}.tsv',sep='\t')
            pre = pd.read_csv(f'{inf}Precursors.tsv',sep='\t')


            run_time = parse(info3['AcquisitionDateTime'])
            auc = float(info1[1].split()[1])
            intercepts = [float(x) for x in info1[3].split(': ')[1].split()]
            mean_intensity = float(info2[9].split(':')[1])
            max_intensity = float(info2[11].split(':')[1])
            tic_max_intensity = tsv['SumIntensity'].max()
            mono_mean = float(info2[1].split(':')[1])
            mono_max = float(info2[3].split(':')[1])
            charge_mean = pre['Charge'].mean()
            precu = int(info2[0].split(':')[1])
            par = int(info2[13].split(':')[1])
            ms = 'TimsTOF 1'
            if run_time < id_format_switch_date:
                run_id = run_name.split('_')[-1]
                newformat = False
            else:
                run_id = run_name.split('_')[0]
                newformat = True
            sample_type = 'BioID'
            new_data.append([
                run_id,
                run_name,
                run_time.strftime('%Y-%m-%d_%H-%M-%S_%z'),
                ms,
                sample_type,
                auc,
                len(intercepts),
                intercepts,
                mean_intensity,
                max_intensity,
                tic_max_intensity,
                mono_mean,
                mono_max,
                charge_mean,
                precu,
                par,
                tsv['Time'].max(),
                newformat
            ])
            if tsv['Time'].max() > mt:
                mt = tsv['Time'].max()
            if tsv['SumIntensity'].max() > mi:
                mi = tsv['SumIntensity'].max()
            tsv.to_csv(os.path.join(destdirfortic, f'{run_id}.tsv'),sep='\t',index=False)
        except FileNotFoundError:
            continue
df = pd.DataFrame(data=new_data,columns=new_cols)
df.to_csv(os.path.join(destdir, 'rundata.tsv'),sep='\t',index=False)
#df.to_csv('rundata.tsv',sep='\t',index=False)

general_information = {
    'Maximum intensity': mi,
    'Maximum time': mt,
    'Information fields': list(df.columns)
}
with open(os.path.join(destdir,'info.json'),'w',encoding='utf-8') as fil:
    json.dump(general_information,fil,indent=4)
for _,row in df.iterrows():
    dic = {
        c: row[c] for c in df.columns
    }
    with open(os.path.join(destdirforjson,f'{row["run_id"]}.json'),'w',encoding='utf-8') as fil:
        json.dump(dic,fil,indent=4)
