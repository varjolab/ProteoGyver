import os
import re
import numpy as np
import pandas as pd
import alphatims.bruker
import sys
import json
from dateutil.parser import parse
from typing import Any

def calculate_auc(df,xcol,ycol):
    auc = 0.0
    cum_auc = []
    for i, row in df.iterrows():
        if i>df.iloc[0].name:
            miny = min([row[ycol], df.loc[i-1,ycol]])
            maxy = max([row[ycol], df.loc[i-1,ycol]])
            xch = row[xcol]-df.loc[i-1,xcol]
            row_value = (miny*xch) + ((maxy*xch)/2)
            cum_auc.append(auc)
            auc += row_value
    cum_auc.append(auc)
    return (auc, cum_auc)

indir = sys.argv[1]
outdir = sys.argv[2]

print('indir:',indir,'outdir:',outdir)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
skip = [0,0,0, 0]
need = []
with open('need.txt',encoding='utf-8') as fil:
    for line in fil:
        need.append(line.strip())
for root, dirs, _ in os.walk(indir):
    for dname in dirs:
        if False:#dname in need:
            noskip = True
        else:
            noskip = False
        if dname.split('.')[-1]!='d':
            continue
        print(dname, noskip)
        js = {}
        if os.path.isfile(os.path.join(outdir, dname.replace('.d',' info3.json'))):
            skip[0] += 1
            if not noskip:
                continue
        else:
            with open(os.path.join(outdir, dname.replace('.d',' info.txt')), 'w', encoding = 'utf-8') as fil:
                fil.write('No data for run')
            with open(os.path.join(outdir, dname.replace('.d',' info3.json')), 'w', encoding = 'utf-8') as fil:
                json.dump(js,fil)
            with open(os.path.join(outdir, dname.replace('.d',' info2.txt')), 'w', encoding = 'utf-8') as fil:
                fil.write('No data for run')
        if os.path.isfile(os.path.join(outdir, dname.replace('.d',' info.txt'))):
            skip[1] += 1
            if not noskip:
                continue
        try:
            print(f'Reading: {dname}')
            diadata = alphatims.bruker.TimsTOF(os.path.join(root,dname))
            js = diadata.meta_data
        except:
            skip[2] += 1
            continue
        
        try:
            print(f'Reading: {dname}')
            diadata = alphatims.bruker.TimsTOF(os.path.join(root,dname))
            precur = diadata.precursors
            precur["Mass"] = precur["MonoisotopicMz"]*precur['Charge']
            with open(os.path.join(outdir, dname.replace('.d',' info2.txt')), 'w', encoding = 'utf-8') as fil:
                fil.write(f'Precursors: {precur.shape[0]}\n')
                fil.write(f'Monoisotopic Mz mean: {precur["MonoisotopicMz"].mean()}\n')
                fil.write(f'Monoisotopic Mz min:{precur["MonoisotopicMz"].min()}\n')
                fil.write(f'Monoisotopic Mz max:{precur["MonoisotopicMz"].max()}\n')
                fil.write(f'Monoisotopic Mz std:{precur["MonoisotopicMz"].std()}\n')
                fil.write(f'Mass mean: {precur["Mass"].mean()}\n')
                fil.write(f'Mass min: {precur["Mass"].min()}\n')
                fil.write(f'Mass max: {precur["Mass"].max()}\n')
                fil.write(f'Mass std: {precur["Mass"].std()}\n')
                fil.write(f'Intensity mean: {precur["Intensity"].mean()}\n')
                fil.write(f'Intensity min: {precur["Intensity"].min()}\n')
                fil.write(f'Intensity max: {precur["Intensity"].max()}\n')
                fil.write(f'Intensity std: {precur["Intensity"].std()}\n')
                fil.write(f'Parents: {len(precur["Parent"].unique())}\n')
            pd.DataFrame(precur['Charge'].value_counts()).reset_index().rename(columns = {'index': 'Charge', 'Charge': 'Count'}).to_csv(os.path.join(outdir, dname.replace('.d','Precursor charges.tsv')),sep='\t',index=False)
            precur[["Intensity",'Charge',"Mass"]].to_csv(os.path.join(outdir, dname.replace('.d','Precursors.tsv')),sep='\t',index=False)
            print(f'Finished: {dname}')
        except:
            skip[3] += 1
            continue
        sumvalues = []
        j = 0
        rtvalnum = len(list(diadata.rt_values))
        for rt in sorted(list(diadata.rt_values)):
            j+=1
            if j % 1000 == 0:
                print(j, '/', rtvalnum)
            rtdf = diadata[rt]
            sumvalues.append([rt,rtdf['intensity_values'].sum()])
        
        xcol = 'Time'
        ycol = 'SumIntensity'
        df = pd.DataFrame(data=sumvalues, columns=[xcol,ycol])
        nw = []
        for i in range(0, int(max(df[xcol]))+1):
            nw.append([
                i,
                df[(i<df[xcol]) & (df[xcol]<=(i+1))][ycol].max()
            ])
        df2 = pd.DataFrame(data=nw, columns=[xcol,ycol])
        auc = calculate_auc(df2, xcol, ycol)[0]
        mean = df2[ycol].mean()
        intercepts = []
        for index,row in df2.iterrows():
            if index == 0:
                continue
            if df2.iloc[index-1][ycol] < mean:
                if row[ycol] > mean:
                    intercepts.append(row[xcol])
            elif df2.iloc[index-1][ycol] > mean:
                if row[ycol] < mean:
                    intercepts.append(row[xcol])
        df2.to_csv(os.path.join(outdir, dname.replace('.d','.tsv')),sep='\t',index=False)
        with open(os.path.join(outdir, dname.replace('.d',' info.txt')), 'w', encoding = 'utf-8') as fil:
            fil.write(f'Mean: {mean}\n')
            fil.write(f'AUC: {auc}\n')
            fil.write(f'Number of intercepts: {len(intercepts)}\n')
            fil.write(f'Intercepts: {" ".join([str(x) for x in intercepts])}\n')
        print(f'Finished: {dname}')
print(skip)
print('Parsing data')

destdir = os.path.join('Run information')
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
id_format_switch_date = parse('2022-10-26T16:00:44+0300')

if os.path.isfile(os.path.join(destdir,'info.json')):
    with open(os.path.join(destdir,'info.json')) as fil:
        general_information = json.load(fil)
        general_information['Information fields'] = set(general_information['Information fields'])
else:    
    general_information = {
        'Maximum intensity': 0,
        'Maximum time': 0,
        'Information fields': set()
    }
mt = 0
mi = 0
skip = [0,0,0,0]
for curdir in [indir]:
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
            try:
                if 'No data for run' in info1[0]:
                    continue
            except:
                continue
            
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
                run_time.strftime('%Y-%m-%dT%H:%M:%S%z'),
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

general_information['Information fields'] |= set(df.columns)
general_information['Information fields'] = sorted(list(general_information['Information fields']))
general_information['Maximum intensity'] = max(general_information['Maximum intensity'], mi)
general_information['Maximum time'] = max(general_information['Maximum time'], mt)

with open(os.path.join(destdir,'info.json'),'w',encoding='utf-8') as fil:
    json.dump(general_information,fil,indent=4)
for _,row in df.iterrows():
    dic = {
        c: row[c] for c in df.columns
    }
    with open(os.path.join(destdirforjson,f'{row["run_id"]}.json'),'w',encoding='utf-8') as fil:
        json.dump(dic,fil,indent=4)