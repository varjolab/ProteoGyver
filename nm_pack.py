import pandas as pd
import os
import numpy as np
from datetime import datetime
from pynmonanalyzer.pyNmonParser import pyNmonParser
from time import sleep

indir = 'nmonstats'
outdir = '.'
allfiles = [
    f for f in os.listdir(indir) if 'nmon' in f
]

def pynmon_to_frames(dict_list):
    frames = {}
    for pynmon_dict in dict_list:
        for key, results in pynmon_dict.items():
            heads = results[0][1:]
            index = [r[0] for r in results[1:]]
            data = [r[1:] for r in results[1:]]
            datakey = (key, results[0][0])
            if datakey not in frames:
                frames[datakey] = []
            frames[datakey].append(pd.DataFrame(data=data,index=index,columns=heads).T)
    return {datakey: pd.concat(frames[datakey]) for datakey in frames.keys()}

sleep(30) # Wait for any nmon to finish writing full output. 

nmon_list = []
for fname in allfiles:
    p = pyNmonParser(os.path.join(indir, fname))
    rd = p.parse()
    nmon_list.append(rd)
resdic = pynmon_to_frames(nmon_list)
wanted = ['start_time','end_time','resource','resource_details','data','max','q1','q2','q3','avg','median','stdev']
discard_cols = ['Busy','Steal%','bigfree','hightotal','highfree','lowfree','buffers','lowtotal','memshared']
skipped_names = ['DISKBSIZE','DISKWRITE','DISKXFER','DISKREAD','DISKBUSY','JFSFILE','PROC','VM','NETPACKET']
data = []
all_starts = []
for (resource_name, detailed_name), df in resdic.items():
    if resource_name in skipped_names:continue
    df = df.sort_index().drop(columns=[c for c in df.columns if c in discard_cols])
    start_time = datetime.strptime(df.index[0],'%d-%b-%Y %H:%M:%S')
    all_starts.append(start_time)
    start = datetime.strftime(start_time, '%Y-%m-%d_%H-%M-%S')
    end = datetime.strftime(datetime.strptime(df.index[-1],'%d-%b-%Y %H:%M:%S'), '%Y-%m-%d_%H-%M-%S')
    for c in df.columns:
        cvals = df[c].astype(float)
        row = [
            start,
            end,
            resource_name,
            detailed_name,
            c, 
            cvals.max(),
            cvals.quantile(0.25),
            cvals.quantile(0.5),
            cvals.quantile(0.75),
            cvals.mean(),
            cvals.median(),
            cvals.std()
        ]
        data.append(row)
parsed = pd.DataFrame(data=data,columns=wanted)
parsed = parsed[~(parsed[parsed.columns[4:]].replace(0, np.nan).isna().sum(axis=1) == len(parsed.columns[4:]))]
parsed.to_pickle(os.path.join(outdir, f'{min(all_starts).strftime("%Y-%m-%d_%H-%M-%S")}_parsed.pickle'))
for f in allfiles:
    os.remove(os.path.join(indir, f))
    
    