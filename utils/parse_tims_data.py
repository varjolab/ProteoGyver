import re
import xmltodict
import sys
import alphatims.bruker
import os
import pandas as pd
from datetime import datetime
import traceback
from pathlib import Path
from plotly import io as pio
from plotly import graph_objects as go
import json
from scipy.ndimage import gaussian_filter1d
import tomlkit

def read_toml(toml_file):
    toml_path = Path(toml_file)
    with toml_path.open('r', encoding='utf-8') as tf:
        data = tomlkit.load(tf)
    return data


def count_intercepts(xydata):
    mean = xydata.mean()
    intercepts = 0
    intercept_ser = []
    prev = -1
    for time, intensity in xydata.items():
        low,high=sorted([prev,intensity])
        if (low<mean) & (high>mean):
            intercepts += 1
            intercept_ser.append(time)
        prev = intensity
    intercept_ser = pd.Series(index=intercept_ser,data=mean)
    return (intercepts, intercept_ser.to_dict())

def calculate_auc(ser):
    auc = 0.0
    prev = 0
    for time, intval in ser.items():
        auc += time*intval + ((time-prev)*intval/2)
        prev = time
    return auc

# Unused
def smooth_tic(tic_ser: pd.Series, sigma: int = 6):
    return pd.Series(gaussian_filter1d(tic_ser, sigma))

def handle_timsfile(root, run_name, run_id_regex):
    timsfile = os.path.join(root, run_name)
    alphatims_file = alphatims.bruker.TimsTOF(timsfile,drop_polarity=False,convert_polarity_to_int=True)
    data = {}
    basedf = alphatims_file.frames.query('MsMsType == 8') # ddapasef
    data['DataType'] = 'DDAPASEF'
    msname = 'Timppa'
    if 'Tomppa' in run_name:
        msname = 'Tomppa'
    data['MSname'] = msname
    if basedf.shape[0] == 0:
        basedf = alphatims_file.frames.query('MsMsType == 9')
        data['DataType'] = 'DIAPASEF'
    basedf2 = alphatims_file.frames.query('MsMsType == 0')  # MS1
    i = -10
    data['Frames'] = alphatims_file.frames[['Id','SummedIntensities','NumPeaks']].to_dict(orient='split')
    try:
        data['NumPrecursors'] = alphatims_file.precursors.shape[0]
    except AttributeError:
        print('failed')
        pass
    sampleinfo = ['']
    sample_name = run_name
    datafolder_name = run_name
    if os.path.isdir(timsfile):
        if 'SampleInfo.xml' in os.listdir(timsfile):
            try:
                with open(os.path.join(timsfile,'SampleInfo.xml'),encoding='utf-16') as fil:
                    sampleinfo = fil.read()
                    sampleinfo = xmltodict.parse(sampleinfo)
                    sample_name = sampleinfo['SampleTable']['Sample']['@SampleID']
                    datafolder_name = sampleinfo['SampleTable']['Sample']['@DataPath'].rsplit('/',maxsplit=1)[-1].rsplit('\\',maxsplit=1)[-1]
            except:
                sampleinfo = ['no sampleinfo']
        data['HasMGF'] = False
        for f in os.listdir(timsfile):
            if f.split('.')[-1].lower()=='mgf':
                data['HasMGF'] = True
    data['SampleInfo'] = sampleinfo
    data['ParsedDate'] = datetime.now().strftime('%Y.%m.%d')
    sample_id_number = 'NoID'
    if datafolder_name.startswith(sample_name):
        # Some early cases where the ID number is at the very end of the folder name
        sample_id_number = datafolder_name.rsplit('_',maxsplit=1)[-1].replace('.d','')
    else:
        match = re.match(run_id_regex, datafolder_name)
        if match:
            num_id, tomppa = match.groups()
            result = num_id + (tomppa if tomppa else '')
            sample_id_number = result
    data['SampleID'] = sample_id_number
    
    for polarity in basedf['Polarity'].unique():
        sers = {}
        sdata = {}
        for _, row in basedf[basedf['Polarity']==polarity].iterrows():
            if row['Time'] > i:
                i = int(row['Time'])
                sdata[i] = []
            frame_df = alphatims_file[int(row['Id'])]
            sdata[i].append(frame_df[frame_df['mz_values'].between(350,2200,inclusive='both')]['intensity_values'].sum())
        ind = sorted(list(sdata.keys()))
        vals = [sum(sdata[i]) for i in ind]
        pser = pd.Series(index=ind, data=vals)
        sers['bpc filtered df'] = pser.to_dict()
        
        sdata = {}
        i = -10
        for _, row in basedf[basedf['Polarity']==polarity].iterrows():
            if row['Time'] > i:
                i = int(row['Time'])
                sdata[i] = []
            frame_df = alphatims_file[int(row['Id'])]
            sdata[i].append(frame_df['intensity_values'].sum())
        ind = sorted(list(sdata.keys()))
        vals = [sum(sdata[i]) for i in ind]
        pser = pd.Series(index=ind, data=vals)
        sers['bpc unfiltered df'] = pser.to_dict()
        
        sdata = {}
        i = -10
        for _, row in basedf2[basedf2['Polarity']==polarity].iterrows():
            if row['Time'] > i:
                i = int(row['Time'])
                sdata[i] = []
            frame_df = alphatims_file[int(row['Id'])]
            sdata[i].append(frame_df['intensity_values'].sum())
        ind = sorted(list(sdata.keys()))
        vals = [sum(sdata[i]) for i in ind]
        pser = pd.Series(index=ind, data=vals)
        sers['tic df'] = pser.to_dict()
        new_set = {}
        maxtimeval = 0
        for serkey, serdf in sers.items():
            ser = pd.Series(serdf)
            new_set[serkey] = {}
            maxtimeval = max(maxtimeval, ser.index.max())
            intercepts, intercept_dict = count_intercepts(ser)
            smoothed = smooth_tic(ser)
            new_set[serkey]['Series'] = ser.to_dict()
            new_set[serkey]['Series_smooth'] = smoothed.to_dict()
            new_set[serkey]['trace'] = pio.to_json(go.Scatter(x=ser.index, y=ser.values,name=sample_id_number))
            new_set[serkey]['trace_smooth'] = pio.to_json(go.Scatter(x=smoothed.index, y=smoothed.values,name=sample_id_number))
            new_set[serkey]['auc'] = calculate_auc(ser)
            new_set[serkey]['mean_intensity'] = float(ser.mean())
            new_set[serkey]['max_intensity'] = int(ser.max())
            new_set[serkey]['intercepts'] = intercepts
            new_set[serkey]['intercept_dict'] = intercept_dict
        data[f'polarity_{polarity}'] = new_set
    return data

if len(sys.argv) != 5:
    print('Input: indir, outdir, error file, parameters file')
indir, outdir, errorfile, parameters_file = sys.argv[1:]
if not os.path.isdir(outdir):
    os.makedirs(outdir)
parameters = read_toml(Path(parameters_file))['Database creation']['MS runs information']
run_id_regex = parameters['MS run ID regex']
for root, dirs, files in os.walk(indir):
    for d in dirs:
        if d.endswith('.d'):
            if os.path.isfile(os.path.join(outdir,f'{d}.json')):
                continue
            try:
                data = handle_timsfile(root, d, run_id_regex)
                with open(os.path.join(outdir,f'{d}.json'),'w',encoding='utf-8') as fil:
                    json.dump(data, fil, indent=2)
            except Exception as e:
                with open(errorfile,'a', encoding="utf-8") as fil:
                    fil.write(f'================================\nFailed:\nroot:{root}\nFile:{d}\n---Errormessage---\n{traceback.format_exc()}\n================================\n')
                continue
