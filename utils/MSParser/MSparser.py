import os
import sys

# Add the directory containing this script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import parse_thermo
import parse_timstof
import traceback
import pandas as pd
from plotly import io as pio
from plotly import graph_objects as go
import json
from scipy.ndimage import gaussian_filter1d
import tomlkit

def read_toml(toml_file):
    with open(toml_file, 'r') as tf:
        data = tomlkit.load(tf)
    return data

def count_intercepts(xydata):
    mean = xydata.mean()
    intercepts = 0
    intercept_times = []
    prev = -1
    for time, intensity in xydata.items():
        low,high=sorted([prev,intensity])
        if (low<mean) & (high>mean):
            intercepts += 1
            intercept_times.append(time)
        prev = intensity
    return (intercepts, intercept_times)

def calculate_auc(ser):
    auc = 0.0
    prev = 0
    for time, intval in ser.items():
        auc += time*intval + ((time-prev)*intval/2)
        prev = time
    return auc

def parse_raw(root, filename):
    data_dict = parse_thermo.parse_file(root, filename)
    return data_dict

def parse_d(root, filename, run_id_regex):
    data_dict = parse_timstof.parse_file(root, filename, run_id_regex)
    return data_dict

def handle_data(data_dict, oname):
    new_traces = {}
    sample_id_number = data_dict['sample_id']
    
    for tracetype, tracedict in data_dict['traces'].items():
        ser = pd.Series(tracedict)
        intercepts, intercept_times = count_intercepts(ser)
        new_traces[f'{tracetype}_raw'] = ser.to_dict()
        new_traces[f'{tracetype}_auc'] = calculate_auc(ser)
        new_traces[f'{tracetype}_intercepts'] = intercepts
        new_traces[f'{tracetype}_intercept_times'] = intercept_times
        new_traces[f'{tracetype}_maxtime'] = int(ser.index.max())
        new_traces[f'{tracetype}_mean_intensity'] = float(ser.mean())
        new_traces[f'{tracetype}_max_intensity'] = int(ser.max())
        sigma = data_dict['smooth_sigma'][tracetype] # This depends on the data, and should be manually figured out
        if sigma > 0:
            ser = pd.Series(gaussian_filter1d(ser, sigma))
            new_traces[tracetype] = ser.to_dict()
        new_traces[f'{tracetype} trace'] = pio.to_json(go.Scatter(x=ser.index, y=ser.values,name=sample_id_number))
    for k,v in new_traces.items():
        data_dict['traces'][k] = v
    with open(oname, 'w', encoding='utf-8') as fil:
        json.dump(data_dict, fil, indent=2)

def analyze(filename, outdir, errorfile):
    filename = filename.rstrip(os.sep)
    if os.sep in filename:
        root_dir, filename = filename.rsplit(os.sep,maxsplit=1)
    else:
        root_dir = '.'
    tims_run_id_regex = '^(\\d+)(?:(_Tomppa))?'
    os.makedirs(outdir, exist_ok=True)
    try:
        oname = os.path.join(outdir, f'{filename.lower()}.json')
        data = None
        if filename.lower().endswith('raw'):
            data = parse_raw(root_dir, filename)
        elif filename.lower().endswith('.d'):
            data = parse_d(root_dir, filename, tims_run_id_regex)
        if data is not None:
            handle_data(data, oname)
            print(filename, 'done')
        else:
            print(filename, 'data none')
        return 0
    except Exception as e:
        name = type(e).__name__
        details = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        with open(errorfile, 'a') as fil:
            from datetime import datetime
            fil.write(f'{datetime.now()}:: {name}:\nFilename:{filename}\nDetails:{details}\n-=-=-=-=-=-=-=-=-=-=-=-=-=-\n')
        print(f'{filename} ERRORED')
        return 1

def main():
    filename, outdir, errorfile = sys.argv[1:]
    analyze(filename, outdir, errorfile)
if __name__ == '__main__':
    main()