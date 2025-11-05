import os
import sys

from typing import Optional, Dict
import re
import unidecode
import parse_thermo
import parse_timstof
import traceback
import pandas as pd
from plotly import io as pio
from plotly import graph_objects as go
import json
from scipy.ndimage import gaussian_filter1d
import tomlkit


# Add the directory containing this script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def remove_accent_characters(text: str) -> str:
    """Replace accented characters with their unaccented equivalents.

    :param text: Input string containing accented characters.
    :returns: String with accented characters replaced by unaccented equivalents.
    """
    return unidecode.unidecode(text)

def replace_special_characters(
    text: str,
    replacewith: str = '.',
    dict_and_re: bool = False,
    replacement_dict: Optional[Dict[str, str]] = None,
    stripresult: bool = True,
    remove_duplicates: bool = False,
    make_lowercase: bool = True,
    allow_numbers: bool = True,
    allow_space: bool = False,
    mask_first_digit: str|None = None
) -> str:
    """Replace special characters in a string with specified replacements.

    :param text: Input string containing special characters.
    :param replacewith: Character to use for replacement.
    :param dict_and_re: Whether to apply both dictionary replacements and regex.
    :param replacement_dict: Mapping of specific substrings to replacements.
    :param stripresult: Strip whitespace and replacement characters from result.
    :param remove_duplicates: Collapse consecutive replacement characters.
    :param make_lowercase: Convert result to lowercase.
    :param allow_numbers: Allow numbers in the result.
    :param allow_space: Allow spaces in the result.
    :param mask_first_digit: Character to prefix when first char is a digit.
    :returns: String with special characters replaced.
    """
    ret: str
    regex_pat = r'[^a-zA-Z0-9]'
    if allow_space:
        regex_pat = r'[^a-zA-Z0-9 ]'
    if not allow_numbers:
        regex_pat = regex_pat.replace('0-9', '')
    if not replacement_dict:
        ret = re.sub(regex_pat, replacewith, text)
    else:
        # Sort replacement keys by length (longest first) to handle overlapping patterns
        for key in sorted(list(replacement_dict.keys()), key=lambda x: len(x), reverse=True):
            if key in text:
                text = text.replace(key, replacement_dict[key])
        if dict_and_re:
            ret = re.sub(regex_pat, replacewith, text)
        else:
            new_text: list[str] = []
            for character in text:
                if not character.isalnum():
                    new_text.append(replacewith)
                else:
                    new_text.append(character)
            ret = ''.join(new_text)

    if stripresult:
        curlen: int = -1
        while len(ret) != curlen:
            curlen = len(ret)
            ret = ret.strip()
            ret = ret.strip(replacewith)
    if remove_duplicates:
        curlen: int = -1
        while len(ret) != curlen:
            curlen = len(ret)
            ret = ret.replace(f'{replacewith}{replacewith}', replacewith)
    if make_lowercase:
        ret = ret.lower()
    if mask_first_digit:
        if ret[0].isdigit():
            ret = mask_first_digit + ret
    return ret

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
    sample_id_number = data_dict['sample']['file_name']
    data_dict['file_name_clean'] = replace_special_characters(remove_accent_characters(data_dict['sample']['file_name'].rsplit('.',maxsplit=1)[0]), replacewith='_', allow_numbers=True)
    
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
        new_traces[f'{tracetype}_trace'] = pio.to_json(go.Scatter(x=ser.index, y=ser.values,name=sample_id_number))
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