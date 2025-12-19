import os
import pymzml
import pandas as pd
from datetime import datetime, timedelta
from xml.etree.ElementTree import ParseError
from contextlib import redirect_stdout
import io


def get_traces(rawfile):
    tic = {}
    bpc = {}
    msn = {}
    spc_num = 0
    lowmass = -1
    highmass = -1
    scantypes = set()
    # Janky bullshit
    while True:
        try:
            spectrum = next(rawfile)
            stime = round(spectrum.scan_time_in_minutes()*60)
            scantype = spectrum.ms_level
            tic.setdefault(stime, [])
            bpc.setdefault(stime, [0])
            msn.setdefault(stime, [])
            try:
                s_lowmass,s_highmass = spectrum.extreme_values('mz')
                if lowmass == -1:
                    lowmass = s_lowmass
                    highmass = s_highmass
                if lowmass > s_lowmass:
                    highmass = s_highmass
                if highmass < s_highmass:
                    highmass = s_highmass
                if scantype == 2:
                    msn[stime].append(spectrum.TIC)
                else:
                    tic[stime].append(spectrum.TIC)
                    bpc[stime].append(max(spectrum.extreme_values('i')))
            except ValueError:
                continue
            spc_num += 1
            try:
                for i, k in spectrum.get_element_by_name('filter string').items():
                    if 'Full' in k:
                        toks = k.split()
                        for i in range(len(toks)):
                            if '@' in toks[i] or '[' in toks[i]:
                                scantypes.add(' '.join(toks[:i]))
                                break
                        else:
                            scantypes.add(k)
            except:
                continue
        except StopIteration:
            break
        except ParseError:
            continue
    sorted_keys = sorted(list(tic.keys()))
    return (
        {
            'TIC': pd.Series([sum(tic[stime]) for stime in sorted_keys], name = 'TIC').to_dict(),
            'BPC': pd.Series([max(bpc[stime]) for stime in sorted_keys], name = 'BPC').to_dict(),
            'MSn': pd.Series([sum(msn[stime]) for stime in sorted_keys], name = 'MSn').to_dict(),
        }, 
        spc_num, lowmass, highmass, scantypes
    )

def read_mzml(filepath):
    """Method to read mzml with workarounds for encountered problems"""
    rf = None
    output_buffer = io.StringIO()
    with redirect_stdout(output_buffer):
        rf = pymzml.run.Reader(filepath)
    output = output_buffer.getvalue()
    if output.strip() == '[Warning] Not index found and build_index_from_scratch is False':
        with redirect_stdout(output_buffer):
            rf = pymzml.run.Reader(filepath, build_index_from_scratch=True)
        output2 = output_buffer.getvalue()
    return rf

def parse_file(data_path, filename) -> dict:
    fdic = {'file_name': filename, 'file_path': os.path.join(data_path,filename)}
    raw_file = read_mzml(fdic['file_path'])
    fdic['file_size'] = os.stat(os.path.join(data_path, fdic['file_name'])).st_size
    file_stats = os.stat(fdic['file_path'])
    fdic['files'] = {
            'rawfile': {
                'size': file_stats.st_size,
                'atime': file_stats.st_atime,
                'mtime': file_stats.st_mtime,
                'ctime': file_stats.st_ctime,
                'path': ''
            }
        }
    try:
        fdic['files']['rawfile']['birthtime'] = file_stats.st_birthtime
    except AttributeError:
        fdic['files']['rawfile']['birthtime'] = ''
    info = raw_file.info
    try:
        start_time = datetime.strptime(info['start_time'], '%Y-%m-%dT%H:%M:%SZ')
    except (TypeError, ValueError):
        start_time = datetime.now()
    fdic['instrument'] = {
        'inst_name': 'unknown',
        'inst_model': 'unknown',
        'inst_serial_no': 'unknown',
        'software_version': 'unknown',
        'firmware_version': 'unknown',
        'extras': '',
        'name': 'unknown'
    }
    traces, num_spectra, lowest_mz, highest_mz, scantypes = get_traces(raw_file)
    fdic['run'] = {
        'first_scan_number': 0,
        'last_scan_number': info['spectrum_count'],
        'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'end_time': (start_time + timedelta(minutes = max(traces['TIC'].keys()))).strftime("%Y-%m-%d %H:%M:%S"),
        'run_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'number_of_scans': num_spectra,
        'low_mass': lowest_mz,
        'high_mass': highest_mz, 
        'method_name': 'unknown',
        'full_method': 'unknown',
        'processing_method': 'unknown',
        'full_processing_method': 'unknown',
    }
    run_name = info['run_id']
    sample_id = ''
    try:
        for inf in info['sample_list_element']:
            for name, value in inf.items():
                if name == 'id':
                    sample_id = value
    except KeyError:
        sample_id = filename
    fdic['sample'] = {
        'file_name': filename,
        'sample_name': run_name,
        'sample_id': sample_id,
    }
    fdic['traces'] = traces
    fdic['smooth_sigma'] = {
        'TIC': 1,
        'BPC': 0,
        'MSn': 0
    }
    guessed_datatype = 'unknown'
    scantype_pairs = [
        ['+ c NSI d Full ms2','DDA'],
        ['+ c NSI Full ms2','DIA'],
    ]
    for stype, data_type in scantype_pairs:
        if stype in scantypes:
            guessed_datatype = data_type
            break
    fdic['data_type'] = guessed_datatype
    fdic['parsed_date'] = datetime.now().strftime('%Y.%m.%d')
    fdic['sample_id'] = fdic['sample']['sample_id'].strip()
    return fdic