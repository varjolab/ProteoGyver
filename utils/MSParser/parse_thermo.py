
import os
import pandas as pd
import math
from datetime import datetime
from fisher_py.raw_file_reader import RawFileReaderAdapter, RawFileAccess
from fisher_py.data.business import ChromatogramTraceSettings, TraceType, ChromatogramSignal
from fisher_py.data import Device
from . import ms_name_identifier

def roundup(x: float) -> int:
    """Round up a float to the nearest 100."""
    return int(math.ceil(x / 100.0)) * 100

def get_traces(rawfile: RawFileReaderAdapter) -> dict:
    """Get traces from a raw file.
    
    :param rawfile: Raw file reader adapter.
    :returns: Dictionary containing the traces.
    """
    tic = {}
    bpc = {}
    msn = {}
    for i in range(1, rawfile.run_header_ex.last_spectrum):
        stats = rawfile.get_scan_stats_for_scan_number(i)
        stime = int(stats.start_time*60)
        tic.setdefault(stime, [])
        bpc.setdefault(stime, [0])
        msn.setdefault(stime, [])
        if 'ms2' in stats.scan_type.lower():
            msn[stime].append(stats.tic)
        else:
            tic[stime].append(stats.tic)
            bpc[stime].append(stats.base_peak_intensity)
    sorted_keys = sorted(list(tic.keys()))
    return {
        'TIC': pd.Series([sum(tic[stime]) for stime in sorted_keys], name = 'TIC').to_dict(),
        'BPC': pd.Series([max(bpc[stime]) for stime in sorted_keys], name = 'BPC').to_dict(),
        'MSn': pd.Series([sum(msn[stime]) for stime in sorted_keys], name = 'MSn').to_dict(),
    }
    
def get_scantypes(rawfile: RawFileReaderAdapter) -> dict:
    """Get scantypes from a raw file.
    
    :param rawfile: Raw file reader adapter.
    :returns: Dictionary containing the scantypes.
    """
    scantypes = {}
    splitchrs = '[@'
    for i in range(1, rawfile.run_header_ex.last_spectrum+1):
        try:
            st = []
            for token in rawfile.get_scan_type(i).split():
                found = False
                for schr in splitchrs:
                    found = schr in token
                    if found:
                        break
                if not found:
                    st.append(token)
                else:
                    break
            st = ' '.join(st)
            scantypes.setdefault(st, 0)
            scantypes[st]+=1
        except:
            break
    return scantypes

def deduplicate_nested_dicts(nest_dict: dict) -> dict:
    """Remove nested dicts based on inner dict values.

    :param nest_dict: Dictionary of dictionaries.
    :returns: A new dictionary with duplicates removed (keeps first occurrence, by sorted keys).
    """
    seen = set()
    unique_dict = {}
    sorted_keys = sorted(list(nest_dict.keys()))
    for key in sorted_keys:
        value = nest_dict[key]
        # convert nested dict to a frozenset of (key, value) pairs so it's hashable
        marker = frozenset(value.items())
        if marker not in seen:
            seen.add(marker)
            unique_dict[key] = value
    return unique_dict

def parse_file(data_path, filename) -> dict:
    """Parse a raw file.
    
    :param data_path: Path to the data file.
    :param filename: Name of the data file.
    :returns: Dictionary containing the parsed data.
    """
    before = set(os.listdir('.'))
    fdic = {'file_name': filename, 'file_path': os.path.join(data_path,filename)}
    raw_file = RawFileReaderAdapter.file_factory(fdic['file_path'])
    raw_file.select_instrument(Device.MS, 1)
    fdic['file_size'] = os.stat(os.path.join(data_path, fdic['file_name'])).st_size

    faims_str = ''
    if (raw_file.get_filter_for_scan_number(1).compensation_voltage_count > 0):
        faims_str = 'FAIMS'
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
    fdic['instrument'] = {
        'inst_name': raw_file.get_instrument_data().name,
        'inst_model': raw_file.get_instrument_data().model,
        'inst_serial_no': raw_file.get_instrument_data().serial_number,
        'software_version': raw_file.get_instrument_data().software_version,
        'firmware_version': raw_file.get_instrument_data().hardware_version,
        'extras': faims_str,
        'name': ms_name_identifier.identify(fdic['file_path'])
    }
    fdic['run'] = {
        'first_scan_number': raw_file.run_header_ex.first_spectrum,
        'last_scan_number': raw_file.run_header_ex.last_spectrum,
        'start_time': raw_file.run_header_ex.start_time,
        'end_time': raw_file.run_header_ex.end_time,
        'run_date': raw_file.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
        'mass_resolution': raw_file.run_header_ex.mass_resolution,
        'number_of_scans': raw_file.run_header_ex.spectra_count,
        'low_mass': raw_file.run_header_ex.low_mass,
        'high_mass': raw_file.run_header_ex.high_mass,    
        'method_name': raw_file.sample_information.instrument_method_file.rsplit('/')[-1].rsplit('\\')[-1],
        'full_method': raw_file.sample_information.instrument_method_file,
        'processing_method': raw_file.sample_information.processing_method_file.rsplit('/')[-1].rsplit('\\')[-1],
        'full_processing_method': raw_file.sample_information.processing_method_file
    }
    fdic['sample'] = {
        'file_name': raw_file.file_name,
        'sample_name': raw_file.sample_information.sample_name,
        'sample_id': raw_file.sample_information.sample_id,
        'sample_type': str(raw_file.sample_information.sample_type),
        'comment': raw_file.sample_information.comment,
        'vial': raw_file.sample_information.vial,
        'sample_volume': raw_file.sample_information.sample_volume,
        'injection_volume': raw_file.sample_information.injection_volume,
        'row_number': raw_file.sample_information.row_number,
        'dilution_factor': raw_file.sample_information.dilution_factor,
        'original_file_path': raw_file.path
    }
    real_warn = []
    for line in raw_file.file_error.warning_message.split('\n'):
        if 'mutex' in line:
            continue
        else:
            if len(line.strip())>0:
                real_warn.append(line.strip())
    real_warn = '\n'.join(real_warn)
    fdic['errors'] = {
        'has_error': raw_file.file_error.has_error,
        'has_warning': raw_file.file_error.has_warning,
        'error_code': raw_file.file_error.error_code,
        'error_message': raw_file.file_error.error_message,
        'warning_message': real_warn
    }
    
    fdic['traces'] = get_traces(raw_file)
    fdic['smooth_sigma'] = {
        'TIC': 1,
        'BPC': 0,
        'MSn': 0
    }
    fdic['scantypes'] = get_scantypes(raw_file)
    guessed_datatype = 'unknown'
    scantype_pairs = [
        ['ITMS + c NSI d w Full ms2','DDA'],
        ['FTMS + c NSI Full ms', 'DDA'],
        ['FTMS + c NSI d Full ms2','DDA'],
        ['FTMS + p NSI Full ms','DDA'],
        ['ASTMS + c NSI Full ms2','DDA'],
        ['ASTMS + c NSI d Full ms2','DIA']
    ]
    for stype, data_type in scantype_pairs:
        if stype in fdic['scantypes']:
            guessed_datatype = data_type
            break
    if guessed_datatype == 'unknown':
        if 'DDA' in raw_file.sample_information.instrument_method_file:
            guessed_datatype = 'DDA'
        if 'DIA' in raw_file.sample_information.instrument_method_file:
            guessed_datatype = 'DIA'
    fdic['data_type'] = guessed_datatype# 'DDA' if () else 'DIA'
    fdic['parsed_date'] = datetime.now().strftime('%Y.%m.%d')
    full_inst_dict = {}
    for device_class in [
        Device.Analog,
        Device.MS,
        Device.MSAnalog,
        Device.Other,
        Device.Pda,
        Device.UV
    ]: 
        for i in range(1, 120):
            iname = f'{device_class.name}_{device_class.value}_{i}'
            try:
                raw_file.select_instrument(device_class, i)    
            except:
                break
            full_inst_dict[iname] = {
                'inst_name': raw_file.get_instrument_data().name,
                'inst_model': raw_file.get_instrument_data().model,
                'inst_serial_no': raw_file.get_instrument_data().serial_number,
                'software_version': raw_file.get_instrument_data().software_version,
                'firmware_version': raw_file.get_instrument_data().hardware_version
            }
    fdic['all_instrument_information'] = full_inst_dict    
    fdic['sample_id'] = fdic['sample']['sample_id'].strip()
    after = set(os.listdir('.'))
    for f in (after-before):
        if 'READWRITE_INFO' in f:
            os.remove(f)
    return fdic