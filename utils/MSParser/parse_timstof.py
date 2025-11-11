import xmltodict
import os
import pandas as pd
from datetime import datetime
from plotly import io as pio
import sqlite3
from plotly import graph_objects as go
import warnings
from . import ms_name_identifier

def get_directory_size(directory_path):
    """Calculate the total size of all files in a directory recursively."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except (OSError, PermissionError):
        # Handle cases where we can't access some files
        pass
    return total_size

def get_traces(ms1_df, ms2_df):
    """Get traces from a Timstof file.
    
    :param ms1_df: DataFrame containing the MS1 data.
    :param ms2_df: DataFrame containing the MS2 data.
    :returns: Dictionary containing the traces.
    """
    bpc_df = ms1_df[['Time','MaxIntensity','SummedIntensities']]
    msn_df = ms2_df[['Time','MaxIntensity','SummedIntensities']]
    tic_data = []
    bpc_data = []
    msn_data = []
    for i in range(1, int(bpc_df['Time'].max()+2)):
        i_df = bpc_df[(bpc_df['Time']<i) & ((i-1)<bpc_df['Time'])]
        i_df_ms2 = msn_df[(msn_df['Time']<i) & ((i-1)<msn_df['Time'])]
        try:
            bpc_data.append(int(i_df['MaxIntensity'].max()))
        except ValueError:
            bpc_data.append(0)
        try:
            tic_data.append(int(i_df['SummedIntensities'].sum()))
        except ValueError:
            tic_data.append(0)
        try:
            msn_data.append(int(i_df_ms2['SummedIntensities'].sum()))
        except ValueError:
            msn_data.append(0)
    bpc = pd.Series(bpc_data,name='BPC').to_dict()
    tic = pd.Series(tic_data,name='TIC').to_dict()
    MSn = pd.Series(msn_data,name='MSn').to_dict()
    return {'TIC': tic, 'BPC': bpc, 'MS2': MSn}

def parse_file(root, run_name, run_id_regex):
    """Parse a Timstof file.
    
    :param root: Root directory of the Timstof file.
    :param run_name: Name of the run.
    :param run_id_regex: Regular expression to match the run ID.
    :returns: Dictionary containing the parsed data.
    """
    warnings.simplefilter(action="ignore", category=FutureWarning)
    timsfile = os.path.join(root, run_name)
    with sqlite3.connect(os.path.join(timsfile, 'analysis.tdf')) as conn:
        df_ms = pd.read_sql_query('SELECT * FROM Frames', conn)
        df_MS1 = df_ms[df_ms['MsMsType']==0]
        df_MS2 = df_ms[df_ms['MsMsType'] == 8]
        if df_MS2.shape[0] > 0:
            datatype = 'ddaPASEF'
        else:
            datatype = 'diaPASEF'
            df_MS2 = df_ms[df_ms['MsMsType'] == 9]
        metadata = {}
        for _, row in pd.read_sql_query('SELECT * FROM GlobalMetadata', conn).iterrows():
            metadata[row['Key']] = row['Value']

    if os.path.isdir(timsfile):
        if 'SampleInfo.xml' in os.listdir(timsfile):
            try:
                with open(os.path.join(timsfile,'SampleInfo.xml'),encoding='utf-16') as fil:
                    sampleinfo = fil.read()
                    sampleinfo = xmltodict.parse(sampleinfo)
            except:
                sampleinfo = {'SampleTable': {}, 'Sample': {}, 'No_sampletable': 'no sampletable'}

    fdic = {'file_name': run_name, 'file_path': timsfile}
    fdic['file_size'] = get_directory_size(timsfile)
    fdic['files'] = {}
    for root, dirs, files in os.walk(fdic['file_path']):
        for f in files:
            file_stats = os.stat(os.path.join(root,f))
            fp = root.replace(fdic['file_path'],'')
            fdic['files'][fp] = {
                'size': file_stats.st_size,
                'atime': file_stats.st_atime,
                'mtime': file_stats.st_mtime,
                'ctime': file_stats.st_ctime,
            }
            try:
                fdic['files'][fp]['birthtime'] = file_stats.st_birthtime
            except AttributeError:
                fdic['files'][fp]['birthtime'] = ''
    fdic['instrument'] = {
        'inst_name': metadata['InstrumentName'],
        'inst_model': f'{metadata["InstrumentName"]}_{metadata["InstrumentFamily"]}_{metadata["InstrumentRevision"]}',
        'inst_serial_no': metadata['InstrumentSerialNumber'],
        'software_version': metadata['AcquisitionSoftwareVersion'],
        'firmware_version': metadata['AcquisitionFirmwareVersion'],
        'extras': '',
        'name': ms_name_identifier.identify(fdic['file_path'])
    }
    if 'DigitizerType' in metadata:
        fdic['instrument']['digitizer'] = metadata['DigitizerType']
    fdic['run'] = {
        'first_scan_number': int(df_ms['Id'].min()),
        'last_scan_number': int(df_ms['Id'].max()),
        'start_time': float(df_ms['Time'].min())/60,
        'end_time': float(df_ms['Time'].max())/60,
        'run_date': datetime.fromisoformat(metadata['AcquisitionDateTime']).strftime("%Y-%m-%d %H:%M:%S"),
        'number_of_scans': int(df_ms['Id'].max()),
        'method_name': metadata['MethodName'],
    }
    
    infodict = sampleinfo['SampleTable']['Sample']
    if '@MS_Method' in infodict:
        fdic['run']['ms_method'] = infodict['@MS_Method'].rsplit('\\')[-1]
        fdic['run']['full_ms_method'] = infodict['@MS_Method']
    if '@Method' in infodict:
        fdic['run']['full_method'] = infodict['@Method']
    if '@SampleID' in infodict:
        fdic['sample'] = {
            'sample_id':  infodict['@SampleID'],
            'sample_name': infodict['@SampleID'],
        }
        fdic['sample_id'] = fdic['sample']['sample_id']
    else:
        fdic['sample'] = {}
    if '@DataPath' in infodict:
        fdic['sample']['file_name'] = infodict['@DataPath'].rsplit('\\',maxsplit=1)[-1]
    if '@Position' in infodict:
        fdic['sample']['vial'] = infodict['@Position']
    if '@Volume' in infodict:
        fdic['sample']['injection_volume'] = infodict['@Volume']
    if '@Line' in infodict:
        fdic['sample']['row_number'] = infodict['@Line']
    if '@Dilution' in infodict:
        fdic['sample']['dilution_factor'] = infodict['@Dilution']
    if '@DataPath' in infodict:
        fdic['sample']['original_file_path'] = infodict['@DataPath']
    
    fdic['metadata'] = metadata
    fdic['data_type'] = datatype
    fdic['parsed_date'] = datetime.now().strftime('%Y.%m.%d')
    fdic['traces'] = get_traces(df_MS1, df_MS2)
    fdic['smooth_sigma'] = {
        'TIC': 2,
        'BPC': 2,
        'MSn': 2
    }

    return fdic