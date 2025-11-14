import os
import json
import pandas as pd
import shutil
import logging
import zipfile
from celery import shared_task
from datetime import datetime
from pathlib import Path
from components.tools.utils import read_toml, normalize_key
from components import db_functions
from celery_once import QueueOnce

logger = logging.getLogger("MS_run_json_parser")

@shared_task(base=QueueOnce, once={'graceful': True})
def parse_json_files():
    """Parse MS run JSON files and emit TSVs for database updater.

    Reads configuration from ``parameters.toml``, processes JSON files in the
    configured input directory, writes two TSVs (``ms_runs`` and ``ms_plots``)
    with timestamped filenames, and optionally moves/compresses processed
    JSONs according to settings.

    :returns: Status string indicating number of files processed and locations.
    """
    logger.info("Starting parsing of json files")
    root_dir = Path(__file__).resolve().parents[1]
    parameters_path = os.path.join(root_dir, 'config','parameters.toml')
    parameters = read_toml(Path(parameters_path))
    input_path = os.path.join(*parameters['Maintenance']['MS run parsing']['Input files'])
    os.makedirs(input_path, exist_ok=True)
    jsons_to_do = os.listdir(input_path)
    jsons_to_do = [j for j in jsons_to_do if j.endswith('.json')]
    if len(jsons_to_do) == 0:
        return ('No json files to parse')
    move_done_dir = os.path.join(input_path, parameters['Maintenance']['MS run parsing']['Move done jsons into subdir'])
    if move_done_dir == input_path:
        move_done_dir = None
    compress_done_when_filecount_over = parameters['Maintenance']['MS run parsing']['Compress done when filecount over']
    trace_keys = ['internal_run_id']
    for tt in ['BPC','MSn','TIC']:
        trace_keys.extend([
            f'{tt}_auc',
            f'{tt}_maxtime',
            f'{tt}_max_intensity',
            f'{tt}_mean_intensity',
            f'{tt}_trace',
        ])

    base_keys = [
        'data_type',
        'file_name',
        'file_size',
        'parsed_date',
        'sample_id'
    ]
    instrument_to_main = [
        'inst_model',
        'inst_serial_no',
        'inst_name',
        #'extras'
    ]
    sample_to_main = [
        'sample_name'
    ]
    run_to_main = [
        #'processing_method',
        #'method_name',
        #'ms_method',
        'run_date',
        'start_time',
        'end_time',
        'last_scan_number'
    ]
    headers = [ 'internal_run_id' ] + base_keys + [ 'file_name_clean' ]
    headers.extend(['run_' + c for c in run_to_main])
    headers.extend([c for c in sample_to_main])
    headers.extend(['inst_' + c for c in instrument_to_main])
    for rep in ['run','inst']:
        headers = [h.replace(f'{rep}_{rep}_', f'{rep}_') for h in headers]
    MS_rows = []
    trace_rows = []

    print_i = 0
    done_jsons = []
    for file in jsons_to_do:
        print_i +=1
        if not os.path.isfile(os.path.join(input_path,file)):
            continue
        with open(os.path.join(input_path, file)) as fil:
            dic = json.load(fil)
        internal_run_ID = f'REPLACE_WITH_INTERNAL_RUN_ID'
            
        new_row = [ internal_run_ID]
        for bk in base_keys:
            if bk in dic:
                new_row.append(dic[bk])
            else:
                new_row.append('NA')
        new_row.append(normalize_key(dic['file_name']))
        for bk in run_to_main:
            if bk in dic['run']:
                new_row.append(dic['run'][bk])
            else:
                new_row.append('NA')
        for bk in sample_to_main:
            if bk in dic['sample']:
                new_row.append(dic['sample'][bk])
            else:
                new_row.append('NA')
        for bk in instrument_to_main:
            if bk in dic['instrument']:
                new_row.append(dic['instrument'][bk])
            else:
                new_row.append('NA')
        trace_row = [ internal_run_ID ]
        if trace_keys is None:
            trace_keys = sorted(list(dic['traces']))
        for tk in trace_keys:
            if tk == 'internal_run_id': continue
            if tk not in dic['traces']:
                print(f'{file} {tk} not in dic["traces"]')
                trace_row.append('placeholder')
            else:
                trace_row.append(dic['traces'][tk])
        MS_rows.append(new_row)
        trace_rows.append(trace_row)
        if print_i % 1000 == 0: 
            logger.info(f'{print_i} files done')
        done_jsons.append(file)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    ms_runs_dir = os.path.join(*parameters['Database updater']['Update files']['ms_runs'])
    ms_plots_dir = os.path.join(*parameters['Database updater']['Update files']['ms_plots'])
    os.makedirs(ms_runs_dir, exist_ok=True)
    os.makedirs(ms_plots_dir, exist_ok=True)
    
    msrows_filename = os.path.join(ms_runs_dir, f'{timestamp}_jsonParser_MSruns.tsv')
    tracerows_filename = os.path.join(ms_plots_dir, f'{timestamp}_jsonParser_MStraces.tsv')
    i = 0
    while os.path.exists(msrows_filename):
        i+=1
        msrows_filename = os.path.join(ms_runs_dir, f'{timestamp}_jsonParser_MSruns_{i}.tsv')
        tracerows_filename = os.path.join(ms_plots_dir, f'{timestamp}_jsonParser_MStraces_{i}.tsv')

    pd.DataFrame(data = MS_rows, columns=headers).to_csv(msrows_filename,sep='\t',index=False)
    pd.DataFrame(data = trace_rows, columns=trace_keys).to_csv(tracerows_filename,sep='\t',index=False)
    compress_str = ''
    if move_done_dir:
        os.makedirs(move_done_dir, exist_ok=True)
        for file in done_jsons:
            shutil.move(os.path.join(input_path, file), os.path.join(move_done_dir, file))
        jsons_in_dir = [j for j in os.listdir(move_done_dir) if j.endswith('.json')]
        move_str = f"Moved: {len(done_jsons)} files to {move_done_dir}."
        if len(jsons_in_dir) > compress_done_when_filecount_over:
            with zipfile.ZipFile(os.path.join(move_done_dir, f'{timestamp}_done_jsons.zip'), 'w') as zipf:
                for file in jsons_in_dir:
                    zipf.write(os.path.join(move_done_dir, file), file)
            for file in jsons_in_dir:
                os.remove(os.path.join(move_done_dir, file))
            compress_str =  f"Compressed: {len(jsons_in_dir)} files to {os.path.join(input_path, f'{timestamp}_done_jsons.zip')}."
    else:
        for file in done_jsons:
            os.remove(os.path.join(input_path, file))
        move_str = f"Deleted: {len(done_jsons)} files from {input_path}."
    logger.info('Parsing of json files complete')
    return f"Parsing of json files complete. {move_str} {compress_str}"