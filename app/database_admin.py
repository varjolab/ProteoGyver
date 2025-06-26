import multiprocessing
import os
import sqlite3
import shutil
from components import db_functions
from components.tools import utils
import database_updater
import database_generator
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

def clean_database(versions_to_keep_dict) -> None:
    names = [k for k in versions_to_keep_dict.keys() if not '_' in k]
    for name in names:
        path = versions_to_keep_dict[name + '_path']
        regex = versions_to_keep_dict[name + '_regex']
        folders = os.listdir(path)
        folders = [(re.match(regex, file).group(1), file) for file in folders]
        folders = sorted(folders, key=lambda x: x[0], reverse=True)
        for folder in folders[versions_to_keep_dict[name]:]:
            print('Removing', os.path.join(path, folder[1]))
            shutil.rmtree(os.path.join(path, folder[1]))

def last_update(conn: sqlite3.Connection, uptype: str, interval: int, time_format: str) -> datetime:
    try:
        last_update = datetime.strptime(db_functions.get_last_update(conn, uptype), time_format)
    except Exception as e:
        last_update = datetime.now() - relativedelta(seconds=interval+1)
    return last_update

if __name__ == "__main__":
    parameters = utils.read_toml('parameters.toml')
    time_format = parameters['Config']['Time format']
    timestamp = datetime.now().strftime(time_format)
    if parameters['Config']['CPU count limit'] == 'ncpus':
        ncpu: int = multiprocessing.cpu_count()
    else:
        ncpu = parameters['Config']['CPU count limit']
    db_path = os.path.join(*parameters['Data paths']['Database file'])
    organisms = set(parameters['Database creation']['Organisms to include in database'])
    # # Connect to the database (create it if it doesn't exist)
    tmpdir = parameters['Database creation']['Temporary directory for database generation']
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)
    if not os.path.exists(db_path):
        print('Database file does not exist, generating database')
        database_generator.generate_database(parameters['Database creation'], db_path, time_format, timestamp, tmpdir, ncpu, organisms)
        print('Database generated')
    else:
        # Export a snapshot, if required:
        cc_cols = parameters['Database creation']['Control and crapome db detailed columns']
        cc_types = parameters['Database creation']['Control and crapome db detailed types']
        ms_runs_parameters = parameters['Database creation']['MS runs information']
        
        parameters = parameters['Database updater']
        update_interval = int(parameters['Update interval seconds'])
        snapshot_interval = int(parameters['Database snapshot settings']['Snapshot interval days'])*24*60*60
        api_update_interval = int(parameters['External data update interval days'])*24*60*60
        clean_interval = int(parameters['Database clean interval days'])*24*60*60
        output_dir = os.path.join(*parameters['Tsv templates directory'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        conn: sqlite3.Connection = db_functions.create_connection(db_path) # type: ignore

        last_external_update_date = last_update(conn, 'external', api_update_interval, time_format)

        do_snapshot = last_update(db_path, 'snapshot', snapshot_interval, time_format) < (datetime.now() - relativedelta(seconds=snapshot_interval))
        do_external_update = last_update(conn, 'external', api_update_interval, time_format) < (datetime.now() - relativedelta(seconds=api_update_interval))
        do_main_db_update = last_update(conn, 'main_db_update', update_interval, time_format) < (datetime.now() - relativedelta(seconds=update_interval))
        do_clean_update = last_update(conn, 'clean', clean_interval, time_format) < (datetime.now() - relativedelta(seconds=clean_interval))
        updates_to_do = [update for update in [
            'External' if do_external_update else '',
            'Main db' if do_main_db_update else '',
            'Clean' if do_clean_update else '',
            'Snapshot' if do_snapshot else ''
        ] if update]
        if len(updates_to_do) > 0:
            print('Going to do updates:', ', '.join(updates_to_do))
        else:
            print('No updates to do')
        if do_snapshot:
            snapshot_dir = os.path.join(*parameters['Database snapshot settings']['Snapshot dir'])
            snapshots_to_keep = parameters['Database snapshot settings']['Snapshots to keep']
            print('Exporting snapshot')
            db_functions.export_snapshot(db_path, snapshot_dir, snapshots_to_keep)
            database_updater.update_log_table(conn, ['snapshot snapshot'], [1], timestamp, 'snapshot')

        if do_external_update:
            print('Updating external data')
            database_updater.update_external_data(conn, parameters, timestamp, organisms, last_external_update_date, ncpu)
            database_updater.update_log_table(conn, ['external update'], [1], timestamp, 'external')
        if do_main_db_update:
            print('Updating database')
            database_updater.update_ms_runs(conn, ms_runs_parameters, timestamp, time_format, os.path.join(*parameters['Update files']['ms_runs']))
            inmod_names, inmod_vals = database_updater.update_database(conn, parameters, cc_cols, cc_types, timestamp)
            database_updater.update_log_table(conn, inmod_names, inmod_vals, timestamp, 'main_db_update')
            db_functions.generate_database_table_templates_as_tsvs(conn, output_dir, parameters['Database table primary keys'])
        if do_clean_update:
            print('Cleaning database')
            clean_database(parameters['Versions to keep'])
            database_updater.update_log_table(conn, ['clean update'], [1], timestamp, 'clean')
        conn.close() # type: ignore
        print('Database update done.')