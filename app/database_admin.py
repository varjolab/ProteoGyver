import os
import sqlite3
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
        files = os.listdir(path)
        files = [(re.match(regex, file).group(1), file) for file in files]
        files = sorted(files, key=lambda x: x[0], reverse=True)
        for file in files[versions_to_keep_dict[name]:]:
            print('Removing', os.path.join(path, file[1]))
            os.remove(os.path.join(path, file[1]))

def last_update(conn: sqlite3.Connection, uptype: str, interval: int, time_format: str) -> datetime:
    try:
        last_update = datetime.strptime(database_updater.get_last_update(conn, uptype), time_format)
    except Exception as e:
        last_update = datetime.now() - relativedelta(seconds=interval+1)
    return last_update

if __name__ == "__main__":
    parameters = utils.read_toml('parameters.toml')
    time_format = parameters['Config']['Time format']
    db_path = os.path.join(*parameters['Data paths']['Database file'])
    organisms = set(parameters['Database creation']['Organisms to include in database'])
    # # Connect to the database (create it if it doesn't exist)
    if not os.path.exists(db_path):
        print('Database file does not exist, generating database')
        database_generator.generate_database(parameters['Database creation'], db_path, time_format, organisms)
        print('Database generated')
    else:
        timestamp = datetime.now().strftime(time_format)
        cc_cols = parameters['Database creation']['control and crapome db detailed columns']
        cc_types = parameters['Database creation']['control and crapome db detailed types']
        ms_runs_parameters = parameters['Database creation']['MS runs information']
        
        parameters = parameters['Database updater']
        update_interval = int(parameters['Update interval seconds'])
        api_update_interval = int(parameters['External data update interval days'])*24*60*60
        clean_interval = int(parameters['Database clean interval days'])*24*60*60
        
        output_dir = os.path.join(*parameters['Tsv templates directory'])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        conn: sqlite3.Connection = db_functions.create_connection(db_path) # type: ignore
        if last_update(conn, 'external', api_update_interval, time_format) < (datetime.now() - relativedelta(seconds=api_update_interval)):
            print('Updating external data')
            database_updater.update_external_data(conn, parameters, timestamp, organisms)
            database_updater.update_log_table(conn, ['external'], [1], timestamp, 'external')
        if last_update(conn, 'main_db_update', update_interval, time_format) < (datetime.now() - relativedelta(seconds=update_interval)):
            print('Updating database')
            database_updater.update_ms_runs(conn, ms_runs_parameters, timestamp, time_format, os.path.join(*parameters['Update files']['ms_runs']))
            inmod_names, inmod_vals = database_updater.update_database(conn, parameters, cc_cols, cc_types, timestamp)
            database_updater.update_log_table(conn, inmod_names, inmod_vals, timestamp, 'main_db_update')
            db_functions.generate_database_table_templates_as_tsvs(conn, output_dir, parameters['Database table primary keys'])
        if last_update(conn, 'clean', clean_interval, time_format) < (datetime.now() - relativedelta(seconds=clean_interval)):
            print('Cleaning database')
            clean_database(parameters['Versions to keep'])
            database_updater.update_log_table(conn, ['clean'], [1], timestamp, 'clean')
        conn.close() # type: ignore
        print('Database updated')