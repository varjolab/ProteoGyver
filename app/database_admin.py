"""Administrative entrypoints and helpers for database lifecycle operations.

Tasks include schema creation, snapshotting, external data updates, and
periodic cleanup of old versions.
"""

import multiprocessing
import os
import sqlite3
from components.tools import utils
from typing import Iterable, Optional
import shutil
from pathlib import Path
from components import db_functions
from components.tools import utils
import database_updater
import re
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

def clean_database(versions_to_keep_dict) -> None:
    """Remove old database directories, keeping a configured number of versions.

    :param versions_to_keep_dict: Mapping with keys '<name>' (keep count),
        '<name>_path' (path list), and '<name>_regex' (folder regex with group 1
        sortable for recency).
    :returns: None.
    """
    names = [k for k in versions_to_keep_dict.keys() if not '_' in k]
    for name in names:
        path = os.path.join(*versions_to_keep_dict[name + '_path'])
        regex = versions_to_keep_dict[name + '_regex']
        folders = os.listdir(path)
        folders = [(re.match(regex, file).group(1), file) for file in folders]
        folders = sorted(folders, key=lambda x: x[0], reverse=True)
        for folder in folders[versions_to_keep_dict[name]:]:
            print('Removing', os.path.join(path, folder[1]))
            shutil.rmtree(os.path.join(path, folder[1]))

def create_sqlite_from_schema(schema_file: str | Path,
                              db_file: str | Path,
                              overwrite: bool = False,
                              pragmas: Optional[Iterable[str]] = ("foreign_keys=ON","journal_mode=WAL")) -> Path:
    """Create a SQLite database from a .sql schema file.

    :param schema_file: Path to the schema file.
    :param db_file: Path of the database to create.
    :param overwrite: Whether to overwrite an existing DB file.
    :param pragmas: PRAGMAs to apply after connecting (e.g., ("foreign_keys=ON",)).
    :returns: Absolute path to the created database.
    :raises FileNotFoundError: If ``schema_file`` does not exist.
    :raises FileExistsError: If ``db_file`` exists and ``overwrite`` is False.
    :raises sqlite3.Error: If executing the schema fails.
    """
    schema_path = Path(schema_file)
    db_path = Path(db_file)

    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    if db_path.exists():
        if overwrite:
            db_path.unlink()
        else:
            raise FileExistsError(f"Database already exists: {db_path} (use overwrite=True)")

    # Read with UTF-8-SIG to gracefully handle a BOM; normalize line endings.
    sql = schema_path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    con = sqlite3.connect(db_path)
    try:
        if pragmas:
            for p in pragmas:
                con.execute(f"PRAGMA {p}")
        con.executescript(sql)   # Executes multiple CREATE TABLE/INDEX/etc. statements
        con.commit()
    except sqlite3.Error:
        con.close()
        # Remove partially-created DB on failure
        try:
            db_path.unlink()
        except Exception:
            pass
        raise
    finally:
        con.close()

    return db_path.resolve()

def last_update(conn: sqlite3.Connection, uptype: str, interval: int, time_format: str) -> datetime:
    """Return the last update time for a given update type or a default.

    If the log lookup fails, defaults to now minus ``interval`` seconds.

    :param conn: SQLite database connection.
    :param uptype: Update type label to query (e.g., 'external').
    :param interval: Interval in seconds to compute a safe default.
    :param time_format: Timestamp format string used in the log table.
    :returns: Datetime of the last update or a computed default.
    """
    try:
        last_update = datetime.strptime(db_functions.get_last_update(conn, uptype), time_format)
        print(uptype, 'last update:', last_update)
    except Exception as e:
        last_update = datetime.now() - relativedelta(seconds=interval+1)
    return last_update

def main():
    """Main entry point for database administration.

    :returns: None.
    """
    force_full_update = False
    parameters = utils.read_toml(Path('config/parameters.toml'))
    time_format = parameters['Config']['Time format']
    if '--force-full-update' in sys.argv:
        force_full_update = True
        print('Forcing full database update')
    timestamp = datetime.now().strftime(time_format)
    if parameters['Config']['CPU count limit'] == 'ncpus':
        ncpu: int = multiprocessing.cpu_count()
    else:
        ncpu = parameters['Config']['CPU count limit']
    db_path = os.path.join(*parameters['Data paths']['Database file'])
    organisms = set(parameters['Database creation']['Organisms to include in database'])
    output_dir = os.path.join(*parameters['Database updater']['Tsv templates directory'])
    os.makedirs(output_dir,exist_ok=True)
    for table_name, path in parameters['Database updater']['Update files'].items():
        os.makedirs(os.path.join(*path), exist_ok = True)
    need_full_update = False
    if not os.path.exists(db_path):
        print('Database file does not exist, generating database')
        dbfile = os.path.join(*parameters['Data paths']['Database file'])
        need_to_create = True
        need_full_update = True
        if 'Minimal database file' in parameters['Data paths']:
            print('Checking for minimal database file')
            minimal_db_path = os.path.join(*parameters['Data paths']['Minimal database file'])
            if os.path.exists(minimal_db_path):
                print('Minimal database file found, using it')
                shutil.copy(minimal_db_path, db_path)
                need_to_create = False
            else:
                print('Minimal database file not found, creating database')
        if need_to_create:
            print('Creating database')
            schema_file = os.path.join(*parameters['Data paths']['Schema file'])
            create_sqlite_from_schema(schema_file, dbfile) # type: ignore
            conn: sqlite3.Connection = db_functions.create_connection(db_path, mode='rw') # type: ignore
            database_updater.update_log_table(conn, ['db creation'], [1], timestamp, 'created')
            db_functions.generate_database_table_templates_as_tsvs(conn, output_dir, parameters['Database updater']['Database table primary keys'])
            conn.close()
    
    # Export a snapshot, if required:
    cc_cols = parameters['Database creation']['Control and crapome db detailed columns']
    cc_types = parameters['Database creation']['Control and crapome db detailed types']
    
    parameters = parameters['Database updater']
    update_interval = int(parameters['Update interval minutes'])*60
    snapshot_interval = int(parameters['Database snapshot settings']['Snapshot interval days'])*24*60*60
    api_update_interval = int(parameters['External data update interval days'])*24*60*60
    clean_interval = int(parameters['Database clean interval days'])*24*60*60
    conn: sqlite3.Connection = db_functions.create_connection(db_path, mode='rw') # type: ignore

    last_external_update_date = last_update(conn, 'external', api_update_interval, time_format)
    do_snapshot = need_full_update or (last_update(conn, 'snapshot', snapshot_interval, time_format) < (datetime.now() - relativedelta(seconds=snapshot_interval)))
    do_external_update = need_full_update or (last_update(conn, 'external', api_update_interval, time_format) < (datetime.now() - relativedelta(seconds=api_update_interval)))
    do_main_db_update = need_full_update or (last_update(conn, 'main_db_update', update_interval, time_format) < (datetime.now() - relativedelta(seconds=update_interval)))
    do_clean_update = need_full_update or (last_update(conn, 'clean', clean_interval, time_format) < (datetime.now() - relativedelta(seconds=clean_interval)))
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
    if force_full_update or do_snapshot:
        snapshot_dir = os.path.join(*parameters['Database snapshot settings']['Snapshot dir'])
        snapshots_to_keep = parameters['Database snapshot settings']['Snapshots to keep']
        print('Exporting snapshot')
        db_functions.export_snapshot(db_path, snapshot_dir, snapshots_to_keep)
        database_updater.update_log_table(conn, ['snapshot snapshot'], [1], timestamp, 'snapshot')
    if force_full_update or do_external_update:
        print('Updating external data')
        database_updater.update_external_data(conn, parameters, timestamp, organisms, last_external_update_date, ncpu)
        database_updater.update_log_table(conn, ['external update'], [1], timestamp, 'external')
    if force_full_update or do_main_db_update:
        print('Updating database')
        inmod_names, inmod_vals = database_updater.update_database(conn, parameters, cc_cols, cc_types, timestamp)
        database_updater.update_log_table(conn, inmod_names, inmod_vals, timestamp, 'main_db_update')
        db_functions.generate_database_table_templates_as_tsvs(conn, output_dir, parameters['Database table primary keys'])
    if force_full_update or do_clean_update:
        print('Cleaning database')
        clean_database(parameters['Versions to keep'])
        database_updater.update_log_table(conn, ['clean update'], [1], timestamp, 'clean')
    
    conn.close() # type: ignore
    print('Database update done.')


if __name__ == "__main__":
    main()