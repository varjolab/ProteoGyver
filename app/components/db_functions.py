import sqlite3
import pandas as pd
import os
import csv
from datetime import datetime


def map_protein_info(uniprot_ids: list, info: list | str = None, placeholder: list | str = None, db_file_path: str|None = None):
    """Map information from the protein table.
    :param uniprot_ids: IDs to map. If ID is not found, placeholder value will be used
    :param info: if str, returned list will have only the mapped values from this column. If type is list, will return a list of lists. By default, will return gene_name column data.
    :param placeholder: Value to use if ID is not found. By default, value from the uniprot_ids list will be used. If type list, the placeholders should be in the same order as info list. 
    """
    ret_info = []
    if info is None:
        info = 'gene_name'
    if isinstance(info, str):
        info = [info]
    if placeholder is None:
        placeholder = 'PLACEHOLDER_IS_INPUT_UPID'
    if isinstance(placeholder, str):
        placeholder = [placeholder for _ in info]
    return_mapping = {}
    if db_file_path is None:
        for u in uniprot_ids:
            return_mapping[u] = ['Not in database' for _ in info]
    else:
        for _, row in get_from_table_by_list_criteria(
                create_connection(db_file_path),
                'proteins',
                'uniprot_id',
                uniprot_ids,
            ).iterrows():
            return_mapping[row['uniprot_id']] = [row[ic] for ic in info]
    for uniprot_id in (set(uniprot_ids)-set(return_mapping.keys())):
        return_mapping[uniprot_id] = [
            uniprot_id if placeholder[i]=='PLACEHOLDER_IS_INPUT_UPID' else placeholder[i] 
            for i in range(len(info))
        ]
    retlist = [
        return_mapping[upid] 
        for upid in uniprot_ids
    ]
    if len(retlist[0]) == 1:
        retlist = [r[0] for r in retlist]
    return retlist

def get_full_table_as_pd(db_conn, table_name, index_col: str|None = None, filter_col: str|None = None, startswith: str|None = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    
    if filter_col and startswith is not None:
        query += f" WHERE {filter_col} LIKE '{startswith}%'"

    return pd.read_sql_query(query, db_conn, index_col=index_col)

def get_last_update(conn, uptype: str) -> str:
    """
    Get the last update timestamp from the update_log table.

    Args:
        conn: SQLite database connection object
        uptype: update type to get the last update for
    Returns:
        str: Last update timestamp
    """
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM update_log WHERE update_type = ? ORDER BY timestamp DESC LIMIT 1", (uptype,))
    last_update = cursor.fetchone()
    cursor.close()
    return last_update[0]

def is_test_db(db_path: str) -> bool:
    conn = create_connection(db_path)
    cursor = conn.cursor()
    try:
        cur = conn.execute("SELECT value FROM metadata WHERE key='is_test'")
        result = cur.fetchone()
        return result and result[0].lower() == 'true'
    except sqlite3.Error:
        return False

def export_snapshot(source_path: str, snapshot_dir: str, snapshots_to_keep: int) -> None:
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source DB file not found: {source_path}")
    os.makedirs(snapshot_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dbname = os.path.basename(source_path)
    snapshot_filename = f"backup_{dbname}_{timestamp}.db"
    snapshot_path = os.path.join(snapshot_dir, snapshot_filename)

    with sqlite3.connect(source_path) as source_conn:
        with sqlite3.connect(snapshot_path) as snapshot_conn:
            source_conn.backup(snapshot_conn)
    print(f"Snapshot created: {snapshot_path}")

    # Cleanup
    if snapshots_to_keep is not None:
        backups = sorted(
            (f for f in os.listdir(snapshot_dir) if f.startswith("backup_") and f.endswith(".db")),
            key=lambda f: os.path.getmtime(os.path.join(snapshot_dir, f))
        )
        excess = len(backups) - snapshots_to_keep
        for old_file in backups[:excess]:
            old_path = os.path.join(snapshot_dir, old_file)
            try:
                os.remove(old_path)
                print(f"Deleted old backup: {old_path}")
            except Exception as e:
                print(f"Failed to delete {old_path}: {e}")

def dump_full_database_to_csv(database_file, output_directory) -> None:
    conn: sqlite3.Connection = create_connection(database_file) # type: ignore
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables:list = cursor.fetchall()
    for table_name in tables:
        table_name: str = table_name[0]
        table: pd.DataFrame = get_full_table_as_pd(table_name, conn)
        table.to_csv(os.path.join(output_directory, f'{table_name}.tsv'),sep='\t', index_label='index')
    cursor.close()
    conn.close()

def list_tables(database_file) -> list[str]:
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    cursor.close()
    conn.close()
    return [table[0] for table in tables]

def add_column(db_conn, tablename, colname, coltype):
    sql_str: str = f"""
        ALTER TABLE {tablename} 
        ADD COLUMN {colname} '{coltype}'
        """
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        cursor.execute(sql_str)
    except sqlite3.Error as error:
        print("Failed to add column to sqlite table", error, sql_str)
        raise
    finally:
        cursor.close()

def modify_multiple_records(db_conn, table, updates):
    """Modify multiple records in a table.
    
    Args:
        db_conn: SQLite database connection
        table (str): Name of the table to update
        updates (list): List of dictionaries, each containing:
            - criteria_col (str): Column name for WHERE clause
            - criteria: Value to match in WHERE clause
            - columns (list): List of column names to update
            - values (list): List of values corresponding to columns
    """
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        for update in updates:
            add_str = f'UPDATE {table} SET {" = ?, ".join(update["columns"])} = ? WHERE {update["criteria_col"]} = ?'
            add_data = update["values"].copy()
            add_data.append(update["criteria"])
            cursor.execute(add_str, add_data)
    except sqlite3.Error as error:
        print("Failed to modify sqlite table", error)
        raise
    finally:
        cursor.close()

# Original function maintained for backwards compatibility
def modify_record(db_conn, table, criteria_col, criteria, columns, values):
    update = {
        "criteria_col": criteria_col,
        "criteria": criteria,
        "columns": columns,
        "values": values
    }
    modify_multiple_records(db_conn, table, [update])
    return f'UPDATE {table} SET {" = ?, ".join(columns)} = ? WHERE {criteria_col} = ?'

def remove_column(db_conn, tablename, colname):
    sql_str: str = f"""
        ALTER TABLE {tablename} 
        DROP COLUMN {colname}
        """
    try:
        # Create a cursor object
        cursor: sqlite3.Cursor = db_conn.cursor()
        cursor.execute(sql_str)
    except sqlite3.Error as error:
        print("Failed to remove column from sqlite table", error, sql_str)
        raise
    finally:
        cursor.close()

def rename_column(db_conn, tablename, old_col, new_col):
    sql_str = f'ALTER TABLE {tablename} RENAME COLUMN {old_col} TO {new_col};'
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        cursor.execute(sql_str)
    except sqlite3.Error as error:
        print(f'Failed to rename column in sqlite table. Error: {error}', sql_str)
        raise

def delete_multiple_records(db_conn, table, deletes):
    """Delete multiple records from a table.
    
    Args:
        db_conn: SQLite database connection
        table (str): Name of the table to delete from
        deletes (list): List of dictionaries, each containing:
            - criteria_col (str): Column name for WHERE clause
            - criteria: Value to match in WHERE clause
    """
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        for delete in deletes:
            sql_str = f"""DELETE from {table} where {delete["criteria_col"]} = ?"""
            cursor.execute(sql_str, [delete["criteria"]])
    except sqlite3.Error as error:
        print("Failed to delete records from sqlite table", error)
        raise
    finally:
        cursor.close()

def delete_record(db_conn, tablename, criteria_col, criteria):
    delete = {
        "criteria_col": criteria_col,
        "criteria": criteria
    }
    delete_multiple_records(db_conn, tablename, [delete])

def add_record(db_conn, tablename, column_names, values):
    sql_str: str = f"""
        INSERT INTO {tablename} ({", ".join(column_names)}) VALUES ({", ".join(["?" for _ in column_names])})
        """
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        cursor.execute(sql_str, values)
    except sqlite3.Error as error:
        print("Failed to add record to sqlite table", error, sql_str)
        raise
    finally:
        cursor.close()

def add_multiple_records(db_conn, tablename, column_names, list_of_values) -> None:
    sql_str: str = f"""
        INSERT INTO {tablename} ({", ".join(column_names)}) VALUES ({", ".join(["?" for _ in column_names])})
    """
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        cursor.executemany(sql_str, list_of_values)
    except sqlite3.Error as error:
        print("Failed to add multiple records to sqlite table", error, sql_str)
        raise
    finally:
        cursor.close()
        
def create_connection(db_file, error_file: str|None = None):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :error_file: file to write errors to. If none, print errors to output
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Exception as e:
        if error_file is None:
            print(e)
        else:
            with open(error_file,'a') as fil:
                fil.write(str(e)+'\n')
    return conn

def generate_database_table_templates_as_tsvs(db_conn, output_dir, primary_keys):
    """Generate TSV templates for each table in an SQLite database.

    Args:
        db_conn (sqlite3.Connection): Connection to the SQLite3 database.
        output_dir (str): Directory to save the TSV template files.
        primary_keys (dict): Dictionary containing primary keys for each database table that a template is generated for.
    
    Notes:
    - The db_conn is not closed after the function is called.
    - The primary keys are used to ensure that the correct columns are included in the TSV file.
    - The TSV files are saved in the output directory.
    """

    # Connect to the SQLite database
    cursor = db_conn.cursor()

    # Get the list of tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    # Discard tables that are not in the parameter file
    tables = [table for table in tables if table in primary_keys]

    if not tables:
        print("No tables found in the database.")
        return

    for table in tables:
        # Get column names for the table
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [row[1] for row in cursor.fetchall()]
        if not columns:
            print(f"Table '{table}' has no columns.")
            continue

        primary_key = primary_keys.get(table)
        if primary_key not in columns:
            print(f"Primary key '{primary_key}' specified for table '{table}' is not a valid column.")
            continue
        columns.insert(0, primary_key)  # Ensure the primary key is the first column
        tsv_file_path = os.path.join(output_dir, f"{table}.tsv")
        with open(tsv_file_path, "w", encoding="utf-8") as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter="\t")
            tsv_writer.writerow(columns)
        print(f"Template generated for table '{table}' at '{tsv_file_path}'.")
    cursor.close()

def get_from_table(conn:sqlite3.Connection, table_name: str, criteria_col:str|None = None, criteria:str|tuple|None = None, select_col:str|None = None, as_pandas:bool = False, pandas_index_col:str|None = None, operator:str = '=') -> list[tuple] | pd.DataFrame:
    """Get data from a table in an SQLite database.

    Args:
        conn (sqlite3.Connection): Connection to the SQLite3 database.
        table_name (str): Name of the table to get data from.
        criteria_col (str|None): Column name to use for the WHERE clause.
        criteria (str|None): Value to match in the WHERE clause.
        select_col (str|None): Column name to select. If None, all columns are selected.
        as_pandas (bool): If True, return a pandas DataFrame. If False, return a list of tuples.
        pandas_index_col (str|None): Column name to use as the index of the pandas DataFrame.
        operator (str): Operator to use in the WHERE clause.
    """
    assert (((criteria is not None) & (criteria_col is not None)) |\
             ((criteria is None) & (criteria_col is None))),\
             'Both criteria and criteria_col must be supplied, or both need to be none.'
    
    if select_col is None:
        select_col = '*'
    elif isinstance(select_col, list):
        select_col = f'{", ".join(select_col)}'
    if criteria_col is not None:
        placeholder = '?'
        if isinstance(criteria, tuple):
            params = criteria
            placeholder = '? AND ?'
        else:
            params = (criteria,)
        query = f"SELECT {select_col} FROM {table_name} WHERE {criteria_col} {operator} {placeholder}"
    else:
        query = f"SELECT {select_col} FROM {table_name}"
        params = ()

    if as_pandas:
        result = pd.read_sql_query(query, conn, params=params, index_col=pandas_index_col)  # type: ignore
    else:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()  # type: ignore
        result = [r[0] for r in result]
        cursor.close()
    return result

def get_from_table_by_list_criteria(conn:sqlite3.Connection, table_name: str, criteria_col:str, criteria:list,as_pandas:bool = True, select_col: str = None):
    """"""
    cursor: sqlite3.Cursor = conn.cursor()
    if select_col is None:
        select_col = '*'
    if isinstance(select_col, list):
        select_col = f'({", ".join(select_col)})'
    query: str = f'SELECT {select_col} FROM {table_name} WHERE {criteria_col} IN ({", ".join(["?" for _ in criteria])})'
    if as_pandas:
        ret: pd.DataFrame = pd.read_sql_query(query, con=conn, params=criteria)
    else:
        cursor.execute(query, criteria)
        ret: list = cursor.fetchall()
    cursor.close()
    return ret

def get_contaminants(db_file: str, protein_list:list = None, error_file: str = None) -> list:
    """Retrieve contaminants from a database.
    :param db_file: database file
    :protein_list: if list is supplied, only return contaminants found in the list
    :error_file: file to write errors to. If none, print errors to output
    :return: list of contaminants
    """
    conn: sqlite3.Connection = create_connection(db_file, error_file)
    ret_list: list = get_from_table(conn, 'contaminants', select_col='uniprot_id')
    conn.close()
    return ret_list

def drop_table(conn:sqlite3.Connection, table_name: str) -> None:
    """Drop a table from the database.
    :param conn: database connection
    :param table_name: name of the table to drop
    """
    sql_str: str = f'DROP TABLE IF EXISTS {table_name}'
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql_str)
    cursor.close()


def get_table_column_names(db_conn, table_name: str) -> list[str]:
    """Get the column names for a table in an SQLite database.

    Args:
        db_conn (sqlite3.Connection): Connection to the SQLite3 database.
        table_name (str): Name of the table to get the column names for.
    
    Notes:
    - The db_conn is not closed after the function is called.
    """

    # Connect to the SQLite database
    cursor = db_conn.cursor()

    # Get the list of tables in the database
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [row[1] for row in cursor.fetchall()]
    cursor.close()
    return columns
