import sqlite3
import pandas as pd
import os
import csv

def get_full_table_as_pd(db_conn, table_name, index_col: str|None = None, filter_col: str|None = None, startswith: str|None = None) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    
    if filter_col and startswith is not None:
        query += f" WHERE {filter_col} LIKE '{startswith}%'"

    return pd.read_sql_query(query, db_conn, index_col=index_col)

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
        
def create_connection(db_file, error_file: str = None):
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

def get_from_table(conn:sqlite3.Connection, table_name: str, criteria_col:str|None = None, criteria:str|None = None, select_col:str|None = None, as_pandas:bool = False, pandas_index_col:str|None = None, operator:str = '=') -> list[tuple] | pd.DataFrame:
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

    if criteria_col is not None:
        query = f"SELECT {select_col} FROM {table_name} WHERE {criteria_col} {operator} ?"
        params = (criteria,)
    else:
        query = f"SELECT {select_col} FROM {table_name}"
        params = ()

    if as_pandas:
        result = pd.read_sql_query(query, conn, params=params, index_col=pandas_index_col)  # type: ignore
    else:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()  # type: ignore
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
    ret_list: list = [
        r[0] for r in get_from_table(conn, 'contaminants', select_col='uniprot_id')
    ]
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
