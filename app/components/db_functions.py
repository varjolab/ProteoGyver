import sqlite3
import pandas as pd
import os

def get_full_table_as_pd(db_conn, table_name, index_col: str = None) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * from {table_name}', db_conn, index_col=index_col)

def dump_full_database_to_csv(database_file, output_directory) -> None:
    conn: sqlite3.Connection = create_connection(database_file)
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
    conn.close()
    return [table[0] for table in tables]

def add_column(db_conn, tablename, colname, coltype):
    sql_str: str = f"""
        ALTER TABLE {tablename} 
        ADD COLUMN {colname} '{coltype}'
        """
    try:
        db_conn.cursor().execute(sql_str)
    except sqlite3.Error as error:
        print("Failed to add column to sqlite table", error, sql_str)
        raise

def modify_record(db_conn, table, criteria_col, criteria, columns, values):
    try:
        add_str = f'UPDATE {table} SET {" = ?, ".join(columns)} = ? WHERE {criteria_col} = ?'
        add_data = values.copy()
        add_data.append(criteria)
        db_conn.cursor().execute(add_str, add_data)
        return add_str
    except sqlite3.Error as error:
        print("Failed to modify sqlite table", error, add_str)
        raise

def remove_column(db_conn, tablename, colname):
    sql_str: str = f"""
        ALTER TABLE {tablename} 
        DROP COLUMN {colname}
        """
    try:
        # Create a cursor object
        db_conn.cursor().execute(sql_str)
    except sqlite3.Error as error:
        print("Failed to remove column from sqlite table", error, sql_str)
        raise

def rename_column(db_conn, tablename, old_col, new_col):
    sql_str = f'ALTER TABLE {tablename} RENAME COLUMN {old_col} TO {new_col};'
    try:
        db_conn.cursor().execute(sql_str)
    except sqlite3.Error as error:
        print(f'Failed to rename column in sqlite table. Error: {error}', sql_str)
        raise

    
def delete_record(db_conn, tablename, criteria_col, criteria):
    sql_str: str = f"""DELETE from {tablename} where {criteria_col} = ?"""
    try:
        db_conn.cursor().execute(sql_str, [criteria])
    except sqlite3.Error as error:
        print("Failed to delete record from sqlite table", error, sql_str)
        raise

def add_record(db_conn, tablename, column_names, values):
    sql_str: str = f"""
        INSERT INTO {tablename} ({", ".join(column_names)}) VALUES ({", ".join(["?" for _ in column_names])})
        """
    try:
        db_conn.cursor().execute(sql_str, values)
    except sqlite3.Error as error:
        print("Failed to add record to sqlite table", error, sql_str)
        raise
        
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

def get_from_table(conn:sqlite3.Connection, table_name: str, criteria_col:str = None, criteria:str = None, select_col:str = None, as_pandas:bool = False, pandas_index_col:str = None, operator:str = '='):
    """"""
    cursor: sqlite3.Cursor = conn.cursor()
    if select_col is None:
        select_col = '*'
    assert (((criteria is not None) & (criteria_col is not None)) |\
             ((criteria is None) & (criteria_col is None))),\
             'Both criteria and criteria_col must be supplied, or both need to be none.'

    if isinstance(select_col, list):
        select_col = ', '.join(select_col)
    if criteria_col is not None:
        selection_string: str = f'SELECT {select_col} FROM {table_name} WHERE {criteria_col} {operator} {criteria}'
    else:
        selection_string = f'SELECT {select_col} FROM {table_name}'
    if as_pandas:
        ret = pd.read_sql_query(selection_string, conn,index_col=pandas_index_col)
    else:
        cursor.execute(selection_string)
        ret: list = cursor.fetchall()
    return ret

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
    conn.cursor().execute(sql_str)

