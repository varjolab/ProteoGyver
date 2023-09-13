import sqlite3
from typing import Union
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

def create_connection(db_file, error_file: str = None) -> Union[sqlite3.Connection , None]:
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
def get_from_table(conn:sqlite3.Connection, table_name: str, criteria_col:str = None, criteria:str = None, select_col:str = None, as_pandas:bool = False) -> Union[list, pd.DataFrame]:
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
        selection_string: str = f"SELECT {select_col} FROM {table_name} WHERE {criteria_col}='{criteria}'"            
    else:
        selection_string = f'SELECT {select_col} FROM {table_name}'
    if as_pandas:
        ret = pd.read_sql_query(selection_string, conn)
    else:
        cursor.execute(selection_string)
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
    ret_list: list
    if protein_list is None:
        ret_list = get_from_table(conn, 'contaminants', select_col='uniprot_id')
    else:
        ret_list = get_from_table(conn, 'contaminants', select_col='uniprot_id', criteria_col='uniprot_id', criteria=protein_list)
    ret_list = [r[0] for r in ret_list]
    conn.close()
    return ret_list
