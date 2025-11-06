import sqlite3
import pandas as pd
import os
import csv
from datetime import datetime
from typing import Iterable, Dict, Any, List, Optional, Tuple


def get_database_versions(db_file_path: str) -> dict:
    """Get the version of the database.
    
    :param db_file_path: Path to the database file.
    :returns: Version of the database.
    """
    conn: sqlite3.Connection = create_connection(db_file_path)

    cursor = conn.cursor()
    cursor.execute('SELECT update_type FROM update_log')
    update_types = set([
        r[0] for r in cursor.fetchall()
    ])
    cursor.close()

    versions = { }
    for update_type in update_types:
        versions[update_type] = get_last_update(conn, update_type)
    conn.close()
    return versions

def map_protein_info(uniprot_ids: list, info: list | str = None, placeholder: list | str = None, db_file_path: str|None = None):
    """Map requested columns from the ``proteins`` table for UniProt IDs.

    :param uniprot_ids: UniProt IDs to map; order is preserved in result.
    :param info: Column name or list of column names to return (default ``'gene_name'``).
    :param placeholder: Placeholder(s) for missing IDs; str or list aligned to ``info``.
    :param db_file_path: SQLite DB path; if None, all IDs are treated as missing.
    :returns: List (or list of lists) with mapped values per input ID.
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
    if len(uniprot_ids) == 0:
        return []
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
    
def get_from_table_match_with_priority(
    conn: sqlite3.Connection,
    criteria_list: Iterable[str],
    table: str,
    criteria_cols: List[str],
    *,
    case_insensitive: bool = False,
    key_col: Optional[str] = None,
    extra_tiebreak: Optional[List[Tuple[str, str]]] = None,  # e.g. [("acq_time","DESC")]
    return_cols: Optional[List[str]] = None,  # default: all columns in table
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Find the best-matching row per value using priority columns.

    :param conn: SQLite connection.
    :param criteria_list: Values to match.
    :param table: Table name.
    :param criteria_cols: Columns to try in priority order.
    :param case_insensitive: If ``True``, match case-insensitively.
    :param key_col: Column ensuring deterministic ordering; defaults to ``rowid``.
    :param extra_tiebreak: Extra (column, direction) order terms.
    :param return_cols: Columns to return (default all).
    :returns: Mapping value -> row dict (or None if no match).
    :raises ValueError: For invalid table/columns.
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- validate table & columns
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    if cur.fetchone() is None:
        raise ValueError(f"Table not found: {table}")

    cur.execute(f'PRAGMA table_info("{table}")')
    table_cols = [r["name"] for r in cur.fetchall()]

    missing = [c for c in criteria_cols if c not in table_cols]
    if missing:
        raise ValueError(f"Columns not in {table}: {missing}")

    if return_cols is None:
        ret_cols = table_cols[:]  # all
    else:
        bad = [c for c in return_cols if c not in table_cols]
        if bad:
            raise ValueError(f"return_cols not in {table}: {bad}")
        ret_cols = return_cols[:]  # may be empty!

    if key_col is None:
        key_col = "rowid"
    elif key_col not in table_cols and key_col.lower() != "rowid":
        raise ValueError(f"key_col '{key_col}' not in {table} (or rowid)")

    # --- create helpful indexes (idempotent)
    for col in criteria_cols:
        cur.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_{col}" ON "{table}"("{col}");')

    # --- temp input table
    cur.execute("DROP TABLE IF EXISTS temp.criteria_list;")
    cur.execute("CREATE TEMP TABLE temp.criteria_list(val TEXT PRIMARY KEY)")
    cur.executemany("INSERT OR IGNORE INTO temp.criteria_list(val) VALUES (?)",
                    ((v,) for v in criteria_list))

    # --- build the UNION ALL (keyed) with a robust select list
    # Always include: pri, the match key, plus any columns needed for ordering (key_col, extra_tiebreak cols)
    needed_for_order: set[str] = set()
    if key_col.lower() != "rowid":
        needed_for_order.add(key_col)
    if extra_tiebreak:
        for col, direction in extra_tiebreak:
            if col not in table_cols:
                raise ValueError(f"extra_tiebreak column not in {table}: {col}")
            if direction.upper() not in ("ASC", "DESC"):
                raise ValueError("extra_tiebreak direction must be 'ASC' or 'DESC'")
            needed_for_order.add(col)

    # The columns we actually SELECT in the CTE:
    # - the requested return columns (may be empty), PLUS
    # - the needed ordering columns (so ORDER BY works), deduped.
    select_payload_cols = []
    seen = set()
    for c in ret_cols + [c for c in needed_for_order if c not in ret_cols]:
        if c.lower() == "rowid":  # rowid is implicit; we won’t list it here
            continue
        if c not in seen:
            select_payload_cols.append(c)
            seen.add(c)

    collate = " COLLATE NOCASE" if case_insensitive else ""

    unions = []
    for pri, col in enumerate(criteria_cols, start=1):
        # Base select list
        select_bits = [f'"{col}" AS key', f"{pri} AS pri"]

        # key_col for ordering (as a stable alias) – include rowid explicitly if used
        if key_col.lower() == "rowid":
            select_bits.append("rowid AS __ord_key")
        else:
            select_bits.append(f'"{key_col}" AS __ord_key')

        # extra tie-break columns (if any)
        for tcol in (c for c in needed_for_order if c != key_col and c.lower() != "rowid"):
            select_bits.append(f'"{tcol}"')

        # return columns (may be empty) – avoid duplicating ones we already added
        for rc in ret_cols:
            if rc == key_col or rc.lower() == "rowid" or rc in needed_for_order:
                continue
            select_bits.append(f'"{rc}"')

        select_list_sql = ", ".join(select_bits)

        unions.append(
            f'SELECT {select_list_sql} FROM "{table}" WHERE "{col}" IS NOT NULL'
        )

    union_sql = "\nUNION ALL\n".join(unions)

    # --- ORDER BY for ranking
    order_terms = ["k.pri", "k.__ord_key"]
    if extra_tiebreak:
        for col, direction in extra_tiebreak:
            order_terms.append(f'k."{col}" {direction.upper()}')
    order_by = ", ".join(order_terms)

    sql = f"""
    WITH keyed AS (
      {union_sql}
    ),
    ranked AS (
      SELECT
        cl.val AS query_val,
        k.*,
        ROW_NUMBER() OVER (
          PARTITION BY cl.val
          ORDER BY {order_by}
        ) AS rnk
      FROM temp.criteria_list cl
      LEFT JOIN keyed k
        ON k.key{collate} = cl.val{collate}
    )
    SELECT * FROM ranked WHERE rnk = 1;
    """

    # Execute & build results
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    for row in cur.execute(sql):
        q = row["query_val"]
        if row["pri"] is None:
            results[q] = None
        else:
            # Return only requested columns (defaulted earlier), not the helper fields
            out: Dict[str, Any] = {}
            for c in ret_cols:
                out[c] = row[c]
            # If caller passed return_cols=[], return {} for matches
            results[q] = out

    # Ensure all inputs present
    cur.execute("SELECT val FROM temp.criteria_list;")
    for (v,) in cur.fetchall():
        results.setdefault(v, None)

    return results



def get_full_table_as_pd(db_conn, table_name, index_col: str|None = None, filter_col: str|None = None, startswith: str|None = None) -> pd.DataFrame:
    """Read an entire table into a pandas DataFrame with optional prefix filter.

    :param db_conn: SQLite connection.
    :param table_name: Table name to read.
    :param index_col: Column to set as index.
    :param filter_col: Column to apply ``LIKE 'startswith%'`` on.
    :param startswith: Prefix for the filter.
    :returns: DataFrame converted to pandas nullable dtypes.
    """
    query = f"SELECT * FROM {table_name}"
    
    if filter_col and startswith is not None:
        query += f" WHERE {filter_col} LIKE '{startswith}%'"

    return pd.read_sql_query(query, db_conn, index_col=index_col).convert_dtypes()

def get_last_update(conn, uptype: str) -> str:
    """Get the last update timestamp of a given type from ``update_log``.

    :param conn: SQLite connection.
    :param uptype: Update type value to filter on.
    :returns: Latest timestamp string.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp FROM update_log WHERE update_type = ? ORDER BY timestamp DESC LIMIT 1", (uptype,))
    last_update = cursor.fetchone()
    cursor.close()
    return last_update[0]

def is_test_db(db_path: str) -> bool:
    """Check if an SQLite DB has metadata key ``is_test`` set to true.

    :param db_path: Path to database file.
    :returns: ``True`` if DB indicates test, else ``False``.
    """
    conn = create_connection(db_path)
    cursor = conn.cursor()
    try:
        cur = conn.execute("SELECT value FROM metadata WHERE key='is_test'")
        result = cur.fetchone()
        return result and result[0].lower() == 'true'
    except sqlite3.Error:
        return False

def export_snapshot(source_path: str, snapshot_dir: str, snapshots_to_keep: int) -> None:
    """Create a timestamped SQLite snapshot and prune old backups.

    :param source_path: Source DB path.
    :param snapshot_dir: Directory to store backups.
    :param snapshots_to_keep: Keep at most this many snapshots (None to skip pruning).
    :returns: None
    :raises FileNotFoundError: If source DB does not exist.
    """
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
    """Dump all tables to TSV files in an output directory.

    :param database_file: Path to database file.
    :param output_directory: Destination directory.
    :returns: None
    """
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
    """List table names in an SQLite database.

    :param database_file: Path to database file.
    :returns: List of table names.
    """
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    cursor.close()
    conn.close()
    return [table[0] for table in tables]

def add_column(db_conn, tablename, colname, coltype):
    """Add a column to a table.

    :param db_conn: SQLite connection.
    :param tablename: Target table name.
    :param colname: New column name.
    :param coltype: Column type string.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
    """
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
    """Modify multiple records according to update specs.

    :param db_conn: SQLite connection.
    :param table: Table name.
    :param updates: List of dicts with keys ``criteria_col``, ``criteria``, ``columns``, ``values``.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
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
    """Modify a single record convenience wrapper.

    :param db_conn: SQLite connection.
    :param table: Table name.
    :param criteria_col: WHERE column.
    :param criteria: WHERE value.
    :param columns: Columns to update.
    :param values: Values to set.
    :returns: Executed SQL template string.
    """
    update = {
        "criteria_col": criteria_col,
        "criteria": criteria,
        "columns": columns,
        "values": values
    }
    modify_multiple_records(db_conn, table, [update])
    return f'UPDATE {table} SET {" = ?, ".join(columns)} = ? WHERE {criteria_col} = ?'

def remove_column(db_conn, tablename, colname):
    """Remove a column from a table.

    :param db_conn: SQLite connection.
    :param tablename: Table name.
    :param colname: Column name to drop.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
    """
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
    """Rename a column.

    :param db_conn: SQLite connection.
    :param tablename: Table name.
    :param old_col: Existing column name.
    :param new_col: New column name.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
    """
    sql_str = f'ALTER TABLE {tablename} RENAME COLUMN {old_col} TO {new_col};'
    try:
        cursor: sqlite3.Cursor = db_conn.cursor()
        cursor.execute(sql_str)
    except sqlite3.Error as error:
        print(f'Failed to rename column in sqlite table. Error: {error}', sql_str)
        raise

def delete_multiple_records(db_conn, table, deletes):
    """Delete multiple records per delete spec.

    :param db_conn: SQLite connection.
    :param table: Table name.
    :param deletes: List of dicts with keys ``criteria_col`` and ``criteria``.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
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
    """Delete a single record convenience wrapper.

    :param db_conn: SQLite connection.
    :param tablename: Table name.
    :param criteria_col: WHERE column.
    :param criteria: WHERE value.
    :returns: None
    """
    delete = {
        "criteria_col": criteria_col,
        "criteria": criteria
    }
    delete_multiple_records(db_conn, tablename, [delete])

def add_record(db_conn, tablename, column_names, values):
    """Insert a single record into a table.

    :param db_conn: SQLite connection.
    :param tablename: Table name.
    :param column_names: List of column names.
    :param values: List of values.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
    """
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
    """Insert multiple records into a table.

    :param db_conn: SQLite connection.
    :param tablename: Table name.
    :param column_names: List of column names.
    :param list_of_values: Iterable of row value sequences.
    :returns: None
    :raises sqlite3.Error: On SQL failure.
    """
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
        
def create_connection(db_file, error_file: str|None = None, mode: str = 'ro'):
    """Create a database connection to an SQLite file.

    :param db_file: Database file path; returns None if file doesn't exist.
    :param error_file: Optional path to append exception messages.
    :param mode: ``'ro'`` for read-only (default) or any other for read-write.
    :returns: Connection object or None.
    """
    if not os.path.exists(db_file):
        return None
    conn = None
    try:
        if mode == 'ro':
            conn = sqlite3.connect(f'file:{db_file}?mode=ro', uri=True)
        else:
            conn = sqlite3.connect(db_file)
    except Exception as e:
        if error_file:
            with open(error_file,'a') as fil:
                fil.write(str(e)+'\n')
        else:
            print(e)
    return conn

def generate_database_table_templates_as_tsvs(db_conn, output_dir, primary_keys):
    """Generate TSV templates (headers only) for selected tables.

    :param db_conn: SQLite connection (not closed by this function).
    :param output_dir: Directory to write TSV files.
    :param primary_keys: Mapping table -> primary key column name to place first.
    :returns: None
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

def get_from_table(
        conn:sqlite3.Connection,
        table_name: str,
        criteria_col:str|None = None,
        criteria:str|tuple|None = None,
        select_col:str|list|None = None,
        as_pandas:bool = False,
        pandas_index_col:str|None = None,
        operator:str = '=') -> list[tuple] | pd.DataFrame:
    """Query a table with optional WHERE and return list or DataFrame.

    :param conn: SQLite connection.
    :param table_name: Table name.
    :param criteria_col: Optional WHERE column.
    :param criteria: WHERE value or tuple for two-parameter condition.
    :param select_col: Column(s) to select (default all).
    :param as_pandas: If ``True``, return DataFrame; else list.
    :param pandas_index_col: Index column for DataFrame.
    :param operator: SQL operator to use (default ``=``).
    :returns: DataFrame or list of values (first column) depending on ``as_pandas``.
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
        result = pd.read_sql_query(query, conn, params=params, index_col=pandas_index_col).convert_dtypes()  # type: ignore
    else:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()  # type: ignore
        result = [r[0] for r in result]
        cursor.close()
    return result

def get_from_table_by_list_criteria(
    conn:sqlite3.Connection,
    table_name: str, 
    criteria_col:str, 
    criteria:list,
    as_pandas:bool = True, 
    select_col: str = None,
    pandas_index_col:str|None = None
    ):
    """Query rows where a column matches any of the given values.

    :param conn: SQLite connection.
    :param table_name: Table name.
    :param criteria_col: Column name for the IN clause.
    :param criteria: List of values for the IN clause.
    :param as_pandas: If ``True``, return DataFrame; else list of tuples.
    :param select_col: Column(s) to select (default all).
    :param pandas_index_col: Optional index column for DataFrame.
    :returns: DataFrame or list depending on ``as_pandas``.
    """
    cursor: sqlite3.Cursor = conn.cursor()
    if select_col is None:
        select_col = '*'
    if isinstance(select_col, list):
        select_col = f'({", ".join(select_col)})'
    query: str = f'SELECT {select_col} FROM {table_name} WHERE {criteria_col} IN ({", ".join(["?" for _ in criteria])})'
    if as_pandas:
        ret: pd.DataFrame = pd.read_sql_query(query, con=conn, params=criteria, index_col = pandas_index_col).convert_dtypes()
    else:
        cursor.execute(query, criteria)
        ret: list = cursor.fetchall()
    cursor.close()
    return ret

def get_contaminants(db_file: str, protein_list:list = None, error_file: str = None) -> list:
    """Retrieve contaminant UniProt IDs from the ``contaminants`` table.

    :param db_file: Database file path.
    :param protein_list: If provided, intersect results with this list.
    :param error_file: Optional error log path for connection errors.
    :returns: List of contaminant UniProt IDs.
    """
    conn: sqlite3.Connection = create_connection(db_file, error_file)
    ret_list: list = get_from_table(conn, 'contaminants', select_col='uniprot_id')
    conn.close()
    return ret_list

def drop_table(conn:sqlite3.Connection, table_name: str) -> None:
    """Drop a table from the database if it exists.

    :param conn: SQLite connection.
    :param table_name: Table name to drop.
    :returns: None
    """
    sql_str: str = f'DROP TABLE IF EXISTS {table_name}'
    cursor: sqlite3.Cursor = conn.cursor()
    cursor.execute(sql_str)
    cursor.close()


def get_table_column_names(db_conn, table_name: str) -> list[str]:
    """Get column names for a table.

    :param db_conn: SQLite connection.
    :param table_name: Table name.
    :returns: List of column names.
    """

    # Connect to the SQLite database
    cursor = db_conn.cursor()

    # Get the list of tables in the database
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [row[1] for row in cursor.fetchall()]
    cursor.close()
    return columns
