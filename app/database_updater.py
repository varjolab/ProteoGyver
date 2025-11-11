"""Utilities to update and synchronize the SQLite database from TSV inputs
and external APIs (UniProt, IntAct, BioGRID).

This module provides helpers for:
- Creating TSV-based inserts/updates with schema reconciliation
- Merging interaction datasets and exporting incremental updates
- Recording update logs and packaging outputs
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import os
from typing import Iterator
import pandas as pd
import numpy as np
from components import text_handling
from components import db_functions
from components.api_tools import intact
from components.api_tools import biogrid

def update_table_with_file(cursor, table_name, file_path, parameters, timestamp, add_info: str = ''):
    """Update a table with data from a TSV file, adding columns if needed.

    :param cursor: SQLite database cursor.
    :param table_name: Target table name.
    :param file_path: Path to the TSV file with new data.
    :param parameters: Configuration parameters; expects keys like 'Allowed new columns', 'Allowed missing columns', 'Ignore diffs'.
    :param timestamp: Current update timestamp string.
    :param add_info: Optional progress info for logging.
    :returns: Tuple of (insertions, modifications).
    :raises ValueError: If too many new or missing columns are detected.
    """
    ignore_diffs = set(parameters['Ignore diffs'])
    modifications = 0
    insertions = 0
    cursor.execute(f"PRAGMA table_info({table_name})")
    table_info = cursor.fetchall()
    db_columns = set(row[1] for row in table_info)

    # Read the file
    df = pd.read_csv(file_path, sep='\t')
    if table_name in ['ms_runs', 'ms_plots']:
        vals = [0]
        try:
            query = 'SELECT internal_run_id FROM ms_runs'
            vals.extend([int(r[0].split('_',maxsplit=1)[-1]) for r in cursor.execute(query, ()).fetchall()])
        except Exception:
            pass
        run_index = max(vals) + 1
        base_id = 'PG_runID_'
        df['internal_run_id'] = [f'{base_id}{i}' for i in range(run_index, run_index + len(df))]

    file_columns = set(df.columns)

    # Compare columns
    missing_cols = db_columns - file_columns
    extra_cols = file_columns - db_columns
    missing_cols = [col for col in missing_cols if col not in ignore_diffs]
    extra_cols = [col for col in extra_cols if col not in ignore_diffs]

    if len(extra_cols) > parameters['Allowed new columns']:
        error_msg = f"Too many new columns in file: {', '.join(extra_cols)}"
        raise ValueError(f"Column mismatch for table {table_name}. {error_msg}")
    if len(missing_cols) > parameters['Allowed missing columns']:
        error_msg = f"Too many missing columns in file {file_path}: {', '.join(missing_cols)}"
        raise ValueError(f"Column mismatch for table {table_name}. {error_msg}")
    
    # Add extra columns to database if they exist in file
    col_renames = {}
    for col in extra_cols:
        # Check if the column in the dataframe is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if is_numeric:
            col_type = 'NUMERIC'
        else:
            col_type = 'TEXT'
            
        # Sanitize column name to only contain lowercase, numbers, and underscore
        sanitized_col = text_handling.sanitize_for_database_use(col)
        
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {sanitized_col} {col_type}")
        print(f"Added new column '{sanitized_col}' to table {table_name} with type {col_type}")
        
        # Add column name to rename mapping if it was changed
        if sanitized_col != col:
            col_renames[col] = sanitized_col
    df.rename(columns=col_renames, inplace=True)
    df.replace('', np.nan, inplace=True)

    primary_key = df.columns[0]  # Assuming the first column is the primary key
    
    for _, row in df.iterrows():
        pk_value = row[primary_key]
        
        # Check if the primary key already exists in the table
        cursor.execute(f"SELECT * FROM {table_name} WHERE {primary_key} = ?", (pk_value,))
        existing_entry = cursor.fetchone()

        if existing_entry:
            cursor.execute(f"PRAGMA table_info({table_name})")
            all_columns = [rr[1] for rr in cursor.fetchall()]
            
            # Create a dictionary of the new values
            new_values = row.to_dict()
            
            # If there's an existing entry, preserve values for missing columns
            for i, col in enumerate(all_columns):
                if (col not in new_values) or (pd.isna(new_values[col])):
                    new_values[col] = existing_entry[i]

            differences = 0
            for i, col in enumerate(all_columns):
                if col in ignore_diffs:
                    continue
                if new_values[col] != existing_entry[i]:
                    differences += 1

            # If new entry is the same as old one, do nothing
            if differences == 0:
                continue
            modifications += 1

            # Prepare the INSERT statement with preserved values from earlier
            columns = ', '.join(all_columns)
            placeholders = ', '.join(['?'] * len(all_columns))
            values = tuple(new_values[col] for col in all_columns)
        else:
            # For new entries, just use the values from the file
            columns = ', '.join(row.index)
            placeholders = ', '.join(['?'] * len(row))
            values = tuple(row)

        # Insert the entry
        cursor.execute(f"INSERT OR REPLACE INTO {table_name} ({columns}) VALUES ({placeholders})", values)
        insertions += 1

    insertions -= modifications
    print(f"Updated table {table_name}: {modifications} modifications, {insertions} insertions. {add_info} data file: {file_path}.")

    return insertions, modifications

def update_database(conn, parameters, cc_cols, cc_types, timestamp):
    """Update multiple database tables using TSV files from configured directories.

    :param conn: SQLite database connection.
    :param parameters: Parameters with 'Update files' table→directory mappings and limits.
    :param cc_cols: Expected column names for creating fresh tables.
    :param cc_types: SQL column types aligned with ``cc_cols``.
    :param timestamp: Current update timestamp string.
    :returns: Tuple of (inmod_names, inmod_vals) listing counts per table and action.
    """
    update_files = parameters['Update files']
    inmod_names = []
    inmod_vals = []
    cursor = conn.cursor()
    update_order = ['remove_data','add_or_replace']
    update_order.extend([table_name for table_name in update_files if table_name not in update_order])
    delete_files = []
    for table_name in update_order:
        insertions = 0
        modifications = 0
        deletions = 0
        folder_path = os.path.join(*update_files[table_name])
        if os.path.exists(folder_path):
            if table_name == 'remove_data':
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.tsv'):
                        dbtable_name = file_name.rsplit('.', 1)[0]
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:  # Skip empty lines and comments
                                    continue
                                if line.startswith('#'):
                                    continue
                                    
                                criteria_list = line.split('\t')
                                where_conditions = []
                                values = []
                                
                                # Process each criterion (column_name, value pair)
                                for criterion in criteria_list:
                                    column, value = [x.strip() for x in criterion.split(',', 1)]
                                    where_conditions.append(f"{column} = ?")
                                    values.append(value)
                                
                                where_clause = " AND ".join(where_conditions)
                                
                                cursor.execute(f"DELETE FROM {dbtable_name} WHERE {where_clause}", tuple(values))
                                deletions += cursor.rowcount
                            delete_files.append(file_path)
                        conn.commit()
            elif table_name == 'add_or_replace':
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.tsv'):
                        file_path = os.path.join(folder_path, file_name)
                        new_table_name = file_name.rsplit('.', 1)[0]
                        df = pd.read_csv(file_path, sep='\t')
                        space_columns = [col for col in df.columns if ' ' in col]
                        assert len(space_columns) == 0, f"Column names must not contain spaces. Problematic column names: {space_columns}"
                        cursor.execute(f"DROP TABLE IF EXISTS {new_table_name}")
                        # Create table with columns from the file
                        if set(cc_cols) != set(df.columns):
                            with open(os.path.join(folder_path, file_name.replace('.tsv','.txt')), 'r') as f:
                                cc_types = [l.strip() for l in f.readlines()]
                                cc_types = [l for l in cc_types if not l.startswith('#')]
                                cc_types = [l for l in cc_types if l != '']
                        columns = []    
                        for i, col in enumerate(df.columns):
                            col_type = cc_types[i]
                            columns.append(f"{col} {col_type}")
                        create_query = f"CREATE TABLE {new_table_name} ({', '.join(columns)})"
                        cursor.execute(create_query)
                        # Insert data
                        columns = ', '.join(df.columns)
                        placeholders = ', '.join(['?'] * len(df.columns))
                        for _, row in df.iterrows():
                            cursor.execute(f"INSERT INTO {new_table_name} ({columns}) VALUES ({placeholders})", tuple(row))
                        insertions += len(df)
                        print(f"Created new table {new_table_name} with {insertions} rows")
                        delete_files.append(file_path)
                        if os.path.isfile(file_path.replace('.tsv','txt')):
                            delete_files.append(file_path.replace('.tsv','txt'))
                        conn.commit()
            else:
                dirfiles = os.listdir(folder_path)
                for i, file_name in enumerate(dirfiles):
                    if file_name.endswith('.tsv'):
                        file_path = os.path.join(folder_path, file_name)

                        add_info = ''
                        if len(dirfiles) > 100:
                            add_info = f'{i}/{len(dirfiles)}'
                        insertions, modifications = update_table_with_file(cursor, table_name, file_path, parameters, timestamp, add_info)
                        delete_files.append(file_path)
                        conn.commit()
        else:
            os.makedirs(folder_path)
            print(f"Created directory {folder_path}")
        inmod_names.append(f'{table_name} insertions')
        inmod_vals.append(insertions)
        inmod_names.append(f'{table_name} modifications')
        inmod_vals.append(modifications)
        inmod_names.append(f'{table_name} deletions')
        inmod_vals.append(deletions)
    try:
        conn.commit()
        for file_path in delete_files:
            os.remove(file_path)
    except Exception as e:
        print(f"Failed to commit changes: {e}")
    finally:
        cursor.close()
    return inmod_names, inmod_vals

def get_dataframe_differences(df1: pd.DataFrame, df2: pd.DataFrame, ignore_columns: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Compare two DataFrames and return modified/new indices and missing indices.

    Columns listed in ``ignore_columns`` are dropped before comparison. The two
    DataFrames must have identical columns after dropping.

    :param df1: Baseline DataFrame.
    :param df2: New DataFrame to compare against baseline.
    :param ignore_columns: Columns to ignore during comparison.
    :returns: Tuple of (new_or_modified_indices, missing_indices).
    """
    if ignore_columns is None:
        ignore_columns = []

    df1 = df1.drop(columns=[col for col in ignore_columns if col in df1.columns])
    df2 = df2.drop(columns=[col for col in ignore_columns if col in df2.columns])
    # Ensure same columns
    assert list(df1.columns) == list(df2.columns), "DataFrames must have the same columns"

    common_idx = df1.index.intersection(df2.index)
    missing_rows = list(df1.index.difference(df2.index))
    new_or_modified = set(df2.index.difference(df1.index))

    for idx in common_idx:
        row1 = df1.loc[idx]
        row2 = df2.loc[idx]
        for col in df1.columns:
            val1 = row1[col]
            val2 = row2[col]

            # Normalize values
            val1 = '' if pd.isna(val1) or val1 == 0 else val1
            val2 = '' if pd.isna(val2) or val2 == 0 else val2

            if val1 != val2:
                new_or_modified.add(idx)
                break  # no need to compare rest of row

    return list(new_or_modified), missing_rows

def update_uniprot(conn, parameters, timestamp, versions: list, organisms: set|None = None):
    """Download and stage UniProt data, writing TSV updates if differences found.

    :param conn: SQLite database connection.
    :param parameters: Updater parameters (paths, ignore diffs, deletion policy).
    :param timestamp: Current update timestamp string.
    :param versions: List of current versions of the UniProt database.
    :param organisms: Optional set of organism IDs to include.
    :returns: Set of UniProt IDs present in the fetched dataset.
    """
    from components.api_tools import uniprot
# Move these into parameters or somesuch:
    uniprot_renames = {
        'Reviewed': 'is_reviewed',
        'Gene names (primary)': 'gene_name',
        'Entry name': 'entry_name',
        'Gene names': 'all_gene_names',
        'Organism': 'organism',
        'Length': 'length',
        'Sequence': 'sequence',
    }
    uniprot_df, new_versions = uniprot.download_uniprot_for_database(versions, organisms=organisms)
    #uniprot_df.to_csv('uniprot_df_full.tsv',sep='\t')
    #uniprot_df = pd.read_csv('uniprot_df_full.tsv', sep='\t', index_col=0)
    uniprot_df.index.name = 'uniprot_id'
    uniprot_id_set = set(uniprot_df.index)
    uniprot_df.rename(columns=uniprot_renames, inplace=True)
    uniprot_df['is_reviewed'] = (uniprot_df['is_reviewed']=='reviewed').astype(int)
    gns = []
    for _,row in uniprot_df.iterrows():
        if pd.isna(row['gene_name']):
            gns.append(row['entry_name'])
        else:
            gns.append(row['gene_name'])
    uniprot_df['gene_name'] = gns
    gns = []
    for _,row in uniprot_df.iterrows():
        if pd.isna(row['all_gene_names']):
            gns.append(row['entry_name'])
        else:
            gns.append(row['all_gene_names'])
    uniprot_df['all_gene_names'] = gns
    existing: pd.DataFrame = db_functions.get_from_table_by_list_criteria(conn, 'proteins', criteria_col='uniprot_id', criteria=list(uniprot_df.index), as_pandas = True) # type: ignore
    existing.set_index('uniprot_id', inplace=True, drop=True)
    updates, missing_rows = get_dataframe_differences(existing, uniprot_df, ignore_columns = parameters['Ignore diffs']) #type: ignore
    if parameters['Delete old uniprots'] and len(missing_rows) > 0:
        os.makedirs(os.path.join(*parameters['Update files']['remove_data']), exist_ok=True)
        del_filepath = os.path.join(*parameters['Update files']['remove_data'], 'proteins.tsv')
        # Read existing deletion criteria if file exists
        deletions = []
        if os.path.exists(del_filepath):
            with open(del_filepath, 'r') as f:
                deletions = [l.strip() for l in f.readlines()]
        
        # Add new deletion criteria for missing UniProt IDs
        # Each line will contain just one criterion: "uniprot_id, UPIDXXX"
        deletions.extend([f'uniprot_id, {upid}' for upid in missing_rows])
        
        # Write all deletion criteria to file
        with open(del_filepath, 'w') as f:
            f.write('\n'.join(deletions))
    uniprot_df = uniprot_df.loc[updates].fillna('')
    uniprot_df['is_latest'] = 1
    uniprot_df['entry_source'] = f'update_{timestamp}'
    uniprot_df['version_update_time'] = timestamp
    pver = []
    for index in updates:
        if index in existing.index:
            pver.append(existing.loc[index]['version_update_time'])
        else:
            pver.append(-1)
    uniprot_df['prev_version'] = pver
    if len(updates) > 0:
        odir = os.path.join(*parameters['Update files']['proteins'])
        if not os.path.exists(odir):
            os.makedirs(odir)
        modfile = os.path.join(odir, f'{timestamp}_proteins.tsv')
        uniprot_df.to_csv(modfile.replace(':','-'), sep='\t')
    return uniprot_id_set, new_versions

def merge_multiple_string_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge DataFrames with semicolon-separated string fields by union of values.

    Each cell is split on ';' and de-duplicated across dataframes; merged rows are
    indexed by 'interaction'.

    :param dfs: List of input DataFrames.
    :returns: Merged DataFrame with unioned semicolon-joined values.
    """
    if not dfs:
        return pd.DataFrame()

    from collections import defaultdict

    merged_data = defaultdict(lambda: defaultdict(set))
    all_columns = set()

    for df in dfs:
        for idx, row in df.iterrows():
            for col, val in row.items():
                all_columns.add(col)
                if pd.notna(val):
                    merged_data[idx][col].update(str(val).split(';'))

    # Now flatten to list of dicts
    rows = []
    for idx, coldata in merged_data.items():
        row = {'interaction': idx}
        for col in all_columns:
            if col in coldata:
                dedup = sorted(coldata[col])
                row[col] = ';'.join(dedup) if dedup else None
            else:
                row[col] = None
        rows.append(row)
    if len(rows) > 0:
        result = pd.DataFrame(rows)
        result.set_index('interaction', inplace=True)
    else:
        result = pd.DataFrame(columns = [c for c in dfs[0].columns if c != 'interaction'])
    return result


def stream_flattened_rows(df: pd.DataFrame) -> Iterator[dict]:
    """Yield rows as dictionaries with set-unioned values split on ';'.

    :param df: Input DataFrame; uses 'interaction' as index when present.
    :returns: Iterator of dict rows with 'interaction' key and set values per column.
    """
    if 'interaction' in df.columns:
        df.set_index('interaction', inplace=True)

    for interaction, row in df.iterrows():
        out_row: dict = {'interaction': interaction}
        for col in df.columns:
            if pd.isna(row[col]):
                continue
            values = [v.strip(';') for v in str(row[col]).split(';') if v]
            key = col
            if key in out_row:
                out_row[key].update(values)
            else:
                out_row[key] = set(values)
        yield out_row

def handle_new(new_interactions, odir, timestamp, L) -> None:
    """Write new interactions to a timestamped TSV file in the output directory.

    :param new_interactions: List of interaction dicts keyed by 'interaction'.
    :param odir: Output directory.
    :param timestamp: Current update timestamp string.
    :param L: Chunk prefix/letter for file naming.
    :returns: None.
    """
    i = 1
    modfile = os.path.join(odir, f'{timestamp}_known_interactions_{L}.tsv')
    while os.path.exists(modfile):
        modfile = os.path.join(odir, f'{timestamp}_known_interactions_{L}_{i}.tsv')
        i += 1
    df = pd.DataFrame(new_interactions).set_index('interaction')
    df.to_csv(modfile.replace(':','-'), sep='\t')

def handle_mods(check_for_mods, existing, timestamp, L, parameters, odir) -> None:
    """Write modified interactions to TSV and optionally queue deletions.

    :param check_for_mods: List of candidate modified interaction dicts.
    :param existing: Existing interactions DataFrame (indexed by 'interaction').
    :param timestamp: Current update timestamp string.
    :param L: Chunk prefix/letter for file naming.
    :param parameters: Updater parameters including deletion settings.
    :param odir: Output directory.
    :returns: None.
    """

    merg = pd.DataFrame(check_for_mods).set_index('interaction')
    updates, missing_rows = get_dataframe_differences(existing, merg, ignore_columns = parameters['Ignore diffs'])
    if parameters['Delete old interactions'] and len(missing_rows) > 0:
        deletions = []
        os.makedirs(os.path.join(*parameters['Update files']['remove_data']), exist_ok=True)
        del_filepath = os.path.join(*parameters['Update files']['remove_data'], 'known_interactions.tsv')
        deletions.extend([f'interaction, {interaction}' for interaction in missing_rows])
        with open(del_filepath, 'a') as f:
            f.write('\n'.join(deletions))
    if len(updates) > 0:
        merg = merg.loc[updates].fillna('')
        merg['version_update_time'] = timestamp
        pver = []
        for index in updates:
            if index in existing.index:
                pver.append(existing.loc[index]['version_update_time'])
            else:
                pver.append(-1)
        merg['prev_version'] = pver
        i = 1
        if not os.path.exists(odir):
            os.makedirs(odir)
        modfile = os.path.join(odir, f'{timestamp}_known_interactions_{L}.tsv')
        while os.path.exists(modfile):
            modfile = os.path.join(odir, f'{timestamp}_known_interactions_{L}_{i}.tsv')
            i += 1
        merg.to_csv(modfile.replace(':','-'), sep='\t')

def handle_merg_chunk(existing:pd.DataFrame, organisms: set|None, timestamp: str, L: str, last_update_date: datetime|None, odir: str, parameters: dict) -> None:
    """Merge IntAct and BioGRID chunks, writing new and modified interactions.

    :param existing: Existing interactions DataFrame (index 'interaction').
    :param organisms: Optional set of organism IDs to include.
    :param timestamp: Current update timestamp string.
    :param L: Chunk prefix letter.
    :param last_update_date: Optional cutoff date for remote queries.
    :param odir: Output directory for TSVs.
    :param parameters: Updater parameters including 'Ignore diffs' and paths.
    :returns: None.
    """
    existing_interactions: set = set(existing.index)
    def stream_cleaned_rows():
        sources = [
            intact.get_latest(organisms, subset_letter=L, since_date = last_update_date),   # type: ignore
            biogrid.get_latest(organisms, subset_letter=L, since_date = last_update_date),  # type: ignore
        ]
        print('Merging', L, 'source df shapes', sources[0].shape, sources[1].shape) # type: ignore
        merged = {}
        for df in sources:
            for row in stream_flattened_rows(df):
                interaction = row.pop('interaction')
                merged.setdefault(interaction, {})
                for k, v in row.items():
                    merged[interaction].setdefault(k, set()).update(v)

        for interaction, fields in merged.items():
            row_data = {
                'interaction': interaction,
                **{k: ';'.join(sorted(v)).strip(';') for k, v in fields.items()}
            }

            for badval in ['nan', '', '-', 'None']:
                for k, v in row_data.items():
                    if v == badval:
                        row_data[k] = None

            pubid = row_data.get('publication_identifier') or ''
            row_data['publication_count'] = len(pubid.strip(';').split(';')) if pubid else 0

            row_data['version_update_time'] = timestamp
            row_data['prev_version'] = -1

            required = [
                'uniprot_id_a', 'uniprot_id_b',
                'uniprot_id_a_noiso', 'uniprot_id_b_noiso',
                'isoform_a', 'isoform_b'
            ]
            if any(row_data.get(k) in [None, '', np.nan] for k in required):
                continue

            yield row_data

    check_for_mods = []
    new_interactions = []
    if not os.path.exists(odir):
        os.makedirs(odir)
    for row in stream_cleaned_rows():
        if row['interaction'] in existing_interactions:
            check_for_mods.append(row)
        else:
            new_interactions.append(row)
        if len(new_interactions) > 1000:
            handle_new(new_interactions, odir, timestamp, L)
            new_interactions = []
        if len(check_for_mods) > 1000:
            handle_mods(check_for_mods, existing, timestamp, L, parameters, odir)
            check_for_mods = []
        
    if len(new_interactions) > 0:
        handle_new(new_interactions, odir, timestamp, L)
    if len(check_for_mods) > 0:
        handle_mods(check_for_mods, existing, timestamp, L, parameters, odir)

def update_version_table(conn, dataset, timestamp, new_versions) -> None:
    """Update the version table with the new versions.
    """
    cursor = conn.cursor()
    # Ensure the table exists
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS data_versions (
            dataset TEXT,
            version TEXT,
            last_update_check TEXT
        )
        """
    )

    if len(new_versions) > 0:
        new_ver_df = pd.DataFrame(
            data = [[dataset, v, timestamp] for v in new_versions],
            columns = ['dataset', 'version', 'last_update_check']
        )
        if '::' in new_versions[0]:
            # Fetch existing entries for this dataset into a DataFrame
            existing_df = pd.read_sql(
                "SELECT dataset, version FROM data_versions WHERE dataset = ?",
                con=conn,
                params=(dataset,),
            )
            new_ver_df = new_ver_df[~new_ver_df['version'].str.startswith('-1::')]

            rows_to_drop = []
            incoming_organisms = {v.split('::')[0] for v in new_versions}
            for index, row in existing_df.iterrows():
                if row['dataset'].split('::')[0] in incoming_organisms:
                    rows_to_drop.append(index)
            if len(rows_to_drop) > 0:
                existing_df.drop(index=rows_to_drop, inplace=True)
            df_new = pd.concat(
                [existing_df, new_ver_df]
            )
        else:
            df_new = new_ver_df
        # Write the new rows
        if len(df_new) > 0:
            cursor.execute("DELETE FROM data_versions WHERE dataset = ?", (dataset,))
            df_new.to_sql('data_versions', conn, if_exists='append', index=False)

    cursor.close()
    conn.commit()

def update_knowns(conn, parameters, timestamp, uniprots, organisms, versions: dict[str, list[str]|str], last_update_date: datetime|None = None, ncpu: int = 1) -> list[tuple[str, str]]:
    """Update known interaction TSVs in parallel by merging external sources.

    :param conn: SQLite connection for reading existing interactions.
    :param parameters: Updater parameters with file paths.
    :param timestamp: Current update timestamp string.
    :param uniprots: Set of UniProt IDs to filter by.
    :param organisms: Optional set of organism IDs to include.
    :param versions: Dictionary of versions for each external source.
    :param last_update_date: Cutoff datetime; ignore older remote entries.
    :param ncpu: Number of worker processes.
    :returns: List of new versions for each external source.
    """
    biogrid_version = biogrid.update(version = versions['biogrid'], uniprots_to_get = uniprots, organisms=organisms)
    intact_version = intact.update(version = versions['intact'], uniprots_to_get = uniprots, organisms=organisms)
    get_chunks = sorted(list(set(intact.get_available()) | set(biogrid.get_available())))
    num_cpus = ncpu
    odir = os.path.join(*parameters['Update files']['known_interactions'])
    if not os.path.exists(odir):
        os.makedirs(odir)
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = []
        for L in get_chunks:    
            existing: pd.DataFrame = db_functions.get_from_table(
                conn,
                'known_interactions',
                criteria_col = 'interaction',
                criteria = f"'{L}%'",
                operator = 'LIKE',
                pandas_index_col = 'interaction',
                as_pandas = True
            )  # type: ignore
            futures.append(executor.submit(handle_merg_chunk, existing, organisms, timestamp, L, last_update_date, odir, parameters))
        for future in as_completed(futures):
            future.result()
    return [('biogrid', biogrid_version), ('intact', intact_version)]

def update_external_data(conn, parameters, timestamp, organisms: set|None = None, last_update_date: datetime|None = None, versions: dict|None = None, ncpu: int = 1):
    """Update external data tables (UniProt and known interactions).

    :param conn: SQLite database connection.
    :param parameters: Updater parameters; includes external update intervals.
    :param timestamp: Current update timestamp string.
    :param organisms: Optional set of organism IDs to update.
    :param last_update_date: Cutoff date; ignore data older than this.
    :param ncpu: Number of CPUs to use for merging chunks.
    :returns: None.
    """ 
    print('Updating uniprot')
    if versions is None:
        versions = { }
    if not 'uniprot' in versions:
        versions['uniprot'] = ['-1::no version']
    for d in ['biogrid', 'intact']:
        if not d in versions:
            versions[d] = 'no version'
    uniprots, new_versions = update_uniprot(conn, parameters, timestamp, versions['uniprot'], organisms)
    print('Updating known interactions')
    known_versions = update_knowns(conn, parameters, timestamp, uniprots, organisms, versions, last_update_date, ncpu)
    update_version_table(conn, 'uniprot', timestamp, new_versions)
    for dataset, version in known_versions:
        update_version_table(conn, dataset, timestamp, version)

def update_log_table(conn, inmod_names, inmod_vals, timestamp, uptype: str) -> None:
    """Record database update info in a log table.

    :param conn: SQLite database connection.
    :param inmod_names: Names like 'table action' (e.g., 'proteins insertions').
    :param inmod_vals: Counts aligned with ``inmod_names``.
    :param timestamp: Update timestamp string.
    :param uptype: Update category label (e.g., 'external', 'snapshot').
    :returns: None.
    """
    hasmods = sum(inmod_vals) > 0
    table_columns = ['update_id TEXT', 'timestamp TEXT', 'update_type TEXT', 'modification_type TEXT', 'tablename TEXT', 'count INTEGER']
    new_rows = []
    
    # Get the next update ID by finding the highest existing one
    cursor = conn.cursor()
    try:
        # Check if update_log table exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='update_log'")
        if cursor.fetchone():
            # Get the highest update_id that matches our pattern
            cursor.execute("SELECT update_id FROM update_log WHERE update_id LIKE 'upd_%' ORDER BY CAST(SUBSTR(update_id, 5) AS INTEGER) DESC LIMIT 1")
            result = cursor.fetchone()
            if result:
                # Extract the number from the last update_id and increment
                last_id = result[0]
                last_num = int(last_id.split('_')[1])
                next_num = last_num + 1
            else:
                # No existing upd_ IDs found, start from 1
                next_num = 1
        else:
            # Table doesn't exist yet, start from 1
            next_num = 1
    except Exception:
        # If any error occurs, start from 1
        next_num = 1
    
    update_id = f'upd_{next_num}'
    if hasmods:
        modifications = {}
        for i, name in enumerate(inmod_names):
            table, mtype = name.rsplit(' ',maxsplit=1)
            modifications.setdefault(mtype, {}).setdefault(table, inmod_vals[i])
        for mtype, tables in modifications.items():
            for table, count in tables.items():
                new_rows.append((update_id,timestamp, uptype, mtype, table, count))
    else:
        new_rows.append((update_id,timestamp, uptype, 'no modifications','',0))
    sep = ',\n    '
    create_strs = f"CREATE TABLE IF NOT EXISTS update_log (\n    {sep.join(table_columns)}\n)"
    # Create update_log table if it doesn't exist
    cursor = conn.cursor()
    cursor.execute(create_strs)
    placeholders = ','.join(['?'] * len(table_columns))
    for nc in new_rows:
        values = tuple(nc)
        insert_str = f"INSERT INTO update_log ({','.join([tc.split()[0] for tc in table_columns])}) VALUES ({placeholders})"
        cursor.execute(insert_str, values)
    cursor.close()
    conn.commit()
