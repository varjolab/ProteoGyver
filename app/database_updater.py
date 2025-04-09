import os
import pandas as pd
import numpy as np
from components import text_handling
from components import db_functions


def update_table_with_file(cursor, table_name, file_path, parameters, timestamp):
    """
    Updates a database table with data from a TSV file, handling column additions and value updates.

    Args:
        cursor: SQLite database cursor object
        table_name (str): Name of the table to update
        file_path (str): Path to the TSV file containing new data
        parameters (dict): Configuration parameters including 'Allowed new columns' and 'Allowed missing columns'
    Returns:
        tuple: (insertions, modifications) count of new entries and modified entries

    Raises:
        ValueError: If there are too many new or missing columns compared to parameters
    """
    ignore_diffs = set(parameters['Ignore diffs'])
    modifications = 0
    insertions = 0
    cursor.execute(f"PRAGMA table_info({table_name})")
    table_info = cursor.fetchall()
    db_columns = set(row[1] for row in table_info)

    # Read the file
    df = pd.read_csv(file_path, sep='\t')
    file_columns = set(df.columns)

    # Compare columns
    missing_cols = db_columns - file_columns
    extra_cols = file_columns - db_columns

    if len(extra_cols) > parameters['Allowed new columns']:
        error_msg = f"Too many new columns in file: {', '.join(extra_cols)}"
        raise ValueError(f"Column mismatch for table {table_name}. {error_msg}")
    if len(missing_cols) > parameters['Allowed missing columns']:
        error_msg = f"Too many missing columns in file: {', '.join(missing_cols)}"
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
            # Get column names for the existing entry
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
                        
            new_pk_value = f"{pk_value}_{timestamp}"
            cursor.execute(f"UPDATE {table_name} SET {primary_key} = ? WHERE {primary_key} = ?", (new_pk_value, pk_value))
            modifications += 1

            # Prepare the INSERT statement with preserved values
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
    print(f"Updated table {table_name} with data from {file_path}. {modifications} modifications, {insertions} insertions.")

    return insertions, modifications

def update_database(conn, parameters, cc_cols, cc_types, timestamp):
    """
    Updates multiple database tables using TSV files from specified directories.

    Args:
        conn: SQLite database connection object
        parameters (dict): Configuration parameters including 'Update files' with table-to-directory mappings

    Returns:
        tuple: (inmod_names, inmod_vals)
            - inmod_names (list): Names of tables with their modification types
            - inmod_vals (list): Corresponding counts of insertions and modifications
    """
    update_files = parameters['Update files']
    inmod_names = []
    inmod_vals = []
    cursor = conn.cursor()
    update_order = ['remove_data','add_or_replace']
    update_order.extend([table_name for table_name in update_files if table_name not in update_order])
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
                                if not line:  # Skip empty lines
                                    continue
                                if line.startswith('#'):
                                    continue
                                    
                                # Split criteria by tabs
                                criteria_list = line.split('\t')
                                where_conditions = []
                                values = []
                                
                                # Process each criterion (column_name, value pair)
                                for criterion in criteria_list:
                                    column, value = [x.strip() for x in criterion.split(',', 1)]
                                    where_conditions.append(f"{column} = ?")
                                    values.append(value)
                                
                                # Construct and execute DELETE query
                                where_clause = " AND ".join(where_conditions)
                                cursor.execute(f"DELETE FROM {dbtable_name} WHERE {where_clause}", tuple(values))
                                deletions += cursor.rowcount
                            os.remove(file_path)
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
                            with open(os.path.join(folder_path, file_name.replace('.tsv','txt')), 'r') as f:
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
                        os.remove(file_path)
                        if os.path.isfile(os.path.join(folder_path, file_name.replace('.tsv','txt'))):
                            os.remove(os.path.join(folder_path, file_name.replace('.tsv','txt')))
            else:
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.tsv'):
                        file_path = os.path.join(folder_path, file_name)
                        insertions, modifications = update_table_with_file(cursor, table_name, file_path, parameters, timestamp)
                        os.remove(file_path)
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
    except Exception as e:
        print(f"Failed to commit changes: {e}")
    finally:
        cursor.close()
    return inmod_names, inmod_vals

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

def get_dataframe_differences(df1: pd.DataFrame, df2: pd.DataFrame, ignore_columns: list[str]|None = None) -> tuple[list[str], list[str]]:
    """Compare two pandas DataFrames and return differences.
    
    Returns:
        tuple: (differences, new_columns, missing_columns, new_rows, missing_rows)
            - differences: List of (index, column, new_value) tuples for each difference
            - new_columns: List of column names present in df2 but not df1 
            - missing_columns: List of column names present in df1 but not df2
    """
    if not ignore_columns:
        ignore_columns = []
    df1 = df1.drop(columns=[i for i in ignore_columns if i in df1.columns])
    df2 = df2.drop(columns=[i for i in ignore_columns if i in df2.columns])
    assert list(df1.columns) == list(df2.columns), "DataFrames must have the same columns"
    # Get common indices
    common_indices = df1.index.intersection(df2.index)
    
    missing_rows = [idx for idx in df1.index if idx not in df2.index]
    modified_or_new_rows = [idx for idx in df2.index if idx not in df1.index]
    # Compare values
    for idx in common_indices:
        for col in df1.columns:
            if ignore_columns and col in ignore_columns:
                continue
            val1 = df1.loc[idx, col]
            val2 = df2.loc[idx, col]

            # Especially in database NaN is often just empty string, so missing values should be handled in this way. Same with zeroes.
            if pd.isna(val1):
                val1 = ''
            if pd.isna(val2):
                val2 = ''
            if val1 == 0:
                val1 = ''
            if val2 == 0:
                val2 = ''
            if val1 != val2:
                modified_or_new_rows.append(idx)
    
    return (modified_or_new_rows, missing_rows)
def update_uniprot(conn, parameters, timestamp, organisms: set|None = None):
    from components.api_tools.annotation import uniprot
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
    #uniprot_df = uniprot.download_uniprot_for_database(organisms=organisms)
    uniprot_df = pd.read_csv('uniprot_df.tsv', sep='\t', index_col=0)
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
        uniprot_df.to_csv(modfile, sep='\t')
    return uniprot_id_set

def merge_multiple_string_dataframes(dfs) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()

    # Get union of all indices and columns
    all_indices = dfs[0].index
    all_columns = dfs[0].columns

    for df in dfs[1:]:
        all_indices = all_indices.union(df.index)
        all_columns = all_columns.union(df.columns)

    # Create empty dataframe with full shape
    merged = pd.DataFrame(index=all_indices, columns=all_columns)

    for idx in all_indices:
        for col in all_columns:
            cell_values = []
            for df in dfs:
                if idx in df.index and col in df.columns:
                    val = df.at[idx, col]
                    if pd.notna(val):
                        cell_values.extend(str(val).split(';'))  # handles pre-merged cells
            dedup = sorted(set(cell_values))
            merged.at[idx, col] = ';'.join(dedup) if dedup else None

    return merged
    
def update_knowns(conn, parameters, timestamp, uniprots, organisms) -> None:
    from components.api_tools.annotation import intact
    from components.api_tools.annotation import biogrid

    biogrid.update(uniprots_to_get = uniprots, organisms=organisms)
    intact.update(uniprots_to_get = uniprots, organisms=organisms)
    dbtables = [intact.get_latest(), biogrid.get_latest()]
    merg = merge_multiple_string_dataframes(dbtables)

    existing: pd.DataFrame = db_functions.get_full_table_as_pd(conn, 'known_interactions')
    existing.set_index('interaction', inplace=True, drop=True)
    existing.drop(existing.index.difference(merg.index), inplace=True)
    merg = merg[merg['uniprot_id_a'].isin(uniprots) & merg['uniprot_id_b'].isin(uniprots)]
    merg['publication_count'] = [len(x.split(';')) for x in merg['publication_identifier'].values]
    merg.to_csv('merg.tsv', sep='\t')
    existing.to_csv('existing.tsv', sep='\t')   

    col_order = [c for c in existing.columns if c in merg.columns]
    col_order.extend([c for c in merg.columns if c not in col_order])
    merg = merg[col_order]
    
    updates, missing_rows = get_dataframe_differences(existing, merg, ignore_columns = parameters['Ignore diffs'])
    if parameters['Delete old interactions'] and len(missing_rows) > 0:
        del_filepath = os.path.join(*parameters['Update files']['remove_data'], 'known_interactions.tsv')
        if os.path.exists(del_filepath):
            with open(del_filepath, 'r') as f:
                deletions = [l.strip() for l in f.readlines()]
        deletions.extend([f'interaction, {interaction}' for interaction in missing_rows])
        with open(del_filepath, 'w') as f:
            f.write('\n'.join(deletions))
    merg = merg.loc[updates].fillna('')
    merg['version_update_time'] = timestamp
    pver = []
    for index in updates:
        if index in existing.index:
            pver.append(existing.loc[index]['version_update_time'])
        else:
            pver.append(-1)
    merg['prev_version'] = pver

    modfile = os.path.join(*parameters['Update files']['known_interactions'], f'{timestamp}_known_interactions.tsv')
    if len(updates) > 0:
        merg.to_csv(modfile, sep='\t')

def update_ms_runs(conn, parameters, timestamp, time_format, output_dir) -> None:
    import json
    import re
    from datetime import datetime
    """
    Updates the ms_runs table in the database.

    Args:
        conn: SQLite database connection object
        parameters (dict): Configuration parameters including 'Update files' with table-to-directory mappings
        timestamp (str): Current timestamp
        time_format (str): Format string for datetime objects

    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ms_runs)")
    ms_cols = [row[1] for row in cursor.fetchall()]
    cursor.close()

    banned_run_dirs = parameters['Ignore runs from dirs']
    run_id_regex = parameters['MS run ID regex']
    if parameters['Additional info excel'] != '':
        runlist = pd.read_excel(os.path.join(*parameters['Additional info excel']))
    else:
        runlist = pd.DataFrame(columns=['Sample name','Who','Sample type','Bait name','Bait / other uniprot or ID','Bait mutation','Cell line / material','Project','Notes','tag'])
    
    ms_run_datadir = os.path.join(*parameters['MS run data dir for updater'])
    new_data = []
    for datafilename in os.listdir(ms_run_datadir):
        with open(os.path.join(ms_run_datadir, datafilename)) as fil:
            try:
                dat = json.load(fil)
            except json.JSONDecodeError:
                continue
        if dat['SampleInfo'] == ['']: continue
        banned = False
        for b in banned_run_dirs:
            if b in dat['SampleInfo']['SampleTable']['AnalysisHeader']['@FileName']:
                banned = True
        if banned:
            continue
        lc_method = None
        ms_method = None
        if isinstance(dat['SampleInfo'], list):
            continue
        if not 'polarity_1' in dat:
            continue
        if not re.match(run_id_regex, dat['SampleID']):
            continue
        for propdic in dat['SampleInfo']['SampleTable']['SampleTableProperties']['Property']:
            if propdic['@Name'] == 'HyStar_LC_Method_Name':
                lc_method = propdic['@Value']
            if propdic['@Name'] == 'HyStar_MS_Method_Name':
                ms_method = propdic['@Value']
        sample_names = {
            dat['SampleInfo']['SampleTable']['Sample']['@SampleID'],
            dat['SampleInfo']['SampleTable']['Sample']['@SampleID']+'.d',
            dat['SampleInfo']['SampleTable']['Sample']['@DataPath'],
        }
        samplerow = runlist[runlist['Raw file'].isin(sample_names)]
        if 'polarity_1_sers' in dat.keys():
            del dat['polarity_1_sers']
        if (lc_method is None) or (ms_method is None):
            print('LC or MS method not found')
            continue
        if len([k for k in dat.keys() if 'polarity' in k]) > 1:
            print('Multiple polarities found!')
            continue
        if samplerow.shape[0] == 0:
            samplerow = pd.Series(index = samplerow.columns, data = ['No data' for c in samplerow.columns])
        else:
            samplerow = samplerow.iloc[0]
        instrument = 'TimsTOF 1'
        runtime = datetime.strftime(
            datetime.strptime(
                dat['SampleInfo']['SampleTable']['AnalysisHeader']['@CreationDateTime'].split('+')[0],
                '%Y-%m-%dT%H:%M:%S'
            ),
            time_format
        )
        
        samplename = samplerow['Sample name']
        author = samplerow['Who']
        sample_type = samplerow['Sample type']
        bait = samplerow['Bait name']
        bait_uniprot = samplerow['Bait / other uniprot or ID']
        bait_mut = samplerow['Bait mutation']
        cell_line = samplerow['Cell line / material']
        project = samplerow['Project']
        author_notes = samplerow['Notes']
        bait_tag = samplerow['tag']
        try:
            precur = dat['NumPrecursors']
        except KeyError:
            precur = 'No precursor data'
        ms_run_row = [
            dat['SampleID'],
            dat['SampleInfo']['SampleTable']['AnalysisHeader']['@SampleID'],
            samplename,
            dat['SampleInfo']['SampleTable']['AnalysisHeader']['@FileName'],
            runtime,
            runtime.split()[0],
            instrument,
            author,
            sample_type,
            dat['DataType'],
            lc_method,
            ms_method,
            precur,
            bait,
            bait_uniprot,
            bait_mut,
            len(pd.Series(dat['polarity_1']['tic df']['Series'])),
            cell_line,
            project,
            author_notes,
            bait_tag,
            timestamp,
            -1
        ]
        for dataname in ['bpc filtered df', 'tic df', 'bpc unfiltered df']:
            ms_run_row.extend([
                dat['polarity_1'][dataname]['auc'],
                dat['polarity_1'][dataname]['intercepts'],
                dat['polarity_1'][dataname]['peaks_per_timepoint'],
                dat['polarity_1'][dataname]['mean_intensity'],
                dat['polarity_1'][dataname]['max_intensity'],
                json.dumps(dat['polarity_1'][dataname]['Series']),
                dat['polarity_1'][dataname]['trace'],
                json.dumps(dat['polarity_1'][dataname]['intercept_dict']),
            ])   
        
        new_data.append(ms_run_row)
    if len(new_data) > 0:
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(data = new_data, columns = ms_cols).to_csv(os.path.join(output_dir, f'{timestamp}_ms_runs.tsv'), sep='\t', index=False)

def update_external_data(conn, parameters, timestamp, organisms: set|None = None):
    """
    Updates external data tables in the database.

    Args:
        conn: SQLite database connection object
        parameters (dict): Configuration parameters including 'External data update interval days'
        timestamp (str): Current timestamp
        organisms (set): Set of organism IDs to update
    """ 
    uniprots = update_uniprot(conn, parameters, timestamp, organisms)
    update_knowns(conn, parameters, timestamp, uniprots, organisms)

def update_log_table(conn, inmod_names, inmod_vals, timestamp, uptype: str) -> None:
    """
    Records database update statistics in a log table.

    Args:
        conn: SQLite database connection object
        inmod_names (list): Names of tables with their modification types
        inmod_vals (list): Corresponding counts of insertions and modifications

    The log table is created if it doesn't exist, and new columns are added as needed.
    Each entry includes a timestamp and the counts of insertions and modifications for each table.
    """
    hasmods = sum(inmod_vals) > 0
    table_columns = ['timestamp TEXT PRIMARY KEY', 'update_type TEXT', 'modification_type TEXT', 'tablename TEXT', 'count INTEGER']
    new_rows = []
    if hasmods:
        modifications = {}
        for i, name in enumerate(inmod_names):
            table, mtype = name.rsplit(' ',maxsplit=1)
            modifications.setdefault(mtype, {}).setdefault(table, inmod_vals[i])
        for mtype, tables in modifications.items():
            for table, count in tables.items():
                new_rows.append((timestamp, uptype, mtype, table, count))
    else:
        new_rows.append((timestamp, uptype, 'no modifications','',0))
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
