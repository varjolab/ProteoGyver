import os
import time
from app.components import db_functions, parsing
import pandas as pd
from datetime import datetime


def update_table_with_file(cursor, table_name, file_path, parameters):
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
    try:
        # Get database table columns and their types
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
        for col in extra_cols:
            # Check if the column in the dataframe is numeric
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if is_numeric:
                col_type = 'NUMERIC'
            else:
                col_type = 'TEXT'
                
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
            print(f"Added new column '{col}' to table {table_name} with type {col_type}")

        modifications = 0
        insertions = 0
        primary_key = df.columns[0]  # Assuming the first column is the primary key

        for _, row in df.iterrows():
            pk_value = row[primary_key]
            
            # Check if the primary key already exists in the table
            cursor.execute(f"SELECT * FROM {table_name} WHERE {primary_key} = ?", (pk_value,))
            existing_entry = cursor.fetchone()

            if existing_entry:
                # Get column names for the existing entry
                cursor.execute(f"PRAGMA table_info({table_name})")
                all_columns = [row[1] for row in cursor.fetchall()]
                
                # Create a dictionary of the new values
                new_values = row.to_dict()
                
                # If there's an existing entry, preserve values for missing columns
                for i, col in enumerate(all_columns):
                    if col in missing_cols:
                        new_values[col] = existing_entry[i]
                            
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
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

    except Exception as e:
        print(f"Failed to update table {table_name} with file {file_path}: {e}")
    return insertions, modifications

def update_database(conn, parameters):
    """
    Updates multiple database tables using TSV files from specified directories.

    Args:
        conn: SQLite database connection object
        parameters (dict): Configuration parameters including 'Update files' with table-to-directory mappings

    Returns:
        tuple: (inmod_names, inmod_vals, hasmods)
            - inmod_names (list): Names of tables with their modification types
            - inmod_vals (list): Corresponding counts of insertions and modifications
            - hasmods (bool): Whether any modifications were made
    """
    update_files = parameters['Update files']
    inmod_names = []
    inmod_vals = []
    hasmods = False
    with conn.cursor() as cursor:  # Use context manager for cursor
        for table_name in update_files:
            insertions = 0
            modifications = 0
            folder_path = os.path.join(update_files[table_name])
            if os.path.exists(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.tsv'):
                        file_path = os.path.join(folder_path, file_name)
                        insertions, modifications = update_table_with_file(cursor, table_name, file_path, parameters)
                        if (insertions + modifications) > 0:
                            hasmods = True
            else:
                os.makedirs(folder_path)
                print(f"Created directory {folder_path}")
            inmod_names.append(f'{table_name} insertions')
            inmod_vals.append(insertions)
            inmod_names.append(f'{table_name} modifications')
            inmod_vals.append(modifications)
        conn.commit()
        return inmod_names, inmod_vals, hasmods

def update_log_table(conn, inmod_names, inmod_vals):
    """
    Records database update statistics in a log table.

    Args:
        conn: SQLite database connection object
        inmod_names (list): Names of tables with their modification types
        inmod_vals (list): Corresponding counts of insertions and modifications

    The log table is created if it doesn't exist, and new columns are added as needed.
    Each entry includes a timestamp and the counts of insertions and modifications for each table.
    """
    # Create update_log table if it doesn't exist
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS update_log (
            timestamp TEXT PRIMARY KEY
        )
    """)
    # Add any new columns that don't exist
    cursor.execute("PRAGMA table_info(update_log)")
    existing_columns = {row[1] for row in cursor.fetchall()} - {'timestamp'}
    for col in inmod_names:
        if col not in existing_columns:
            cursor.execute(f"ALTER TABLE update_log ADD COLUMN {col} INTEGER")
    
    # Insert the new log entry
    columns = ['timestamp'] + inmod_names
    placeholders = ','.join(['?'] * (len(inmod_names) + 1))
    values = [datetime.now().strftime('%Y%m%d%H%M%S')] + inmod_vals
    cursor.execute(f"""
        INSERT INTO update_log ({','.join(columns)})
        VALUES ({placeholders})
    """, values)
    conn.commit()

if __name__ == "__main__":
    while True:
        parameters = parsing.read_toml('parameters.toml')
        database_path = parameters['Data paths']['Database file']
        uniprot_fields = parameters['Database creation']['Uniprot fields']
        parameters = parameters['Database updater']

        update_interval = int(parameters['Update interval seconds'])
        api_update_interval = int(parameters['External data update interval days'])

        db_path = input("Enter the path to the SQLite3 database file: ").strip()
        output_dir = input("Enter the directory to save TSV templates: ").strip()
        primary_keys_file = input("Enter the path to the JSON file with primary keys: ").strip()
        try:
            conn = db_functions.create_connection(db_path)
            inmod_names, inmod_vals, hasmods = update_database(conn, parameters)
            if hasmods:
                update_log_table(conn, inmod_names, inmod_vals)
            db_functions.generate_database_table_templares_as_tsvs(conn, output_dir, parameters['Database table primary keys'])
            conn.close() # type: ignore
        except Exception as e:
            print(f"Failed to update database: {e}")
        time.sleep(update_interval)