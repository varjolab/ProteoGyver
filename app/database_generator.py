import sqlite3
import os
from components.tools import utils

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

def create_sqlite_from_schema(schema_file: str | Path,
                              db_file: str | Path,
                              overwrite: bool = False,
                              pragmas: Optional[Iterable[str]] = ("foreign_keys=ON","journal_mode=WAL")) -> Path:
    """Create a SQLite database from a .sql schema file.

    Args:
        schema_file: Path to the schema).
        db_file: Path of the database to create.
        pragmas: PRAGMAs to apply after connecting (e.g., ("foreign_keys=ON",)).

    Returns:
        Path: Absolute path to the created database.

    Raises:
        FileNotFoundError: If `schema_file` doesn’t exist.
        FileExistsError: If `db_file` exists and `overwrite=False`.
        sqlite3.Error: If executing the schema fails.
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


if __name__ == '__main__':
    parameters = utils.read_toml(Path('parameters.toml'))
    dbfile = os.path.join(*parameters['Data paths']['Database file'])
    schema_file = os.path.join(*parameters['Data paths']['Schema file'])
    create_sqlite_from_schema(schema_file, dbfile) 