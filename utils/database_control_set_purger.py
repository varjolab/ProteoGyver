#!/usr/bin/env python3
"""
Purge control_* tables based on entries in the `control_sets` table.

For each value in `control_sets.control_set`, this script will:
  - DROP TABLE IF EXISTS control_{value}
  - DROP TABLE IF EXISTS control_{value}_overall
  - DELETE FROM control_sets WHERE control_set = {value}

All operations happen inside a single transaction with foreign keys enabled.

Exit codes:
  0 = success
  1 = error
  2 = dry-run (no changes made)
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from typing import Iterable, List, Tuple


SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def fetch_control_sets(conn: sqlite3.Connection) -> List[str]:
    """Fetch all ``control_set`` values from the ``control_sets`` table.

    :param conn: Open SQLite connection.
    :type conn: sqlite3.Connection
    :returns: List of ``control_set`` values.
    :rtype: List[str]
    """
    cur = conn.execute("SELECT control_set FROM control_sets")
    return [row[0] for row in cur.fetchall()]


def validate_identifier_piece(piece: str) -> bool:
    """Validate a single identifier component for table names.

    Only characters matching the pattern ``[A-Za-z0-9_]+`` are allowed.

    :param piece: Identifier piece to validate.
    :type piece: str
    :returns: ``True`` if safe, else ``False``.
    :rtype: bool
    """
    return bool(SAFE_NAME.fullmatch(piece))


def drop_table_if_exists(conn: sqlite3.Connection, table: str) -> None:
    """Drop a table if it exists.

    :param conn: Open SQLite connection.
    :type conn: sqlite3.Connection
    :param table: Table name (assumed safe/validated).
    :type table: str
    :raises sqlite3.Error: If the DROP statement fails.
    """
    # Double-quote the identifier; we've validated allowed chars already.
    sql = f'DROP TABLE IF EXISTS "{table}"'
    conn.execute(sql)


def delete_control_set_row(conn: sqlite3.Connection, value: str) -> int:
    """Delete the row from ``control_sets`` for a specific value.

    :param conn: Open SQLite connection.
    :type conn: sqlite3.Connection
    :param value: ``control_set`` value to delete.
    :type value: str
    :returns: Number of rows deleted.
    :rtype: int
    :raises sqlite3.Error: If the DELETE statement fails.
    """
    cur = conn.execute("DELETE FROM control_sets WHERE control_set = ?", (value,))
    return cur.rowcount or 0


def process(conn: sqlite3.Connection, dry_run: bool = False) -> Tuple[int, int, List[str]]:
    """Process all control sets: drop tables and delete rows.

    For each value in ``control_sets.control_set`` this will drop
    ``control_{value}`` and ``control_{value}_overall`` (if present) and then
    remove the corresponding row from ``control_sets``. All operations are
    executed within a single transaction unless ``dry_run`` is enabled.

    :param conn: Open SQLite connection.
    :type conn: sqlite3.Connection
    :param dry_run: If ``True``, do not modify the database; only report actions.
    :type dry_run: bool
    :returns: A tuple ``(tables_dropped, rows_deleted, warnings)``.
    :rtype: Tuple[int, int, List[str]]
    :raises sqlite3.Error: If any SQL statement fails when not in dry-run mode.
    :raises Exception: For unexpected errors; transaction is rolled back.
    """
    warnings: List[str] = []
    values = fetch_control_sets(conn)

    tables_dropped = 0
    rows_deleted = 0

    # Use a transaction for the whole batch
    if not dry_run:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN")

    try:
        for v in values:
            if not validate_identifier_piece(v):
                warnings.append(
                    f"Skipped unsafe control_set value {v!r} "
                    "(only [A-Za-z0-9_] allowed)."
                )
                continue

            t1 = f"control_{v}"
            t2 = f"control_{v}_overall"

            if dry_run:
                print(f"[DRY-RUN] Would drop tables: {t1}, {t2}")
                print(f"[DRY-RUN] Would delete row from control_sets where control_set={v!r}")
                continue

            # Drop tables if they exist
            drop_table_if_exists(conn, t1)
            tables_dropped += 1
            drop_table_if_exists(conn, t2)
            tables_dropped += 1

            # Delete the row
            rows_deleted += delete_control_set_row(conn, v)

        if not dry_run:
            conn.commit()

    except Exception as e:
        if not dry_run:
            conn.rollback()
        raise
    return tables_dropped, rows_deleted, warnings


def main() -> int:
    """CLI entry point.

    Parses command line arguments and coordinates the purge operation.

    :returns: Exit code (``0`` success, ``1`` error, ``2`` dry-run).
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Delete control_* tables and corresponding rows based on control_sets."
    )
    parser.add_argument("db", help="Path to the SQLite database file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making changes.",
    )
    args = parser.parse_args()

    try:
        with sqlite3.connect(args.db) as conn:
            # Slightly stricter isolation (optional)
            conn.isolation_level = None  # We'll manage transactions manually
            tables_dropped, rows_deleted, warnings = process(conn, dry_run=args.dry_run)

            for w in warnings:
                print(f"WARNING: {w}", file=sys.stderr)

            conn.execute("VACUUM")

            print(
                f"Done. Tables dropped: {tables_dropped}, rows deleted from control_sets: {rows_deleted}."
            )

        if args.dry_run:
            return 2
        return 0

    except sqlite3.Error as e:
        print(f"SQLite error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
