import os
import shutil
import time
import logging
import zipfile
from pathlib import Path
from celery import shared_task
from datetime import datetime, timedelta
from components.tools.utils import read_toml

logger = logging.getLogger("cleanup_tasks")

@shared_task
def cleanup_cache_folders():
    """Clean old cache folders and optionally archive them.

    Uses parameters in ``parameters.toml`` under ``Maintenance -> Cleanup``.
    For each configured directory, folders older than the interval are removed;
    if an archive directory is configured, non-empty folders are zipped and moved
    there, while empty folders are deleted.

    :returns: None
    """
    logger.info("Starting cleanup of cache folders")
    parameters = read_toml(Path('config/parameters.toml'))
    cleanup_settings = parameters.get('Maintenance', {}).get('Cleanup', {})
    if not cleanup_settings:
        return
    for dirname, settings in cleanup_settings.items():
        if settings['Path'] == 'Data paths':
            cache_dir = os.path.join(*parameters['Data paths'][dirname])
        else:
            cache_dir = os.path.join(*settings['Path'])
        days_unused = settings['Clean unused interval days']
        archive_dir = settings['Archive cache dir'].strip()
    

        if isinstance(days_unused, str):
            if days_unused.upper() == 'NO CLEANUP':
                continue
        else:
            days_unused = int(days_unused)

        now = time.time()
        cutoff = now - days_unused * 24 * 3600
        removed = []
        archived = []

        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)

        for entry in os.listdir(cache_dir):
            path = os.path.join(cache_dir, entry)
            manual_days_since = datetime.now() - datetime.strptime(entry.split('--')[0], "%Y-%m-%d-%H-%M-%S")
            if os.path.isdir(path):
                last_access = os.path.getatime(path)
                last_mod = os.path.getmtime(path)
                if last_access < cutoff and last_mod < cutoff and manual_days_since > timedelta(days=days_unused):
                    # Archive mode
                    if archive_dir:
                        # Remove empty folders
                        if not os.listdir(path):
                            try:
                                shutil.rmtree(path)
                                removed.append(entry)
                            except Exception as e:
                                logger.warning(f"During cleanup of {dirname}, failed to remove empty folder {path}: {e}")
                        else:
                            # Zip non-empty folder
                            zip_name = f"{entry}.zip"
                            zip_path = os.path.join(archive_dir, zip_name)
                            try:
                                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                                    for root, dirs, files in os.walk(path):
                                        for file in files:
                                            abs_file = os.path.join(root, file)
                                            arcname = os.path.relpath(abs_file, path)
                                            zipf.write(abs_file, arcname)
                                archived.append(zip_name)
                            except Exception as e:
                                logger.warning(f"During cleanup of {dirname}, failed to archive {path}: {e}")
                    # Normal cleanup mode
                    else:
                        try:
                            shutil.rmtree(path)
                            removed.append(entry)
                        except Exception as e:
                            logger.warning(f"During cleanup of {dirname}, failed to remove {path}: {e}")

        logger.info(f"Cleanup for {dirname} complete. Removed: {removed}. Archived: {archived}")


@shared_task
def rotate_logs():
    """Rotate and prune logs per configuration in parameters.toml.

    Configuration keys:
    - ``Config.LogDir``: logs directory.
    - ``Config."Log compress days"``: age (days) to compress into monthly zips.
    - ``Config."Log keep days"``: age (days) to delete logs/zips.

    Log files are expected to start with ``YYYY-MM-DD_``; monthly zips are named
    ``YYYY-MM_logs.zip``.

    :returns: None
    """
    parameters = read_toml(Path('config/parameters.toml'))
    config = parameters.get('Config', {})
    log_dir = Path(config.get('LogDir', 'logs'))
    compress_days = int(config.get('Log compress days', 7))
    keep_days = int(config.get('Log keep days', 365))

    if not log_dir.exists() or not log_dir.is_dir():
        logger.warning(f"Log directory does not exist or is not a directory: {log_dir}")
        return

    now = datetime.now()

    # Helpers to parse dates from filenames
    def parse_log_date(name: str):
        # Expect prefix YYYY-MM-DD_
        try:
            return datetime.strptime(name[:10], "%Y-%m-%d")
        except Exception:
            return None

    def parse_zip_month(name: str):
        # Expect prefix YYYY-MM_ or YYYY-MM (we use YYYY-MM_logs.zip)
        try:
            return datetime.strptime(name[:7], "%Y-%m")
        except Exception:
            return None

    # Group log files by calendar year-month for compression
    month_to_files = {}
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() == '.zip':
            continue
        file_dt = parse_log_date(entry.name)
        if not file_dt:
            continue
        age_days = (now - file_dt).days
        if age_days < compress_days:
            continue
        ym_key = (file_dt.year, file_dt.month)
        month_to_files.setdefault(ym_key, []).append(entry)

    # Compress per month with max compression and then remove originals
    for (yy, mm), files in sorted(month_to_files.items()):
        zip_name = f"{yy}-{mm:02d}_logs.zip"
        zip_path = log_dir / zip_name
        try:
            # Append-safe: avoid duplicate arcnames
            existing = set()
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    existing = set(zf.namelist())
            with zipfile.ZipFile(zip_path, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
                for f in files:
                    arc = f.name
                    if arc in existing:
                        continue
                    zf.write(str(f), arc)
            # Remove originals after successful compression
            for f in files:
                try:
                    f.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete original log after compression: {f}: {e}")
        except Exception as e:
            logger.error(f"Failed to compress logs for month {yy}-{mm:02d}: {e}")

    # Deletion phase: remove logs or zips older than keep_days based on their timestamp
    for entry in log_dir.iterdir():
        if not entry.is_file():
            continue
        cutoff_dt = None
        if entry.suffix.lower() == '.zip':
            mo_dt = parse_zip_month(entry.name)
            cutoff_dt = mo_dt
        else:
            cutoff_dt = parse_log_date(entry.name)
        if not cutoff_dt:
            continue
        if (now - cutoff_dt).days > keep_days:
            try:
                entry.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete old log artifact: {entry}: {e}")