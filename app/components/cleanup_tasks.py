import os
import shutil
import time
import logging
import zipfile
from celery import shared_task
from datetime import datetime, timedelta
from components.tools.utils import read_toml

logger = logging.getLogger("cleanup_tasks")


@shared_task
def cleanup_cache_folders():
    """
    Remove cache folders not touched in recently enough, as per parameter Maintenance->Clean unused cache interval days.
    If Maintenance->Archive cache dir is set, zip non-empty folders and move to archive, only delete empty folders.
    """
    parameters = read_toml('parameters.toml')
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