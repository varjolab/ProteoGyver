from __future__ import annotations

import os
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from celery import shared_task
from celery_once import QueueOnce

try:
    import tomllib  # Python 3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

# Import the pipeline runner
try:
    from run_as_pipeline import run_batch_pipeline
except Exception:  # pragma: no cover
    run_batch_pipeline = None  # type: ignore


LOCK_FILENAME = ".pg_analyzing.lock"
ERRORS_FILENAME = "ERRORS.txt"
PG_OUTPUT_DIRNAME = "PG output"


def _now() -> float:
    return time.time()


def _write_text(target_file: Path, content: str) -> None:
    target_file.write_text(content)


def _iter_all_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            files.append(Path(dirpath) / name)
    return files


def _latest_mtime_in_tree(root: Path) -> float:
    latest: float = 0.0
    for file_path in _iter_all_files(root):
        try:
            mtime = file_path.stat().st_mtime
            if mtime > latest:
                latest = mtime
        except FileNotFoundError:
            # File might be in flux; ignore
            continue
    return latest


def _tree_is_stable(root: Path, stable_seconds: int = 60) -> bool:
    if not root.exists():
        return False
    latest_mtime = _latest_mtime_in_tree(root)
    return (_now() - latest_mtime) >= stable_seconds


def _select_pipeline_toml(dir_path: Path) -> Tuple[Optional[Path], Optional[str]]:
    # Case-insensitive discovery of .toml files
    tomls = sorted([p for p in dir_path.iterdir() if p.is_file() and p.name.lower().endswith(".toml")])
    if len(tomls) == 0:
        return None, (
            "Pipeline toml file not found. Add a pipeline .toml file." 
        )
    if len(tomls) == 1:
        return tomls[0], None
    # Multiple: prefer one ending with pipeline.toml
    for t in tomls:
        if t.name.lower().endswith("pipeline.toml"):
            return t, None
    return None, (
        "Pipeline toml file not found. If there are multiple toml files in the directory, "
        "one of them MUST have a name ending in \"pipeline.toml\"."
    )


def _load_toml(toml_path: Path) -> Dict[str, Any]:
    if tomllib is None:
        raise RuntimeError("tomllib is unavailable in this Python environment")
    with open(toml_path, "rb") as fh:
        return tomllib.load(fh)


def _validate_pipeline_toml(toml_path: Path) -> Optional[str]:
    try:
        data = _load_toml(toml_path)
    except Exception as e:
        return f"Failed to parse TOML: {e}"

    # Required top-level keys
    for key in ("workflow", "data", "sample table"):
        if key not in data:
            return f"Required key missing in TOML: {key}"

    # Validate data path
    data_path_value = data["data"]
    sample_table_value = data["sample table"]

    def _resolve_path(value: Any) -> Optional[Path]:
        if isinstance(value, str):
            p = (toml_path.parent / value).resolve()
            return p
        return None

    data_path = _resolve_path(data_path_value)
    sample_table_path = _resolve_path(sample_table_value)

    if data_path is None or not data_path.exists():
        return f"Data path does not exist: {data_path_value}"
    if sample_table_path is None or not sample_table_path.exists():
        return f"Sample table path does not exist: {sample_table_value}"

    return None


def _has_error_file(dir_path: Path) -> bool:
    return (dir_path / ERRORS_FILENAME).exists()


def _pg_output_dir(dir_path: Path) -> Path:
    return dir_path / PG_OUTPUT_DIRNAME


def _outside_files_latest_mtime(dir_path: Path, exclude: Path) -> float:
    latest: float = 0.0
    for dp, _, fns in os.walk(dir_path):
        current_dir = Path(dp)
        if exclude in current_dir.parents or current_dir == exclude:
            continue
        for fn in fns:
            fp = current_dir / fn
            try:
                mt = fp.stat().st_mtime
                if mt > latest:
                    latest = mt
            except FileNotFoundError:
                continue
    return latest


def _should_reanalyze(dir_path: Path) -> bool:
    # Condition 1: no ERROR.txt or ERRORS.txt present
    if _has_error_file(dir_path):
        return False
    output_dir = _pg_output_dir(dir_path)
    if not output_dir.exists():
        return True
    try:
        output_mtime = output_dir.stat().st_mtime
    except FileNotFoundError:
        return True
    outside_latest = _outside_files_latest_mtime(dir_path, exclude=output_dir)
    return outside_latest > output_mtime


def _is_currently_analyzing(dir_path: Path, max_age_seconds: int = 3600) -> bool:
    lock_path = dir_path / LOCK_FILENAME
    if not lock_path.exists():
        return False
    try:
        mtime = lock_path.stat().st_mtime
        if (_now() - mtime) > max_age_seconds:
            # Stale lock: clean up
            lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            return False
    except Exception:
        # On any error checking the lock, assume stale and remove
        try:
            lock_path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        return False
    return True


def _mark_analyzing(dir_path: Path) -> None:
    (dir_path / LOCK_FILENAME).write_text(datetime.now().isoformat())


def _clear_analyzing(dir_path: Path) -> None:
    try:
        (dir_path / LOCK_FILENAME).unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass


def _safe_write_error(dir_path: Path, filename: str, message: str) -> None:
    try:
        _write_text(dir_path / filename, message)
    except Exception:
        # Last resort: attempt to write to a timestamped error file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _write_text(dir_path / f"{filename}.{ts}", message)


def _launch_pipeline(dir_path: Path, toml_file: Path) -> Optional[str]:
    if run_batch_pipeline is None:
        return "run_batch_pipeline is unavailable in this environment"
    try:
        run_batch_pipeline(str(toml_file))
        return None
    except Exception as e:
        tb = traceback.format_exc()
        return f"Pipeline execution failed: {e}\n\n{tb}"


@shared_task(base=QueueOnce, once={"graceful": True})
def watch_pipeline_input(watch_directory: list[str]) -> None:
    watch_dir = Path(*watch_directory)
    if not watch_dir.exists() or not watch_dir.is_dir():
        return

    # Iterate immediate subdirectories of the watch directory
    for entry in sorted(watch_dir.iterdir()):
        if not entry.is_dir():
            continue

        # Skip if currently analyzing
        if _is_currently_analyzing(entry):
            continue

        # Determine whether to process (new or re-analyze conditions)
        output_dir = _pg_output_dir(entry)
        should_process: bool
        if output_dir.exists():
            should_process = _should_reanalyze(entry)
        else:
            should_process = True

        if not should_process:
            continue

        # If there is an error marker, skip per policy
        if _has_error_file(entry):
            continue

        # Ensure directory tree is stable (not being copied to)
        if not _tree_is_stable(entry, stable_seconds=60):
            continue

        # Resolve TOML selection
        toml_file, selection_error = _select_pipeline_toml(entry)
        if selection_error is not None or toml_file is None:
            _safe_write_error(entry, ERRORS_FILENAME, selection_error or "Unknown TOML selection error")
            continue

        # Validate TOML keys and paths
        validation_error = _validate_pipeline_toml(toml_file)
        if validation_error is not None:
            _safe_write_error(entry, ERRORS_FILENAME, validation_error)
            continue

        # Mark as analyzing to prevent duplicate runs
        _mark_analyzing(entry)
        try:
            # Double-check stability right before launch
            if not _tree_is_stable(entry, stable_seconds=60):
                _clear_analyzing(entry)
                continue

            # Launch the pipeline
            error_message = _launch_pipeline(entry, toml_file)
            if error_message:
                _safe_write_error(entry, ERRORS_FILENAME, error_message)
        finally:
            _clear_analyzing(entry)