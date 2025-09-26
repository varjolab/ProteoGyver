import re
import tomlkit

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

def prefix_relative_paths(
    data: Dict[str, Any],
    basepath: Union[str, os.PathLike, None] = None,
) -> Dict[str, Any]:
    """Recursively prefix relative paths in a nested dict.

    Values are strings, lists, or dicts.
    - If value is dict → recurse.
    - If value is str → if not absolute, prepend basepath.
    - If value is list[str] → look at first element; if it's relative,
      prepend basepath to every relative string in the list.

    Args:
        data: Input dictionary.
        basepath: Base path to prepend. If None, uses global BASEPATH.

    Returns:
        A new dictionary with paths adjusted.
    """
    if basepath is None:
        basepath = BASEPATH

    def _absify(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: _absify(val) for k, val in v.items()}
        if isinstance(v, str):
            return v if os.path.isabs(v) else os.path.join(str(basepath), v)
        if isinstance(v, list):
            if not v:
                return []
            first = v[0]
            # Only treat as a list of paths if first element is a string
            if isinstance(first, str) and not os.path.isabs(first):
                out: List[Any] = [os.path.join(str(basepath), first)] + v[1:]
                return out
            else:
                return v[:]  # leave as-is
        # Any other type is left untouched
        return v

    return _absify(data)

def read_toml(toml_file, baseify = ['Data paths']):
    basepath = os.path.dirname(os.path.realpath(toml_file))
    with open(toml_file, 'r') as tf:
        data = tomlkit.load(tf)
    for key in baseify:
        data[key] = prefix_relative_paths(data[key], basepath)
    return data

def normalize_key(s: str) -> str:
    """
    Normalize strings for consistent matching:
      - Lowercase
      - Replace all non-alphanumeric chars with underscore
      - Collapse multiple underscores
      - Strip leading/trailing underscores
    """
    if s is None:
        return ""
    s = s.lower().split('/')[-1].split('\\')[-1].rsplit('.',maxsplit=1)[0]
    s = re.sub(r'[^a-z0-9]+', '_', s)  # replace anything not a-z0-9 with "_"
    s = re.sub(r'_+', '_', s)          # collapse multiple "_"
    s = s.strip('_')                   # trim leading/trailing "_"
    return s

def zipdir(src_dir: str | Path, tmpfs_base: str | Path | None = None) -> Path:
    """Zip a directory into a tmpfs-backed unique folder and return the zip path.

    The zip will contain paths prefixed by the source directory's basename, e.g.:
        dirname/file1
        dirname/subdir/file2

    Args:
        src_dir: Path to the directory to zip.
        tmpfs_base: Optional base directory for the temporary working folder.
            Defaults to the first writeable of ['/dev/shm', f'/run/user/{uid}', '/tmp'].

    Returns:
        Path: Full path to the created ZIP file.

    Raises:
        FileNotFoundError: If src_dir doesn't exist.
        NotADirectoryError: If src_dir is not a directory.
    """
    src = Path(src_dir).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    if not src.is_dir():
        raise NotADirectoryError(src)

    tmp_base = '/tmp'
    # Unique working directory on tmpfs
    workdir = Path(tempfile.mkdtemp(prefix="zipwork-", dir=tmp_base))
    zip_path = workdir / f"{src.name}.zip"

    # Create the zip with deterministic structure: 'dirname/...'
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        # Add directories explicitly so empty dirs are preserved
        for root, dirs, files in os.walk(src):
            rel_root = os.path.relpath(root, src)
            arcdir = src.name if rel_root == "." else f"{src.name}/{rel_root}"

            # directory entry (ensure trailing slash)
            zinfo = ZipInfo(arcdir.rstrip("/") + "/")
            zinfo.external_attr = 0o40755 << 16  # drwxr-xr-x
            zf.writestr(zinfo, b"")

            # files
            for fname in files:
                fpath = Path(root) / fname
                rel_file = os.path.relpath(fpath, src)
                arcname = f"{src.name}/{rel_file}"
                zf.write(fpath, arcname)

    return zip_path
