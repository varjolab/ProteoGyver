import re
import tomlkit

import os
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

def read_toml(toml_file):
    with open(toml_file, 'r') as tf:
        data = tomlkit.load(tf)
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
