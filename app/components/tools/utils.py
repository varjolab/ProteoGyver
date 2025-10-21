import re
import tomlkit

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union, Mapping, MutableMapping
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

def load_toml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return tomlkit.parse(f.read())

def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """
    Deep-merge 'override' into 'base'.
    - Dicts merge recursively.
    - Scalars/lists replace the base value.
    """
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_merge(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base

def save_toml(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(data))
        
def prefix_relative_paths(
    data: Dict[str, Any],
    basepath: Union[str, os.PathLike, None] = None,
    check_exists: bool = True
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
            if check_exists:
                final_path = v if os.path.isabs(v) else os.path.join(str(basepath), v)
                if os.path.exists(final_path):
                    return final_path
                else:
                    return v
            else:
                return v if os.path.isabs(v) else os.path.join(str(basepath), v)
        if isinstance(v, list):
            if not v:
                return []
            # Convert to regular Python list to avoid tomlkit object issues
            v = list(v)
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

def dig_dict(in_dict, key_path):
    if len(key_path) > 1:
        return dig_dict(in_dict[key_path[0]], key_path[1:])
    else:
        return in_dict[key_path[0]]

def expand_path(path, param_dict):
    if path.startswith('params:'):
        tokens = path.split(':')[1].split('.')
        ret = dig_dict(param_dict, tokens)
    else:
        ret = path
    if isinstance(ret, list):
        ret = os.path.join(*ret)
    return ret

def expand_paths(in_dict, param_dict):
    for key, value in in_dict.items():
        test_str = None
        if isinstance(value, str):
            test_str = value
            in_dict[key] = expand_path(value, param_dict)
        elif isinstance(value, list) and (len(value) > 0) and isinstance(value[0], str):
            value[0] = expand_path(value[0], param_dict)
        elif isinstance(value, dict):
            value = expand_paths(value, param_dict)
    return in_dict

def read_toml(toml_file: Union[str, Path], baseify = ['Data paths'], check_exists = False, expand_paths_in_full_dict = True):
    toml_path = Path(toml_file)
    basepath = str(toml_path.parent.resolve())
    with toml_path.open('r', encoding='utf-8') as tf:
        data = tomlkit.load(tf)
    for key in baseify:
        data[key] = prefix_relative_paths(data[key], basepath, check_exists)
    if expand_paths_in_full_dict:
        data = expand_paths(data, data)
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
