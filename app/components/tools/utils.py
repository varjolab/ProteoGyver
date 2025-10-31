import re
import tomlkit

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union, Mapping, MutableMapping
from zipfile import ZipFile, ZIP_DEFLATED, ZipInfo

def load_toml(path: Path) -> dict[str, Any]:
    """Load a TOML file into a Python dict using tomlkit.

    :param path: Path to TOML file.
    :returns: Parsed dict.
    """
    with path.open("r", encoding="utf-8") as f:
        return tomlkit.parse(f.read())

def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep-merge ``override`` into ``base``.

    Rules:
    - Dicts merge recursively.
    - Scalars/lists replace the base value.

    :param base: Destination mapping to mutate.
    :param override: Source mapping providing updates.
    :returns: Mutated ``base`` mapping.
    """
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            deep_merge(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base

def save_toml(data: dict[str, Any], path: Path) -> None:
    """Write a TOML document to disk using tomlkit.

    :param data: Dict to serialize.
    :param path: Output path.
    :returns: None
    """
    with path.open("w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(data))
        
def prefix_relative_paths(
    data: Dict[str, Any],
    basepath: Union[str, os.PathLike, None] = None,
    check_exists: bool = True
) -> Dict[str, Any]:
    """Recursively prefix relative paths in a nested dict.

    Values can be strings, lists, or dicts.

    :param data: Input dictionary.
    :param basepath: Base path to prepend; if None, uses global ``BASEPATH``.
    :param check_exists: If ``True``, only replace when resulting path exists.
    :returns: New dict with adjusted paths.
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
    """Traverse nested dict by list of keys.

    :param in_dict: Input mapping.
    :param key_path: Sequence of keys to descend.
    :returns: Value at the nested path.
    """
    if len(key_path) > 1:
        return dig_dict(in_dict[key_path[0]], key_path[1:])
    else:
        return in_dict[key_path[0]]

def expand_path(path, param_dict):
    """Expand a 'params:' reference into a concrete path from a dict.

    :param path: String possibly starting with ``'params:'``.
    :param param_dict: Parameter dictionary.
    :returns: Expanded path string or original value.
    """
    if path.startswith('params:'):
        tokens = path.split(':')[1].split('.')
        ret = dig_dict(param_dict, tokens)
    else:
        ret = path
    if isinstance(ret, list):
        ret = os.path.join(*ret)
    return ret

def expand_paths(in_dict, param_dict):
    """Recursively expand 'params:' paths in a nested dict structure.

    :param in_dict: Input dict to mutate.
    :param param_dict: Parameter dictionary supplying values.
    :returns: Mutated input dict with paths expanded.
    """
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
    """Read a TOML file and optionally prefix/expand embedded paths.

    :param toml_file: TOML file path.
    :param baseify: Top-level keys whose values should get path prefixing.
    :param check_exists: If ``True``, only replace with existing paths.
    :param expand_paths_in_full_dict: If ``True``, expand ``params:`` references.
    :returns: Parsed and transformed dictionary.
    """
    toml_path = Path(toml_file)
    basepath = str(toml_path.parent.parent.resolve())
    with toml_path.open('r', encoding='utf-8') as tf:
        data = tomlkit.load(tf)
    for key in baseify:
        data[key] = prefix_relative_paths(data[key], basepath, check_exists)
    if expand_paths_in_full_dict:
        data = expand_paths(data, data)
    return data

def normalize_key(s: str) -> str:
    """Normalize a string for consistent matching.

    Steps:
    - Lowercase and strip directory/extension.
    - Replace non-alphanumerics with underscore; collapse repeats.
    - Trim leading/trailing underscores.

    :param s: Input string (e.g., filename).
    :returns: Normalized key string.
    """
    if s is None:
        return ""
    s = s.lower().split('/')[-1].split('\\')[-1].rsplit('.',maxsplit=1)[0]
    s = re.sub(r'[^a-z0-9]+', '_', s)  # replace anything not a-z0-9 with "_"
    s = re.sub(r'_+', '_', s)          # collapse multiple "_"
    s = s.strip('_')                   # trim leading/trailing "_"
    return s

def zipdir(src_dir: str | Path, tmpfs_base: str | Path | None = None) -> Path:
    """Zip a directory and return the created archive path.

    The zip contains paths prefixed by the source directory's basename.

    :param src_dir: Directory to zip.
    :param tmpfs_base: Base directory for temporary working folder (unused placeholder).
    :returns: Path to the created ZIP file.
    :raises FileNotFoundError: If ``src_dir`` doesn't exist.
    :raises NotADirectoryError: If ``src_dir`` is not a directory.
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
