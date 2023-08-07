from typing import Any

def list_to_chunks(original_list, chunk_size):
    # looping till length l
    for i in range(0, len(original_list), chunk_size):
        yield original_list[i:i + chunk_size]
        
def format_dictionary(dic: dict, level = 0) -> list:
    """Formats a dictionary into a more file-output friendly format."""
    retlist: list = []
    prefix: str = '\t'*level
    keys: list = sorted(list(dic.keys()))
    for k in keys:
        retlist.append(f'{prefix}{k}:')
        v: Any = dic[k]
        if isinstance(v, dict):
            retlist.extend(format_dictionary(v, level=level+1))
        elif isinstance(v, list) or isinstance(v,set):
            retlist.extend([
                f'{prefix}\t{l}' for l in v
            ])
        else:
            retlist.append(f'\t{v}')
    if level > 0:
        return retlist
    else:
        return '\n'.join(retlist)   