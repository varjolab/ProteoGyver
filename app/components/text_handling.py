
import re
import unidecode

def remove_accent_characters(text:str) -> str:
    """Replaces accented characters with a unaccented equivalents in string."""
    return unidecode.unidecode(text)

def replace_special_characters(text:str,replacewith:str='.', replacement_dict: dict=None, stripresult: bool=True, remove_duplicates:bool = False, make_lowercase: bool = True) -> str:
    """Replaces special(e.g. -,_,/,},(,\,)) characters with a given character or replacement dict in string.
    Replacement dict can be used to replace specific special characters with user input, \
        while all others will be replaced by replacewith

    """
    ret: str
    if not replacement_dict:
        ret = re.sub(r'[^a-zA-Z0-9]', replacewith, text)
    else:
        new_text = []
        for character in text:
            if character in replacement_dict:
                new_text.append(replacement_dict[character])
            elif not character.isalnum():
                new_text.append(replacewith)
            else:
                new_text.append(character)
        ret =  ''.join(new_text)
    if stripresult:
        curlen: int = -1
        while len(ret) != curlen:
            curlen = len(ret)
            ret = ret.strip()
            ret = ret.strip(replacewith)
    if remove_duplicates:
        curlen: int = -1
        while len(ret) != curlen:
            curlen = len(ret)
            ret = ret.replace(f'{replacewith}{replacewith}',replacewith)
    if make_lowercase:
        ret = ret.lower()
    return ret

def replace_accent_and_special_characters(text:str,replacewith:str='.', replacement_dict=None) -> str:
    """Replaces accented and special(e.g. -,_,/,},(,\,)) characters with a given character or \
        replacement dict in string."""
    text: str = replace_special_characters(text,replacewith=replacewith,replacement_dict=replacement_dict)
    return remove_accent_characters(text)

def clean_text(text: str) -> str:
    """Simplified alias for replace_accent_and_special_characters"""
    return replace_accent_and_special_characters(text)