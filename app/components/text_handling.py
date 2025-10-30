"""Text handling utilities for cleaning and normalizing strings.

Utilities for:
- Removing accent marks from characters
- Replacing special characters with specified replacements
- Combined accent and special character handling
- Simplified text cleaning interface
"""

from typing import Optional, Dict
import re
import unidecode

def remove_accent_characters(text: str) -> str:
    """Replace accented characters with their unaccented equivalents.

    :param text: Input string containing accented characters.
    :returns: String with accented characters replaced by unaccented equivalents.
    """
    return unidecode.unidecode(text)

def replace_special_characters(
    text: str,
    replacewith: str = '.',
    dict_and_re: bool = False,
    replacement_dict: Optional[Dict[str, str]] = None,
    stripresult: bool = True,
    remove_duplicates: bool = False,
    make_lowercase: bool = True,
    allow_numbers: bool = True,
    allow_space: bool = False,
    mask_first_digit: str|None = None
) -> str:
    """Replace special characters in a string with specified replacements.

    :param text: Input string containing special characters.
    :param replacewith: Character to use for replacement.
    :param dict_and_re: Whether to apply both dictionary replacements and regex.
    :param replacement_dict: Mapping of specific substrings to replacements.
    :param stripresult: Strip whitespace and replacement characters from result.
    :param remove_duplicates: Collapse consecutive replacement characters.
    :param make_lowercase: Convert result to lowercase.
    :param allow_numbers: Allow numbers in the result.
    :param allow_space: Allow spaces in the result.
    :param mask_first_digit: Character to prefix when first char is a digit.
    :returns: String with special characters replaced.
    """
    ret: str
    regex_pat = r'[^a-zA-Z0-9]'
    if allow_space:
        regex_pat = r'[^a-zA-Z0-9 ]'
    if not allow_numbers:
        regex_pat = regex_pat.replace('0-9', '')
    if not replacement_dict:
        ret = re.sub(regex_pat, replacewith, text)
    else:
        # Sort replacement keys by length (longest first) to handle overlapping patterns
        for key in sorted(list(replacement_dict.keys()), key=lambda x: len(x), reverse=True):
            if key in text:
                text = text.replace(key, replacement_dict[key])
        if dict_and_re:
            ret = re.sub(regex_pat, replacewith, text)
        else:
            new_text: list[str] = []
            for character in text:
                if not character.isalnum():
                    new_text.append(replacewith)
                else:
                    new_text.append(character)
            ret = ''.join(new_text)

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
            ret = ret.replace(f'{replacewith}{replacewith}', replacewith)
    if make_lowercase:
        ret = ret.lower()
    if mask_first_digit:
        if ret[0].isdigit():
            ret = mask_first_digit + ret
    return ret

def replace_accent_and_special_characters(
    text: str,
    **kwargs
) -> str:
    """Replace both accented and special characters in a string.

    :param text: Input string containing accented and special characters.
    :param kwargs: Passed through to ``replace_special_characters``.
    :returns: Cleaned string.
    """
    return replace_special_characters(remove_accent_characters(text), **kwargs)

def clean_text(text: str) -> str:
    """Simplified alias for ``replace_accent_and_special_characters``.

    :param text: Input string to clean.
    :returns: Cleaned string with default handling.
    """
    return replace_accent_and_special_characters(text)

def sanitize_for_database_use(text: str) -> str:
    """Sanitize a string for use in a database column name.

    :param text: Input string to sanitize.
    :returns: Sanitized string (alnum/underscore, prefixed if starting with digit).
    """
    return  replace_special_characters(remove_accent_characters(text), replacewith='_', allow_numbers=False, mask_first_digit = 'c')