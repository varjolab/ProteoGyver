"""Text handling utilities for cleaning and normalizing strings.

This module provides functions for handling special characters, accented characters,
and general text cleaning operations. It includes utilities for:
    - Removing accent marks from characters
    - Replacing special characters with specified replacements
    - Combined accent and special character handling
    - Simplified text cleaning interface

Example:
    >>> from app.components.text_handling import clean_text
    >>> clean_text("Hello, café!")
    "hello.cafe"

Typical usage example:
    text = "Some raw text with special chars: café, étude!"
    cleaned = clean_text(text)
    # Or for more control:
    cleaned = replace_special_characters(text, replacewith="_", 
                                       remove_duplicates=True)
"""

from typing import Optional, Dict
import re
import unidecode

def remove_accent_characters(text: str) -> str:
    """Replaces accented characters with their unaccented equivalents in a string.
    
    Args:
        text (str): Input string containing accented characters
        
    Returns:
        str: String with accented characters replaced by unaccented equivalents
        
    Example:
        >>> remove_accent_characters("café")
        "cafe"
    """
    return unidecode.unidecode(text)

def replace_special_characters(
    text: str,
    replacewith: str = '.',
    dict_and_re: bool = False,
    replacement_dict: Optional[Dict[str, str]] = None,
    stripresult: bool = True,
    remove_duplicates: bool = False,
    make_lowercase: bool = True
) -> str:
    """Replaces special characters in a string with specified replacements.
    
    Args:
        text (str): Input string containing special characters
        replacewith (str, optional): Character to replace special characters with. Defaults to '.'
        dict_and_re (bool, optional): Whether to apply both dictionary replacements and regex. 
            Defaults to False
        replacement_dict (dict, optional): Dictionary mapping specific special characters to 
            their replacements. Defaults to None
        stripresult (bool, optional): Whether to strip whitespace and replacement characters 
            from result. Defaults to True
        remove_duplicates (bool, optional): Whether to remove consecutive replacement 
            characters. Defaults to False
        make_lowercase (bool, optional): Whether to convert result to lowercase. 
            Defaults to True
            
    Returns:
        str: String with special characters replaced according to specifications
        
    Example:
        >>> replace_special_characters("Hello, World!", replacewith="_")
        "hello_world"
        >>> replace_special_characters("Hello, World!", 
        ...                          replacement_dict={",": " COMMA "})
        "hello COMMA world"
    """
    ret: str
    if not replacement_dict:
        ret = re.sub(r'[^a-zA-Z0-9]', replacewith, text)
    else:
        # Sort replacement keys by length (longest first) to handle overlapping patterns
        for key in sorted(list(replacement_dict.keys()), key=lambda x: len(x), reverse=True):
            if key in text:
                text = text.replace(key, replacement_dict[key])
        if dict_and_re:
            ret = re.sub(r'[^a-zA-Z0-9]', replacewith, text)
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
    return ret

def replace_accent_and_special_characters(
    text: str,
    replacewith: str = '.',
    replacement_dict: Optional[Dict[str, str]] = None
) -> str:
    """Replaces both accented and special characters in a string.
    
    Args:
        text (str): Input string containing accented and special characters
        replacewith (str, optional): Character to replace special characters with. 
            Defaults to '.'
        replacement_dict (dict, optional): Dictionary mapping specific special characters 
            to their replacements. Defaults to None
            
    Returns:
        str: String with both accented and special characters replaced
        
    Example:
        >>> replace_accent_and_special_characters("café, étude!")
        "cafe.etude"
    """
    text: str = replace_special_characters(text, replacewith=replacewith, 
                                         replacement_dict=replacement_dict)
    return remove_accent_characters(text)

def clean_text(text: str) -> str:
    """Simplified alias for replace_accent_and_special_characters.
    
    Args:
        text (str): Input string to clean
        
    Returns:
        str: Cleaned string with default accent and special character handling
        
    Example:
        >>> clean_text("Hello, café!")
        "hello.cafe"
    """
    return replace_accent_and_special_characters(text)