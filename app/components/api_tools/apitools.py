import os
from datetime import datetime, timedelta, date
import nbib
import random
from pathlib import Path
import string
__all__ = [
    'get_timestamp', 'parse_timestamp_from_str', 'is_newer',
    'get_nbibfile', 'get_save_location', 'get_files_newer_than',
    'get_newest_file', 'get_pub_ref'
]

def get_timestamp() -> str:
    """
    Returns standardized timestamp.

    :returns: standardized timestamp for current day as a string.
    """
    return datetime.now().strftime('%Y-%m-%d')

def parse_timestamp_from_str(stamp_text:str) -> date:
    """
    Parses standardized time stamp from string.
    :param stamp_text: str with the timestamp

    :returns: Datetime for the timestamp
    """
    return datetime.strptime(stamp_text,'%Y-%m-%d').date()


def is_newer(reference:str, new_date:str) -> bool:
    """
    Checks whether new_date is newer than reference

    :param reference: reference timestamp string
    :param new_date: new_date timestamp string

    :returns: True, if new_date is more recent than reference.
    """
    reference_date: date = parse_timestamp_from_str(reference)
    new_date_date: date = parse_timestamp_from_str(new_date)
    return (new_date_date>reference_date)

def get_nbibfile(databasename:str) -> str:
    """
    Fetches the nbib reference file for the given database. Does not check, if the file exists.

    :param databasename: name of the database

    :returns: path to the nbib file for that database.
    """
    nbibpath: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'nbibs',
        f'{databasename.lower()}.nbib')
    return nbibpath

def get_save_location(databasename) -> str:
    """
    Finds the appropriate location to save a database file

    :param databasename: Name of the database
    :returns: path to the database directory, where files should be written.
    """
    base_location: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'data'
    )
    dir_name: str = os.path.join(base_location, databasename)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

def get_files_newer_than(directory:str, date_str:str, days:int, namefilter:str=None) -> list:
    """
    Gets files newer than a given date in a dictionary, only going back a given number of days. Uses os.path.getmtime to establish file dates.

    :param directory: where to search
    :param date: base date for the search, e.g. today
    :param days: how many days move the cutoff date to the past from the date parameter
    :param namefilter: only return files, which have this string in their name

    :returns: list of files newer than date-days, that optionally contain namefilter in their name.
    """
    if not os.path.isdir(directory):
        return ''
    # Convert the date string to a datetime object
    date:datetime = datetime.strptime(date_str, '%Y-%m-%d')
    # Calculate the cutoff date
    cutoff_date = date - timedelta(days=days)
    newer_files: list = []
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # Check if the file should be checked, if namefilter is set:
        if namefilter:
            if not namefilter in filename:
                continue
        file_path:str = os.path.join(directory, filename)
        # Get the modification time of the file
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        # If the modification time is after the cutoff date, add the file to the list
        if mod_time > cutoff_date:
            newer_files.append(filename)
    return newer_files

def get_newest_file(directory:str, namefilter:str = None) -> str:
    """
    Finds the latest modified file in a directory

    :param directory: where to search
    :param namefilter: only consider files with this in their name

    :returns: name of the newest file, not the full path. If no files are found, returns a randomized string.
    """
    if not os.path.isdir(directory):
        return ''
    existing_files = set(os.listdir(directory))
    newest_file = ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    while newest_file in existing_files:
        newest_file += ''.join(random.choices(string.ascii_letters + string.digits, k=3))
    newest_time = datetime.min
    for filename in os.listdir(directory):
        # Check if the file should be checked, if namefilter is set:
        if namefilter:
            if not namefilter in filename:
                continue
        file_path:str = os.path.join(directory, filename)
        
        # Get the modification time of the file
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        if mod_time > newest_time:
            newest_file = filename
            newest_time = mod_time
    
    return newest_file

def get_pub_ref(databasename:str) -> list:
    """
    Fetches reference information for the given database

    :param databasename: which database to get information for

    :returns: list of reference information: [
        short description,
        long description,
        PMID
    ]
    """
    
    nbibdata: list = nbib.read_file(get_nbibfile(databasename))[0]
    pubyear: str = nbibdata['publication_date'].split(maxsplit=1)[0]
    try:
        authors: list = [a['author_abbreviated'] for a in nbibdata['authors']]
    except KeyError:
        authors = [nbibdata['corporate_author']]
    ref: str = f'{nbibdata["journal_abbreviated"]}.{nbibdata["publication_date"]}:{nbibdata["doi"]}'
    title: str = nbibdata['title']
    pmid: str = str(nbibdata['pubmed_id'])
    if len(authors) < 3:
        short: str = f'({" and ".join(authors)}, {pubyear})'
    elif len(authors)==1:
        short = f'({authors[0]} ({pubyear})'
    else:
        short = f'({authors[0]} et al., {pubyear})'
    long: str = f'{", ".join(authors)}. {title} {ref}'
    return [short, long, pmid]
