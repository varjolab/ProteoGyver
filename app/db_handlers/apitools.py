import os
from datetime import datetime, timedelta
import nbib

def get_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d')

def parse_timestamp_from_str(stamp_text:str) -> str:
    return datetime.strptime(stamp_text,'%Y-%m-%d')


def is_newer(reference:str, new_date:str) -> bool:
    reference: datetime.date = datetime.strptime(reference, '%y-%m-%d').date()
    new_date: datetime.date = datetime.strptime(new_date, '%Y-%m-%d').date()
    return (new_date>reference)

def get_save_location(databasename) -> str:
    base_location: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'Datafiles'
    )
    dir_name: str = os.path.join(base_location, databasename)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    return dir_name

def get_files_newer_than(directory:str, date:str, days:int, namefilter:str=None) -> list:
    if not os.path.isdir(directory):
        return ''
    # Convert the date string to a datetime object
    date:str = datetime.strptime(date, '%Y-%m-%d')
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

def get_newest_file(directory, namefilter:str = None) -> str:
    if not os.path.isdir(directory):
        return ''
    # Initialize variables to store the name and modification time of the newest file
    newest_file: str = ''
    newest_time = datetime.min
    directory: str = os.path.join(__file__.rsplit(os.sep)[0],'Datafiles',directory)
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

def get_nbibfile(databasename:str) -> str:
    nbibpath: str = os.path.join(
        os.path.dirname(os.path.realpath(__file__)).rsplit(os.sep,maxsplit=1)[0],
        'assets',
        'nbibs',
        f'{databasename.lower()}.nbib')
    return nbibpath

def get_pub_ref(databasename:str) -> list:
    
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
