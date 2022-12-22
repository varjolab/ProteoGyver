import os
from datetime import datetime, timedelta

def get_timestamp() -> str:
    return datetime.now().strftime('%Y-%m-%d')

def is_newer(reference:str, new_date:str) -> bool:
    reference: datetime.date = datetime.strptime(reference, '%y-%m-%d').date()
    new_date: datetime.date = datetime.strptime(new_date, '%Y-%m-%d').date()
    return (new_date>reference)

def get_save_location(databasename) -> str:
    base_location: str = 'Datafiles'
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
    return os.path.join('..','assets','nbibs',f'{databasename.lower()}.nbib')

def get_pub_ref(databasename:str) -> list:
    authors: list = []
    title: str = ''
    pmid: str = ''
    ref: str = ''
    pubyear: str = ''
    with open(get_nbibfile(databasename), 'r', encoding='utf-8') as fil:
        for line in fil:
            entry, data = line.split('-',maxsplit=1)
            entry: str=entry.strip()
            data: str=data.strip()
            if entry == 'TI':
                title = data
            elif entry == 'PMID':
                pmid = data
            elif entry == 'DP':
                pubyear = data.split()[0]
            elif entry =='AU':
                authors.append(data)
            elif entry == 'SO':
                ref = data
    if len(authors) < 3:
        short: str = f'({" and ".join(authors)}, {pubyear})'
    elif len(authors)==1:
        short = f'({authors[0]} ({pubyear})'
    else:
        short = f'({authors[0]} et al., {pubyear})'
    long: str = f'{", ".join(authors)}. {title} {ref}'
    return [short, long, pmid]
