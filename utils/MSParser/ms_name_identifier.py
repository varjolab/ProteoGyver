import os

def identify(raw_file_path:str) -> str:
    
    if raw_file_path.lower().endswith('.d'):
        ms_name = 'Timppa'
        if '_tomppa' in raw_file_path.lower():
            ms_name = 'Tomppa'
    elif raw_file_path.lower().endswith('.raw'):
        filename = raw_file_path.split('/')[-1].split('\\')[-1]
        if filename.lower().startswith('qe_'):
            ms_name = 'QE'
        elif filename.lower().startswith('oe'):
            ms_name = 'Elite'
        else:
            path_list = []
            for sep in ['\\','/']:
                if sep in raw_file_path:
                    path_list = raw_file_path.split(sep)
            if len(path_list) == 0:
                ms_name = 'ERROR'
            else:
                ms_name = path_list[-2]
    return ms_name