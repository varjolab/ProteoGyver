import os
import subprocess
import platform
import textwrap
import uuid
import pandas as pd
import tempfile
import sh
import numpy as np

def vsn(dataframe: pd.DataFrame, random_seed: int, errorfile: str) -> pd.DataFrame:
    """Does vsn transformation on a dataframe using justvsn from vsn package."""
    tempname: uuid.UUID = str(uuid.uuid4())
    dataframe.to_csv('debug_testvsn.csv')
    dataframe.index.name = 'PROTID'
    script: list = [
        'library("vsn")',
        f'set.seed({random_seed})',
        f'setwd("{sh.pwd().strip()}")',
        f'data <- read.table("{tempname}",sep="\\t",header=TRUE,row.names="{dataframe.index.name}")',
        'm = justvsn(data.matrix(data))',
        f'write.table(m,file="{tempname}",sep="\\t",col.names=NA,quote=FALSE)'
    ]
    return run_rscript_2(script, dataframe, [tempname], errorfile)

def run_rscript_2(r_script_contents:list, r_script_data: pd.DataFrame, replace_names: list, errorfile: str):
    with tempfile.NamedTemporaryFile() as datafile:
        repwith = datafile.name
        for repwhat in replace_names:
            r_script_contents = [line.replace(repwhat,repwith) for line in r_script_contents]
        r_script_data.to_csv(datafile, sep='\t')
        with tempfile.NamedTemporaryFile() as scriptfile:
            scriptfile.write('\n'.join(r_script_contents).encode('utf-8'))
            scriptfile.flush()
            try:
                sh.Rscript(scriptfile.name)
            except Exception as e:
                from datetime import datetime
                datestr = str(datetime.now()).split()[0] # quick n dirty way to get just the date without time
                with open(f'{errorfile}.txt','a') as fil:
                    fil.write(f'===================\n{datestr}\n\n{e}\n\n{str(e.stderr)}\n-----------------------\n')
                raise e
            with open(datafile.name, "r") as f:
                out = f.read().split('\n')
                out = [o.split('\t')[1:] for o in out[1:] if len(o)>0] # skip empty rows
    script_output_df = pd.DataFrame(data = out)
    script_output_df = script_output_df.replace('NA',np.nan).replace('',np.nan).astype(float)
    script_output_df.columns = r_script_data.columns # Restore columns and index in case R renames anything from either.
    script_output_df.index = r_script_data.index
    return script_output_df

def impute_qrilc(dataframe: pd.DataFrame, random_seed: int, errorfile: str) -> pd.DataFrame:
    tempname: uuid.UUID = str(uuid.uuid4())
    script: list = [
        'library("imputeLCMD")',
        f'set.seed({random_seed})',
        f'df <- read.csv("{tempname}",sep="\\t",row.names=1)',
        f'write.table(data.frame(impute.QRILC(df,tune.sigma=1)[1]),file="{tempname}",sep="\\t")'
    ]
    return run_rscript_2(script, dataframe, [tempname], errorfile)

def impute_qrilc_old(dataframe: pd.DataFrame, random_seed: int, rpath:str=None, tempdir:str=None) -> pd.DataFrame:
    """Impute missing values in dataframe using QRILC method

    Calls an R function to qrilc-impute missing values into the input dataframe.
    Input dataframe should only have numerical data with missing values.

    Parameters:
    df: pandas dataframe with the missing values. Should not have any text columns
    rpath: path to Rscript.exe
    """
    if rpath is None:
        rpath: str = 'C:\\Program Files\\R\\newest-r\\bin'
    if not tempdir: 
        tempdir: str = '.'
    tempname: uuid.UUID = uuid.uuid4()
    temp_r_file: str = os.path.join(tempdir,f'fromimpute_{tempname}.R')
    tempdffile: str = os.path.join(tempdir,f'fromimpute_{tempname}_df.tsv')
    tempdffile_dest: str = os.path.join(tempdir,f'fromimpute_{tempname}_dest_df.tsv')
    dataframe.to_csv(tempdffile, sep='\t')

    with open(temp_r_file, 'w', encoding='utf-8') as fil:
        fil.write(textwrap.dedent(f"""
                    library("imputeLCMD")
                    set.seed({random_seed})
                    df <- read.csv("{tempdffile}",sep="\\t",row.names=1)
                    df2 = impute.QRILC(df, tune.sigma = 1)
                    imputed = df2[1]
                    df3 = data.frame(imputed)
                    write.table(imputed,file="{tempdffile_dest}",sep="\\t")
                    """))
    process: subprocess.CompletedProcess = run_r_script(temp_r_file, rpath=rpath)
    df2: pd.DataFrame = pd.DataFrame()
    try:
        df2 = pd.read_csv(tempdffile_dest, index_col=0, sep='\t')
    except FileNotFoundError:
        errormsg: str = '\n'.join([
            'R script FAILED for QRILC imputation.',
            'Some likely causes include: ',
            '- Columns or rows with nothing but missing values in input',
            '- Rscript.exe not in given path',
            '-----------------------------------------------------------',
            'Detailed error information:',
            'Subprocess exit code:',
            f'{process.returncode}',
            '-----',
            'Subprocess stderr:',
            f'{process.stderr.decode()}',
            '-----',
            'Subprocess stdout:',
            f'{process.stdout.decode()}'
        ])

        raise RuntimeError(errormsg) from None
    finally:
        for tempfile in [temp_r_file, tempdffile, tempdffile_dest]:
            try:
                os.remove(tempfile)
            except PermissionError:
                time.sleep(2)
                os.remove(tempfile)
            except FileNotFoundError:
                continue

    df2.index.name = dataframe.index.name

    column_first_letter: list = list({x[0] for x in df2.columns})
    if len(column_first_letter) == 1:
        column_first_letter = column_first_letter[0]
        if column_first_letter in ('X', 'Y'):
            rename_dict: dict = {c: c[1:] for c in df2.columns}
            df2.rename(columns=rename_dict, inplace=True)
    df2.columns = dataframe.columns
    return df2

def get_newest_r(rpath) -> str:
    """Returns rpath, where the string "newest-r" has been replaced by the newest R version found\
        in the preceding directory"""
    sort_list: list = []
    rfolds: dict = {}
    rbase: str
    rend: str
    rbase, rend = rpath.split(os.sep+'newest-r'+os.sep)
    for filename in os.listdir(rbase):
        if not os.path.isdir(os.path.join(rbase, filename)):
            continue
        folder_tuple: tuple = tuple(int(i) for i in filename.split('-')[1].split('.'))
        sort_list.append(folder_tuple)
        rfolds[folder_tuple] = filename
    sort_list = sorted(sort_list, reverse=True)
    return os.path.join(rbase, rfolds[sort_list[0]], rend)

def run_r_script(scriptfilepath: str, rpath: str = 'C:\\Program Files\\R\\newest-r\\bin',
                output_file: str = None) -> subprocess.CompletedProcess:
    """Runs an R script

    Parameters:
    script_file: path to script file.
    rpath: Path to bin directory of R, where Rscript.exe is located. If you use 'newest-r' instead\
        of the actual R directory (e.g. 'R-4.2.1'), program will default to newest R version\
            identified.
    output_file: filename where to save output, if desired
    """
    if platform.system() == 'Linux': ## Linux
        rpath: str = 'Rscript'
    elif 'newest-r' in rpath: ## Windows probably
        rpath: str = get_newest_r(rpath)
        rpath = os.path.join(rpath, 'Rscript.exe')
    rpath = 'Rscript'
    cmd: list = [rpath, scriptfilepath]
    process: subprocess.CompletedProcess = subprocess.run(cmd, capture_output=True, check=False)
    if output_file:
        output: list = process.args
        output.extend([
            '-------',
            'STDOUT:',
            process.stdout.decode("utf-8"),
            '-------',
            'STDERR:',
            process.stderr.decode("utf-8"),
        ])
        with open(output_file, 'w', encoding='utf-8') as fil:
            fil.write('\n'.join(output))
    return process