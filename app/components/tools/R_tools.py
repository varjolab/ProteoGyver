import os
import subprocess
import platform
import textwrap
import uuid
import pandas as pd


def impute_qrilc(dataframe: pd.DataFrame, rpath:str=None, tempdir:str=None) -> pd.DataFrame:
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