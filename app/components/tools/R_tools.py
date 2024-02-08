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
    dataframe.index.name = 'PROTID'
    script: list = [
        'library("vsn")',
        f'set.seed({random_seed})',
        f'setwd("{sh.pwd().strip()}")',
        f'data <- read.table("{tempname}",sep="\\t",header=TRUE,row.names="{dataframe.index.name}")',
        'm = justvsn(data.matrix(data))',
        f'write.table(m,file="{tempname}",sep="\\t",col.names=NA,quote=FALSE)'
    ]
    return run_rscript(script, dataframe, [tempname], errorfile)

def run_rscript(r_script_contents:list, r_script_data: pd.DataFrame, replace_names: list, errorfile: str):
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
    return run_rscript(script, dataframe, [tempname], errorfile)
