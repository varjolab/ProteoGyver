from datetime import datetime
import uuid
import pandas as pd
import tempfile
import sh
import numpy as np

def vsn(dataframe: pd.DataFrame, random_seed: int, errorfile: str) -> pd.DataFrame:
    """Apply VSN normalization via R's vsn::justvsn.

    :param dataframe: Input DataFrame (rows=ids in index; columns=samples).
    :param random_seed: Seed for reproducibility in R.
    :param errorfile: Base path to append Rscript stderr on failure.
    :returns: DataFrame of VSN-transformed values, aligned to input.
    """
    tempname: str = str(uuid.uuid4())
    tempdir: str = str(uuid.uuid4())
    dataframe.index.name = 'PROTID'
    script: list = [
        f'tempdir <- "{tempdir}"',
        'dir.create(tempdir, showWarnings = FALSE, recursive = TRUE)',
        'Sys.setenv(TMPDIR=tempdir)',  # Set temporary directory
        'library("vsn")',
        f'set.seed({random_seed})',
        f'setwd("{sh.pwd().strip()}")',
        f'Sys.setenv(R_USER="{sh.pwd().strip()}")',  # Set R user directory
        f'data <- read.table("{tempname}",sep="\\t",header=TRUE,row.names="{dataframe.index.name}")',
        'm = justvsn(data.matrix(data))',
        f'write.table(m,file="{tempname}",sep="\\t",col.names=NA,quote=FALSE)',
        ''
    ]
    return run_rscript(script, dataframe, tempname, errorfile, replace_dir = tempdir)

def run_rscript(r_script_contents:list, r_script_data: pd.DataFrame, replace_name: str, errorfile: str, replace_dir:str|None = None, input_df_has_index:bool = True):
    """Execute an R script with a temp data file and return parsed output.

    :param r_script_contents: Lines of the R script; occurrences of ``replace_name`` are replaced with temp paths.
    :param r_script_data: DataFrame to write to temp file for R to read.
    :param replace_name: Placeholder token to be replaced with temp filename.
    :param errorfile: Base path for error log on failure.
    :param replace_dir: Optional directory placeholder to replace with temp dir.
    :param input_df_has_index: Whether to include index when writing CSV/TSV.
    :returns: DataFrame parsed from R output file.
    :raises Exception: Re-raises Rscript execution errors after logging.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with tempfile.NamedTemporaryFile() as datafile:
            repwith = datafile.name
            r_script_contents = [line.replace(replace_name,repwith) for line in r_script_contents]
            # Ensure R finds packages installed into the CI user library
            lib_setup = [
                'lib_user <- Sys.getenv("R_LIBS_USER")',
                'if (nzchar(lib_user)) .libPaths(c(lib_user, .libPaths()))'
            ]
            r_script_contents = lib_setup + r_script_contents
            if replace_dir:
                r_script_contents = [line.replace(replace_dir,tmpdir) for line in r_script_contents]
            r_script_data.to_csv(datafile, sep='\t', index=input_df_has_index)
            with tempfile.NamedTemporaryFile() as scriptfile:
                scriptfile.write('\n'.join(r_script_contents).encode('utf-8'))
                scriptfile.flush()
                try:
                    sh.Rscript(scriptfile.name)
                except Exception as e:
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
    """Impute missing values using QRILC (via imputeLCMD).

    :param dataframe: Input DataFrame with missing values.
    :param random_seed: Seed for reproducibility.
    :param errorfile: Base path for error log on failure.
    :returns: DataFrame with imputed values.
    """
    tempname: str = str(uuid.uuid4())
    script: list = [
        'library("imputeLCMD")',
        f'set.seed({random_seed})',
        f'df <- read.csv("{tempname}",sep="\\t",row.names=1)',
        f'write.table(data.frame(impute.QRILC(df,tune.sigma=1)[1]),file="{tempname}",sep="\\t")'
    ]
    return run_rscript(script, dataframe, tempname, errorfile)

def impute_random_forest(dataframe: pd.DataFrame, random_seed: int, rev_sample_groups: dict, errorfile: str) -> pd.DataFrame:
    """Impute missing values using randomForest::rfImpute grouped by sample groups.

    :param dataframe: Input DataFrame with missing values (rows=ids; cols=samples).
    :param random_seed: Seed for reproducibility.
    :param rev_sample_groups: Mapping sample -> group for supervised imputation.
    :param errorfile: Base path for error log on failure.
    :returns: DataFrame with imputed values.
    """
    tempname: str = str(uuid.uuid4())
    with tempfile.NamedTemporaryFile() as groupsfile:
        groupsfile.write('sample\tgroup\n'.encode('utf-8'))
        groupsfile.write('\n'.join([f'{k}\t{v}' for k,v in rev_sample_groups.items()]).encode('utf-8'))
        groupsfile.flush()
        script: list = [
            'suppressPackageStartupMessages({',
            '  library(readr)',
            '  library(randomForest)',
            '})',

            f'INPUT_TSV  <- "{tempname}"   # col1 = protein IDs, rest = numeric samples',
            f'GROUPS_TSV <- "{groupsfile.name}"     # sample, group',
            f'OUTPUT_TSV <- "{tempname}"',

            f'SEED  <- {random_seed}',
            'NTREE <- 300',
            'ITER  <- 5',
            'set.seed(SEED)',
            'dat <- read_tsv(INPUT_TSV, col_types = cols(.default = col_guess()))',
            'id_colname  <- names(dat)[1]',
            'protein_ids <- as.character(dat[[1]])',
            'expr_df     <- dat[, -1, drop = FALSE]',
            'X           <- as.matrix(expr_df)',
            'rownames(X) <- protein_ids',
            'groups <- read_tsv(GROUPS_TSV,',
            '                   col_types = cols(sample = col_character(), group = col_character()))',
            'groups <- groups[match(colnames(X), groups$sample), , drop = FALSE]',
            'y <- factor(groups$group)',
            'if (!any(is.na(X))) {',
            '  out_df <- data.frame(protein_ids, X, check.names = FALSE)',
            '  names(out_df)[1] <- id_colname',
            '  write_tsv(out_df, OUTPUT_TSV)',
            '  cat("No NAs detected. Wrote input as final output.\n")',
            '  quit(save = "no", status = 0)',
            '}',
            'X_df <- as.data.frame(t(X))',
            'tmp <- tempfile()',
            'invisible(capture.output({',
            '  imp <- rfImpute(x = X_df, y = y, iter = ITER, ntree = NTREE)',
            '}, file = tmp))',
            'X_final <- t(as.matrix(imp[, -1, drop = FALSE]))',
            'out_df <- data.frame(protein_ids, X_final, check.names = FALSE)',
            'names(out_df)[1] <- id_colname',
            'write_tsv(out_df, OUTPUT_TSV)',
        ]
        return run_rscript(script, dataframe, tempname, errorfile)