import os
import qnorm
import pandas as pd
import io
import time
import json
import numpy as np
import base64
from typing import Tuple
import platform
import uuid
import subprocess
from importlib import util as import_util
import textwrap
from typing import Any

class DataFunctions:

    @property
    def handlers(self) -> dict:
        return self._api_handlers
    
    @property
    def available_enrichments(self) -> list:
        if self._available_enrichments is None:
            available: list = []
            for handlername, handlerdict in self.handlers.items():
                for enrichment in handlerdict['available']['enrichment']:
                    available.append(': '.join([handlerdict['name'], enrichment]))
            self._available_enrichments = available
        return self._available_enrichments

    @property
    def default_enrichments(self, apis:list=None) -> list:
        if self._default_enrichments is None:
            defaults: list = []
            for name, handlerdict in self.handlers.items():
                if apis is not None:
                    if (name not in apis) and (handlerdict['name'] not in apis):
                        continue
                for enrichment in handlerdict['defaults']:
                    defaults.append(': '.join([handlerdict['name'], enrichment]))
            self._default_enrichments = defaults
        return self._default_enrichments


    def enrich_all_per_bait(self, data_table: pd.DataFrame,enrichment_strings: list) -> list:
        enrichments_to_do: dict = {}
        for e_str in enrichment_strings:
            apiname:str
            enrichment_name:str
            apiname, enrichment_name = e_str.split(': ',maxsplit=1)
            apiname = apiname.lower()
            if apiname not in enrichments_to_do:
                enrichments_to_do[apiname] = []
            enrichments_to_do[apiname].append(enrichment_name)
        enrichment_results: list = []
        enrichment_names: list = []
        for api, enrichmentlist in enrichments_to_do.items():
            enrichment_str: str = ';'.join(enrichmentlist)
            names: list
            api_enrichment_result: list
            names, api_enrichment_result = self.enrich_per_bait(data_table, api, enrichment_str)
            enrichment_names.extend(names)
            enrichment_results.extend(api_enrichment_result)
        return (enrichment_names, enrichment_results)

    def enrich_per_bait(self, data_table: pd.DataFrame, api: str, options: str) -> list:
        enrich_lists: list = []
        for bait in data_table['Bait'].unique():
            enrich_lists.append([bait, list(data_table[data_table['Bait']==bait]['Prey'])])
        handler: Any = self.handlers[api]['handler']
        result_names: list
        return_dataframes: list
        done_information: list
        result_names, return_dataframes, done_information = handler.enrich(enrich_lists, options)
        if not 'Enrichment' in self._done_operations:
            self._done_operations['Enrichment'] = {}
        self._done_operations['Enrichment'][api] = done_information
        return (result_names, return_dataframes)

    def __init__(self, module_base_dir: str) -> None:
        self._api_handlers: dict = {}
        self._handler_types: dict = {}
        self._done_operations: dict = {}
        self._default_enrichments: list = None
        self._available_enrichments: list = None
        self.import_handlers(module_base_dir)
	
    def register_handler(self, module_name: str, filepath: str) -> None:
        module_name: str = module_name.replace('.py','')
        spec = import_util.spec_from_file_location(
            'module.name', filepath)
        api_module = import_util.module_from_spec(spec)
        spec.loader.exec_module(api_module)
        handler = api_module.handler()
        self._api_handlers[module_name] = {
            'handler': handler,
            'available': handler.get_available(),
            'name': handler.nice_name,
            'defaults': handler.get_default_panel()
        }
        for htype in handler.handler_types:
            if htype not in self._handler_types:
                self._handler_types[htype] = []
            self._handler_types[htype].append(module_name)

    def import_handlers(self, module_dir_root) -> None:
        for root, _, files in os.walk(module_dir_root):
            for f in files:
                if f.endswith('.py'):
                    self.register_handler(f, os.path.join(root,f))


    def read_df_from_content(self, content, filename) -> pd.DataFrame:
        """Reads a dataframe from uploaded content.
        
        Filenames ending with ".csv" are read as comma separated, filenames ending with ".tsv", ".tab" or
        ".txt" are read as tab-delimed files, and ".xlsx" and ".xls" are read as excel files.
        Filename ending identification is case-insensitive.
        """
        _: str
        content_string: str
        _, content_string = content.split(',')
        decoded_content: bytes = base64.b64decode(content_string)
        f_end: str = filename.rsplit('.', maxsplit=1)[-1].lower()
        data = None
        if f_end == 'csv':
            data: pd.DataFrame = pd.read_csv(io.StringIO(
                decoded_content.decode('utf-8')), index_col=False)
        elif f_end in ['tsv', 'tab', 'txt']:
            data: pd.DataFrame = pd.read_csv(io.StringIO(
                decoded_content.decode('utf-8')), sep='\t', index_col=False)
        elif f_end in ['xlsx', 'xls']:
            data: pd.DataFrame = pd.read_excel(io.StringIO(decoded_content))
        return data

    def map_known(self, data_table:pd.DataFrame, known_data) -> None:
        if 'Bait uniprot' in data_table.columns:
            known_ints:dict
            new_columns:list
            known_ints, new_columns = known_data

            known_col_values: dict = {c: [] for c in new_columns}
            any_known_col:list = []
            for _,row in data_table.iterrows():
                #values: list = [np.nan for _ in new_columns]
                any_known: bool = False
                for k in new_columns:
                    try:
                        val: str = known_ints[row['Bait uniprot']][row['Prey']][k]
                        any_known = True
                       # values[i] = known_ints[row['Bait uniprot']][row['Prey']][k]
                    except KeyError:
                        # Bait, prey, or key not known
                        val = np.nan
                        #continue
                    known_col_values[k].append(val)
                any_known_col.append(any_known)
                #for i, value in enumerate(values):
                 #   known_col_values[i].append(value)
            data_table['Known interaction'] = any_known_col
            for column_name, column_data in known_col_values.items():
                data_table['Known ' + column_name] = column_data
            #data_table['Known interaction'] = pd.Series(known_col_values[1]).notna()
            #for i, column_name in enumerate(new_columns):
            #    data_table['Known ' + column_name] = known_col_values[i]
                    
    def median_normalize(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Median-normalizes a dataframe by dividing each column by its median.

        Args:
            df (pandas.DataFrame): The dataframe to median-normalize.
            Each column represents a sample, and each row represents a measurement.

        Returns:
            pandas.DataFrame: The median-normalized dataframe.
        """
        # Calculating the medians prior to looping is about 2-3 times more efficient,
        # than calculating the median of each column inside of the loop.
        medians: pd.Series = data_frame.median(axis=0)
        mean_of_medians: float = medians.mean()
        for col in data_frame.columns:
            data_frame[col] = (data_frame[col] / medians[col]) * mean_of_medians
        return data_frame


    def quantile_normalize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Quantile-normalizes a dataframe.

        Args:
            df (pandas.DataFrame): The dataframe to quantile-normalize.
            Each column represents a sample, and each row represents a measurement.

        Returns:
            pandas.DataFrame: The quantile-normalized dataframe.
        """
        return qnorm.quantile_normalize(dataframe)



    def count_per_sample(self, data_table: pd.DataFrame, rev_sample_groups: dict) -> pd.Series:
        """Counts non-zero values per sample (sample names from rev_sample_groups.keys()) and returns a series with sample names in index and counts as values."""
        index: list = list(rev_sample_groups.keys())
        retser: pd.Series = pd.Series(
            index=index,
            data=[data_table[i].notna().sum() for i in index]
        )
        return retser

    def normalize(self, data_table: pd.DataFrame, normalization_method: str) -> pd.DataFrame:
        """Normalizes a given dataframe with the wanted method."""
        if normalization_method == 'Median':
            data_table: pd.DataFrame = self.median_normalize(data_table)
        elif normalization_method == 'Quantile':
            data_table = self.quantile_normalize(data_table)
        return data_table

        
    def filter_missing(self, data_table: pd.DataFrame, sample_groups: dict, threshold: int = 60) -> pd.DataFrame:
        """Discards rows with more than threshold percent of missing values in all sample groups"""
        threshold: int = threshold/100
        keeps: list = []
        for _, row in data_table.iterrows():
            keep: bool = False
            for _, sample_columns in sample_groups.items():
                keep = keep | (row[sample_columns].notna().sum()
                            >= (threshold*len(sample_columns)))
                if keep:
                    break
            keeps.append(keep)
        return data_table[keeps]


    def impute(self, data_table:pd.DataFrame, method:str='QRILC', tempdir:str='.') -> pd.DataFrame:
        """Imputes missing values into the dataframe with the specified method"""
        ret: pd.DataFrame = data_table
        if method == 'minProb':
            ret = self.impute_minprob_df(data_table)
        elif method == 'minValue':
            ret = self.impute_minval(data_table)
        elif method == 'QRILC':
            ret = self.impute_qrilc(data_table, tempdir = tempdir)
        return ret

    def impute_qrilc(self, dataframe: pd.DataFrame, rpath:str=None, tempdir:str=None) -> pd.DataFrame:
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
        process: subprocess.CompletedProcess = self.run_r_script(temp_r_file, rpath=rpath)
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


    def get_newest_r(self, rpath) -> str:
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

    def run_r_script(self, scriptfilepath: str, rpath: str = 'C:\\Program Files\\R\\newest-r\\bin',
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
            rpath: str = self.get_newest_r(rpath)
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

    def impute_minval(self, dataframe: pd.DataFrame, impute_zero:bool=False) -> pd.DataFrame:
        """Impute missing values in dataframe using minval method

        Input dataframe should only have numerical data with missing values.
        Missing values will be replaced by the minimum value of each column.

        Parameters:
        df: pandas dataframe with the missing values. Should not have any text columns
        impute_zero: True, if zero should be considered a missing value
        """
        newdf: pd.DataFrame = pd.DataFrame(index=dataframe.index)
        for column in dataframe.columns:
            newcol: pd.Series = dataframe[column]
            if impute_zero:
                newcol = newcol.replace(0,np.nan)
            newcol = newcol.fillna(newcol.min())
            newdf.loc[:, column] = newcol
        return newdf

    def impute_gaussian(self, data_table: pd.DataFrame, dist_width: float = 0.3, dist_down_shift: float = 1.8) -> pd.DataFrame:
        """Impute missing values in dataframe using values from random numbers from normal distribution.

        Based on the method used by Perseus (http://www.coxdocs.org/doku.php?id=perseus:user:activities:matrixprocessing:imputation:replacemissingfromgaussian)

        Parameters:
        data_table: pandas dataframe with the missing values. Should not have any text columns
        dist_width: Gaussian distribution relative to stdev of each column. 
            Value of 0.5 means the width of the distribution is half the standard deviation of the sample column values.
        dist_down_shift: How far downwards the distribution is shifted. By default, 1.8 standard deviations down.
        """
        newdf: pd.DataFrame = pd.DataFrame(index=data_table.index)
        for column in data_table.columns:
            newcol: pd.Series = data_table[column]
            stdev: float = newcol.std()
            distribution: np.ndarray = np.random.normal(
                    loc = newcol.mean - (dist_down_shift*stdev),
                    scale = dist_width*stdev,
                    size = column.shape[0]*100
                )
            replace_values: pd.Series = pd.Series(
                index = data_table.index,
                data = np.random.choice(a=distribution,size=column.shape[0],replace=False)
            )
            newcol = newcol.fillna(replace_values)
            newdf.loc[:, column] = newcol
        return newdf

    def impute_minprob(self, series_to_impute: pd.Series, scale: float = 1.0,
                    tune_sigma: float = 0.01, impute_zero=True) -> pd.Series:
        """Imputes missing values with randomly selected entries from a distribution \
            centered around the lowest non-NA values of the series.

        Arguments:
        series_to_impute: pandas series with possible missing values

        Keyword arguments:
        scale: passed to numpy.random.normal
        tune_sigma: fraction of values from the lowest end of the series to use for \
            generating the distribution
        impute_zero: treat 0 values as missing values and impute new values for them
        """

        ser: pd.Series = series_to_impute.sort_values(ascending=True)
        ser = ser[ser > 0].dropna()
        ser = ser[:int(len(ser)*tune_sigma)]

        # implement q value
        distribution: np.ndarray = np.random.normal(
            loc=ser.median(), scale=scale, size=len(series_to_impute*100))

        output_series: pd.Series = series_to_impute.copy()
        for index, value in output_series.items():
            impute_value: bool = False
            if pd.isna(value):
                impute_value = True
            elif (value == 0) and impute_zero:
                impute_value = True
            if impute_value:
                output_series[index] = np.random.choice(distribution)
        return output_series

    def impute_minprob_df(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """imputes whole dataframe with minprob imputation. Dataframe should only have numerical columns

        Parameters:
        df: dataframe to impute
        kwargs: keyword args to pass on to impute_minprob
        """
        newdf: pd.DataFrame = pd.DataFrame(index=dataframe.index)
        for column in dataframe.columns:
            newdf.loc[:, column] = self.impute_minprob(dataframe[column], **kwargs)
        return newdf


    def read_dia_nn(self, data_table: pd.DataFrame) -> pd.DataFrame:
        """Reads dia-nn report file into an intensity matrix"""
        protein_col: str = 'Protein.Group'
        protein_lengths:dict = None
        if 'Protein Length' in data_table.columns:
            protein_lengths = {}
            for _,row in data_table[[protein_col,'Protein Length']].drop_duplicates().iterrows():
                protein_lengths[row[protein_col]] = row['Protein Length']
        is_report: bool = False
        for c in data_table.columns:
            if c == 'Run':
                is_report = True
                break
        if is_report:
            table: pd.DataFrame = pd.pivot_table(
                data=data_table, index=protein_col, columns='Run', values='PG.MaxLFQ')
        else:
            data_cols: list = []
            for column in data_table.columns:
                col: list = column.split('.')
                if col[-1] == 'd':
                    data_cols.append(column)
            if len(data_cols) == 0:
                gather: bool = False
                for c in data_table.columns:
                    if gather:
                        data_cols.append(c)
                    elif c == 'First.Protein.Description':
                        gather = True
            #if len(data_cols) == 0:
                #data_cols = numerical_columns: set = set(data_table.select_dtypes(include=np.number).columns)

                
            table: pd.DataFrame = data_table[data_cols]
            table.index = data_table['Protein.Group']
        # Replace zeroes with missing values
        table.replace(0, np.nan, inplace=True)
        return [table, pd.DataFrame({'No data': ['No data']}), protein_lengths]

    def read_fragpipe(self, data_table: pd.DataFrame) -> pd.DataFrame:
        intensity_cols: list = []
        spc_cols: list = []
        uniq_intensity_cols: list = []
        uniq_spc_cols: list = []
        has_maxlfq: bool = False
        for column in data_table.columns:
            if 'Total' in column:
                continue
            if 'Combined' in column:
                continue
            if 'Intensity' in column:
                if 'maxlfq' in column.lower():
                    has_maxlfq = True
                if 'unique' in column.lower():
                    uniq_intensity_cols.append(column)
                else:
                    intensity_cols.append(column)
            elif 'Spectral Count' in column:
                if 'unique' in column.lower():
                    uniq_spc_cols.append(column)
                else:
                    spc_cols.append(column)
        if len(uniq_intensity_cols) > 0:
            intensity_cols = uniq_intensity_cols
        if len(uniq_spc_cols) > 0:
            spc_cols = uniq_spc_cols
        if has_maxlfq:
            intensity_cols = [i for i in intensity_cols if 'maxlfq' in i.lower()]
        protein_col: str = 'Protein ID'
        if 'Protein Length' in data_table.columns:
            protein_lengths: dict = {}
            for _,row in data_table[[protein_col,'Protein Length']].drop_duplicates().iterrows():
                protein_lengths[row[protein_col]] = row['Protein Length']
        else:
            protein_lengths = None
        table: pd.DataFrame = data_table
        # Replace zeroes with missing valuese
        table.replace(0, np.nan, inplace=True)
        table.index = table[protein_col]
        intensity_table: pd.DataFrame = table[intensity_cols]
        replace_str: str = ''
        if len(uniq_spc_cols) > 0:
            replace_str = 'Unique '
        spc_table: pd.DataFrame = table[spc_cols].rename(
            columns={ic: ic.replace(f'{replace_str}Spectral Count', '').strip()
                    for ic in spc_cols}
        )
        replace_str = ''
        if len(uniq_intensity_cols) > 0:
            replace_str = 'Unique '
        if intensity_table[intensity_cols[0:2]].sum().sum() == 0:
            intensity_table = pd.DataFrame({'No data': ['No data']})
        else:
            intensity_table.rename(
            columns={ic: ic.replace(f'{replace_str}Intensity', '').replace('MaxLFQ', '').strip()
                    for ic in intensity_cols},
            inplace=True)
        return (intensity_table, spc_table, protein_lengths)

    def read_matrix(self, data_table: pd.DataFrame, is_spc_table:bool=False, max_spc_ever:int=0) -> pd.DataFrame:
        """Reads a generic matrix into a data table. Either the returned SPC or intensity table is an empty dataframe.
        
        Matrix is assumed to be SPC matrix, if the maximum value is smaller than max_spc_ever.
        """
        with open(os.path.join('debug','read_matrix.txt'),'w') as fil:
            fil.write('Reading matrix')
            protein_id_column: str = 'Protein.Group'
            table: pd.DataFrame = data_table
            if protein_id_column not in table.columns:
                protein_id_column = table.columns[0]
            protein_lengths:dict = None
            protein_length_cols:list = ['PROTLEN','Protein Length','Protein.Length']
            protein_length_cols.extend([x.lower() for x in protein_length_cols])
            for plencol in protein_length_cols:
                if plencol in table.columns:
                    fil.write('pl:\t'+str(plencol) + '\n')
                    protein_lengths = {}
                    for _,row in table[[protein_id_column,plencol]].drop_duplicates().iterrows():
                        protein_lengths[row[protein_id_column]] = row[plencol]
                    table = table.drop(plencol)
                    break
            fil.write('ts:\t'+str(table.shape) + '\n')
            table.index = table[protein_id_column]
            table = table[table.index != 'na']
            for c in table.columns:
                isnumber: bool = np.issubdtype(table[c].dtype, np.number)
                fil.write('in:\t'+str(isnumber) + '\n')
                if not isnumber:
                    try:
                        table[c] = pd.to_numeric(table[c])
                    except ValueError:
                        continue
            # Replace zeroes with missing values
            table.replace(0, np.nan, inplace=True)
            table.drop(columns=[protein_id_column,],inplace=True)
            spc_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
            intensity_table: pd.DataFrame = pd.DataFrame({'No data': ['No data']})
            if is_spc_table:
                spc_table = table
            else:
                fil.write('sel:\t' + str(table.select_dtypes(include=[np.number]).columns)+ '\n')
                fil.write('tm:\t' + str(table.select_dtypes(include=[np.number]).max().max()) + '\n')
                if table.select_dtypes(include=[np.number]).max().max() <= max_spc_ever:
                    fil.write('is_spc' + '\n')
                    spc_table = table
                else:
                    fil.write('not_spc' + '\n')
                    intensity_table = table
            return (intensity_table, spc_table, protein_lengths)

    def run_saint(self, 
        data_table: pd.DataFrame,
        rev_sample_groups: dict,
        protein_lengths: dict,
        protein_names: dict,
        saint_command: str,
        control_table: pd.DataFrame = None,
        control_groups:set = None,
        output_directory:str = None) -> Tuple[pd.DataFrame,list]:
        """This function will run SAINTexpress analysis on the given data table and control sets.
        
        :param data_table: input data

        :return: tuple of (pd.DataFrame, list). Results are in the dataframe, and the list contains proteins for which no length could be found.        
        """

        if control_groups is None:
            control_groups: set = set()
        baitfile: list = []
        for column in data_table:
            groupname: str = rev_sample_groups[column]
            if groupname in control_groups:
                baitfile.append(f'{column}\t{groupname}\tC\n')
            else:
                baitfile.append(f'{column}\t{groupname}\tT\n')
        if control_table is not None:
            for col in control_table.columns:
                baitfile.append(f'{col}\tChosen_control\tC\n')
                rev_sample_groups[col] = 'Chosen_control'
        
        preyfile: list = []
        discarded: set = set()
        all_proteins: set = set(data_table.index.values)
        if control_table is not None:
            all_proteins |= set(control_table.index.values)
        for prey_protein in all_proteins:
            if prey_protein in protein_lengths:
                preyfile.append(f'{prey_protein}\t{protein_lengths[prey_protein]}\t{prey_protein}\n')
            else:
                discarded.add(prey_protein)
        
        intfile: pd.DataFrame
        intfile = data_table.reset_index().melt(id_vars='index').dropna()
        if control_table is not None:
            intfile = pd.concat([intfile, control_table.reset_index().melt(id_vars='index').dropna()])
        intfile.rename(columns={'variable': 'Bait', 'index': 'Prey', 'value': 'SPC'},inplace=True)
        intfile = intfile[~intfile['Prey'].isin(discarded)]
        intfile.loc[:,'BaitGroup'] = intfile.apply(lambda x: rev_sample_groups[x['Bait']],axis=1)

        saint_dir:str
        saint_cmd:str
        saint_dir,saint_cmd = saint_command
        temp_dir: str = os.path.join(saint_dir, f'{uuid.uuid4()}')
        if not os.path.isdir(temp_dir): 
            os.makedirs(temp_dir)

        saint_cmd = os.path.join('..',saint_cmd)

        intfile_name: str = os.path.join(temp_dir,'int.txt')
        baitfile_name: str = os.path.join(temp_dir,'bait.txt')
        preyfile_name: str = os.path.join(temp_dir, 'prey.txt')
        saint_output_file:str = os.path.join(temp_dir,'list.txt')

        intfile[['Bait','BaitGroup','Prey','SPC']].to_csv(intfile_name,index=False,header=False,sep='\t',encoding = 'utf-8')
        with open(baitfile_name,'w', encoding='utf-8') as fil:
            fil.write(''.join(baitfile))
        with open(preyfile_name,'w', encoding='utf-8') as fil:
            fil.write(''.join(preyfile))
        cmd: list = [saint_cmd,'int.txt','prey.txt','bait.txt']

        process: subprocess.CompletedProcess = subprocess.run(cmd, capture_output=True, check=False,cwd=temp_dir)

        output_dataframe: pd.DataFrame = pd.read_csv(saint_output_file,sep='\t')
        gene_names: list = []
        for _,row in output_dataframe.iterrows():
            name: str =row['PreyGene']
            if name in protein_names: name = protein_names[name]
            gene_names.append(name)
        output_dataframe['PreyGene'] = gene_names
        if output_directory is not None:
            output_report_file:str = os.path.join(output_directory, 'SAINT_output.txt')
            with open(output_report_file,'w',encoding='utf-8') as fil:
                fil.write(f'Subprocess exit code: {process.returncode}\n')
                fil.write(f'----------\nSubprocess output:\n{process.stdout.decode()}\n')
                fil.write(f'----------\nSubprocess errors:\n{process.stderr.decode()}\n')
            os.rename(intfile_name, os.path.join(output_directory,'interaction.txt'))
            os.rename(baitfile_name, os.path.join(output_directory,'bait.txt'))
            os.rename(preyfile_name, os.path.join(output_directory,'prey.txt'))
            os.rename(saint_output_file, os.path.join(output_directory,'saint_output_list.txt'))

        return (output_dataframe, sorted(list(discarded)))

    def rename_columns_and_update_expdesign(self, expdesign: pd.DataFrame, tables: list, discard_samples: list) -> Tuple[dict, dict]:
        # Get rid of file paths and timstof .d -file extension, if present:
        expdesign['Sample name'] = [
            oldvalue.rsplit('\\', maxsplit=1)[-1]\
                .rsplit('/', maxsplit=1)[-1]\
                .rstrip('.d') for oldvalue in expdesign['Sample name'].values
        ]
        expdesign: pd.DataFrame = expdesign[~expdesign['Sample name'].isin(discard_samples)]
        discarded_columns:list = []
        sample_groups: dict = {}
        sample_group_columns: dict = {}
        rev_intermediate_renaming: list = []
        for table_ind, table in enumerate(tables):
            intermediate_renaming: dict = {}
            rev_intermediate_renaming.append([])
            if len(table.columns) < 2:
                continue
            for column_name in table.columns:
                # Discard samples that are not named
                col: str = column_name
                if col not in expdesign['Sample name'].values:
                    # Try to see if the column without possible file path is in the expdesign:
                    col = col.rsplit(
                        '\\', maxsplit=1)[-1].rsplit('/', maxsplit=1)[-1]
                    if col not in expdesign['Sample name'].values:
                        col = col.rsplit('.d',maxsplit=1)[0]
                        if col not in expdesign['Sample name'].values:
                            # Discard column if not found
                            discarded_columns.append(col)
                            continue
                intermediate_renaming[column_name] = col
                sample_group: str = expdesign[expdesign['Sample name']
                                            == col].iloc[0]['Sample group']
                # If no value is available for sample in the expdesign
                # (but sample column name is there for some reason), discard column
                if pd.isna(sample_group):
                    continue
                newname: str = str(sample_group)
                # We expect replicates to not be specifically named; they will be named here.
                if newname[0].isdigit():
                    newname = f'Sample_{newname}'
                if newname not in sample_group_columns:
                    sample_group_columns[newname] = [[] for _ in range(len(tables))]
                sample_group_columns[newname][table_ind].append(col)
            if len(intermediate_renaming.keys()) > 0:
                table.rename(columns=intermediate_renaming, inplace = True)
            rev_intermediate_renaming[-1] = {value: key for key,value in intermediate_renaming.items()}
        column_renames: list = [{} for _ in range(len(tables))]
        used_columns: list = [{} for _ in range(len(tables))]
        with open(os.path.join('debug','rev_intermediate_renaming.json'),'w') as fil:
            json.dump(rev_intermediate_renaming, fil, indent=4)
        with open(os.path.join('debug','inprocess.txt'),'w') as fil:
            for nname, list_of_all_table_columns in sample_group_columns.items():
                first_len: int = 0
                fil.write('nn::\t' + nname + '\n')
                fil.write('latc::\t' + str(list_of_all_table_columns) + '\n')
                fil.write('------\n')
                for table_index, table_columns in enumerate(list_of_all_table_columns):
                    if len(table_columns) < 2:
                        continue
                    if first_len == 0:
                        first_len = len(table_columns)
                    else:
                        # Should have same number of columns/replicates for SPC and intensity tables
                        assert len(table_columns) == first_len
                    fil.write('ti::\t' + str(table_index) + '\n')
                    fil.write('tc::\t' + str(table_columns) + '\n')
                    fil.write('fl::\t' + str(first_len) + '\n')
                    for column_name in table_columns:
                        i: int = 1
                        while f'{nname}_Rep_{i}' in column_renames[table_index]:
                            i += 1
                        newname_to_use: str = f'{nname}_Rep_{i}'
                        if nname not in sample_groups:
                            sample_groups[nname] = set()
                        sample_groups[nname].add(newname_to_use)
                        fil.write('cn::\t' + str(column_name) + '\n')
                        fil.write('ntu::\t' + str(newname_to_use) + '\n')
                        fil.write('sg::\t' + str(sample_groups[nname]) + '\n')
                        fil.write('o::\t' + str(i) + '\n')
                        column_renames[table_index][newname_to_use] = column_name
                        fil.write('cr::\t' + str(column_renames[table_index][newname_to_use]) + '\n')
                        used_columns[table_index][newname_to_use] = rev_intermediate_renaming[table_index][column_name]
                fil.write('---\n')
            fil.write('======\n')
        sample_groups = {k: sorted(list(v)) for k, v in sample_groups.items()}   
        with open(os.path.join('debug','column_renames.json'),'w') as fil:
            json.dump(column_renames, fil, indent=4)
        with open(os.path.join('debug','sample_groups.json'),'w') as fil:
            json.dump(sample_groups, fil, indent=4)
        with open(os.path.join('debug','used_columns.json'),'w') as fil:
            json.dump(used_columns, fil, indent=4)
        for table_index, table in enumerate(tables):
            rename_columns: dict = {value: key for key, value in column_renames[table_index].items()}
            table.drop(
                columns= [c for c in table.columns if c not in rename_columns],
                inplace=True
                )
            table.rename(columns=rename_columns, inplace=True)

        rev_sample_groups: dict = {}
        for group, samples in sample_groups.items():
            for sample in samples:
                rev_sample_groups[sample] = group
        return (sample_groups, rev_sample_groups, discarded_columns, used_columns)

    def remove_duplicate_protein_groups(self, data_table: pd.DataFrame) -> pd.DataFrame:
        aggfuncs: dict = {}
        numerical_columns: set = set(data_table.select_dtypes(include=np.number).columns)
        for column in data_table.columns:
            if column in numerical_columns:
                aggfuncs[column] = sum
            else:
                aggfuncs[column] = 'first'
        return data_table.groupby(data_table.index).agg(aggfuncs).replace(0,np.nan)

    def parse_data(self, data_content, data_name, expdes_content, expdes_name, max_theoretical_spc: int=0, discard_samples: list=None) -> list:
        table: pd.DataFrame = self.read_df_from_content(data_content, data_name)
        expdesign: pd.DataFrame = self.read_df_from_content(expdes_content, expdes_name)
        read_funcs: dict[tuple[str, str]] = {
            ('DIA', 'DIA-NN'): self.read_dia_nn,
            ('DDA', 'FragPipe'): self.read_fragpipe,
            ('DDA/DIA', 'Unknown'): self.read_matrix,

        }

        data_type: tuple = None
        keyword_args: dict = {}
        if 'Protein.Ids' in table.columns:
            if 'First.Protein.Description' in table.columns:
                data_type = ('DIA', 'DIA-NN')
        elif 'Top Peptide Probability' in table.columns:
            if 'Protein Existence' in table.columns:
                data_type = ('DDA', 'FragPipe')
        if data_type is None:
            data_type = ('DDA/DIA', 'Unknown')
            keyword_args['max_spc_ever'] = max_theoretical_spc
        with open('datatype','w') as fil:
            fil.write(str(data_type))
        intensity_table: pd.DataFrame
        spc_table: pd.DataFrame
        protein_length_dict: dict
        intensity_table, spc_table, protein_length_dict = read_funcs[data_type](table, **keyword_args)
        intensity_table = self.remove_duplicate_protein_groups(intensity_table)
        spc_table = self.remove_duplicate_protein_groups(spc_table)

        sample_groups: dict
        rev_sample_groups: dict
        discarded_columns: list
        used_columns: list
        
        if discard_samples is None:
            discard_samples = []
        sample_groups, rev_sample_groups, discarded_columns, used_columns = self.rename_columns_and_update_expdesign(
            expdesign,
            [intensity_table, spc_table],
            discard_samples
        )
        spc_table = spc_table[sorted(list(spc_table.columns))]

        if len(intensity_table.columns) > 1:
            intensity_table = intensity_table[sorted(list(intensity_table.columns))]
            untransformed_intensity_table: pd.DataFrame = intensity_table
            intensity_table = intensity_table.apply(np.log2)
        else:
            untransformed_intensity_table = intensity_table
        bait_uniprots: dict = {}
        for _,row in expdesign.iterrows():
            bval: str = ''
            try:
                bval: str = str(row['Bait uniprot'])
            except KeyError:
                bval = ''
            if (len(bval) == 0) or (bval == 'nan'):
                bval = 'No bait uniprot'
            bait_uniprots[row['Sample group']] = bval
        return_dict: dict = {
            'sample groups': {
                'norm': sample_groups,
                'rev': rev_sample_groups
            }, 
            'data tables': {
                'raw intensity': untransformed_intensity_table.to_json(orient='split'),
                'spc': spc_table.to_json(orient='split'),
                'intensity': intensity_table.to_json(orient='split'),
                'experimental design': expdesign.to_json(orient='split')
            }, 
            'info': {
                'discarded columns': discarded_columns,
                'used columns': used_columns,
                'data type': data_type,
                'discarded samples': discard_samples
            },
            'other': {
                'protein lengths': protein_length_dict,
                'bait uniprots': bait_uniprots,
            }
        }
        if len(intensity_table.columns) < 2:
            return_dict['data tables']['main table'] = return_dict['data tables']['spc']
            return_dict['info']['values'] = 'SPC'
            return_dict['other']['all proteins'] =  list(spc_table.index)
        else:
            return_dict['data tables']['main table'] = return_dict['data tables']['intensity']
            return_dict['info']['values'] = 'intensity'
            return_dict['other']['all proteins'] =  list(intensity_table.index)
        
        
        return_dict['sample groups']['guessed control samples'] = self.guess_controls(sample_groups)

        return return_dict

    def guess_controls(self, sample_groups: dict) -> Tuple[list,list]:
        rl: list = []
        rl_samples: list = []
        for group_name, samples in sample_groups.items():
            if 'gfp' in group_name.lower():
                rl.append(group_name)
                rl_samples.append(samples)
            
        return (rl, rl_samples)

    def get_count_data(self, data_table) -> pd.DataFrame:
        data: pd.DataFrame = data_table.\
            notna().sum().\
            to_frame(name='Protein count')
        data.index.name = 'Sample name'
        return data


    def get_sum_data(self, data_table) -> pd.DataFrame:
        data: pd.DataFrame = data_table.sum().\
            to_frame(name='Value sum')
        data.index.name = 'Sample name'
        return data


    def get_avg_data(self, data_table) -> pd.DataFrame:
        data: pd.DataFrame = data_table.mean().\
            to_frame(name='Value mean')
        data.index.name = 'Sample name'
        return data


    def get_na_data(self, data_table) -> pd.DataFrame:
        data: pd.DataFrame = ((data_table.
                            isna().sum() / data_table.shape[0]) * 100).\
            to_frame(name='Missing value %')
        data.index.name = 'Sample name'
        return data
