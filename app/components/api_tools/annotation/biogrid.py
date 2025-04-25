import sys
import os
from zipfile import ZipFile
from datetime import datetime
import pandas as pd
import numpy as np
from requests import get
from urllib.request import urlretrieve
from itertools import chain

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

def deduplicate_str_dataframe_by_index(df) -> pd.DataFrame:
    def merge_row_group(group):
        merged_row = {}
        for col in group.columns:
            split_vals = group[col].dropna().astype(str).str.split(';')
            all_vals = list(chain.from_iterable(split_vals))
            deduped = sorted(set(all_vals))
            merged_row[col] = ';'.join(deduped).strip(';') if deduped else None
        return pd.Series(merged_row)

    return df.groupby(df.index).apply(merge_row_group)
def get_upid(upcol: pd.Series) -> pd.Series:
    news = []
    for v in upcol:
        for nullchar in ['-1','nan','none','-','0']:
            v = v.replace(nullchar,'')
        news.append(v)
    retser = pd.Series(news, index=upcol.index, name=upcol.name)
    retser.replace('', np.nan, inplace=True)
    return retser

def filter_chunk(df: pd.DataFrame, uniprots_to_get:set|None, organisms: set|None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = df[df['Experimental System Type']=='physical'][['Experimental System', 'Experimental System Type', 'Publication Source', 'Organism ID Interactor A', 
            'Organism ID Interactor B', 'Throughput', 'Score', 'Modification', 'Qualifications', 
            'Tags', 'Source Database', 'SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B', 
            'Ontology Term IDs', 'Ontology Term Names', 'Ontology Term Categories', 'Ontology Term Qualifier IDs',
            'Ontology Term Qualifier Names', 'Ontology Term Types']].drop_duplicates()
    df = df[df['Experimental System Type']=='physical']
    for c in ['SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B']:
        df = df[df[c].notna()]
    df = df[df['SWISS-PROT Accessions Interactor A'] != df['SWISS-PROT Accessions Interactor B']]
    not_used = [
        'Co-localization','Co-fractionation'
    ]
    if organisms:
        organisms = {str(o) for o in organisms}
        df = df[(df['Organism ID Interactor A'].astype(str).isin(organisms)) | (df['Organism ID Interactor B'].astype(str).isin(organisms))]
    df = df[~df['Experimental System'].isin(not_used)]
    swcols = [c for c in df.columns if 'Interactor ' in c]
    normcols = [c for c in df.columns if c not in swcols]
    renames = {c: c.lower().replace(' ','_') for c in normcols+swcols}
    renames.update({
        'Source Database': 'source_database',
        'Experimental System Type': 'interaction_type',
        'Experimental System': 'interaction_detection_method',
        'Publication Source': 'publication_identifier',
        'Organism ID Interactor A': 'organism_interactor_a',
        'Organism ID Interactor B': 'organism_interactor_b',
        'Modification': 'modification',
        'Score': 'confidence_value'
    })
    normcols = [renames[c] for c in normcols if c != 'Source Database']
    swcols = [renames[c] for c in swcols if not 'SWISS-PROT' in c]
    df = df.rename(columns=renames)
    df['uniprot_id_a'] = get_upid(df['swiss-prot_accessions_interactor_a'])
    df['uniprot_id_b'] = get_upid(df['swiss-prot_accessions_interactor_b'])
    df = df[df['uniprot_id_a'].notna() & df['uniprot_id_b'].notna()]
    if uniprots_to_get:
        df = df[df['uniprot_id_a'].isin(uniprots_to_get) | df['uniprot_id_b'].isin(uniprots_to_get)]
    df['uniprot_id_a_noiso'] = [c.split('-')[0] for c in df['uniprot_id_a']]
    df['uniprot_id_b_noiso'] = [c.split('-')[0] for c in df['uniprot_id_b']]
    df['isoform_a'] = df['uniprot_id_a']
    df['isoform_b'] = df['uniprot_id_b']
    df['biological_role_interactor_a'] = ''
    df['annotation_interactor_a'] = ''
    df['biological_role_interactor_b'] = ''
    df['annotation_interactor_b'] = ''
    df['organism_interactor_a'] = df['organism_interactor_a'].astype(str)
    df['organism_interactor_b'] = df['organism_interactor_b'].astype(str)
    if df.shape[0] == 0:
        for c in df.columns:
            df[f'{c}_processed'] = ''
    # Pre-process normcols data using vectorized operations
    else:
        for c in df.columns:
            try:
                df[f'{c}_processed'] = (
                    df[c].str.split('|')
                    .apply(lambda x: [v for val in x for v in val.split('__')])  # flatten
                    .apply(lambda vals: [
                        v.strip().lower().replace('-', '').replace(';', '').replace('_', '')
                        for v in vals
                        if v.strip().lower() not in ('', '0', 'nan', 'none')
                    ])
                    .str.join(';')
                )
            except AttributeError:
                df[f'{c}_processed'] = df[c]
    nc = []
    for _,row in df.iterrows():
        nc.append(f'Experiment throughput from BioGRID: {";".join(list(set(row["throughput"].split(";"))))}')
        if pd.notna(row["qualifications"]):
            if len(row["qualifications"].strip().strip(";"))>0:
                nc[-1] += f'+Q:{row["qualifications"].strip().strip(";")}'
        if pd.notna(row['modification']):
            nc[-1] += f'+Mod:{row["modification"]}'
    df['notes'] = nc
    df['update_time'] = 'BioGRID:'+str(datetime.today()).split()[0]
    df.drop(columns=[
        'throughput', 
        'modification', 
        'qualifications', 
        'tags', 
        'ontology_term_ids',
        'ontology_term_names',
        'ontology_term_categories',
        'ontology_term_qualifier_ids',
        'ontology_term_qualifier_names',
        'ontology_term_types'
    ],inplace=True)
    df.drop_duplicates(inplace=True)
    return df, normcols, swcols

def get_final_df(chunk_df: pd.DataFrame, previous: pd.DataFrame|None = None,time_format: str = '%Y/%m/%d') -> pd.DataFrame:
    datarows = {}
    for intname, row in chunk_df.iterrows():
        datarows.setdefault(intname, {v: set() for v in chunk_df.columns})
        for c in chunk_df.columns:
            datarows[intname][c]|= set(str(row[c]).split(';'))
    findf = pd.DataFrame.from_dict({ind: {key: ';'.join(sorted(list(val))).strip(';') for key, val in ind_dict.items()} for ind, ind_dict in datarows.items()},orient='index')
    findf.index.name = 'interaction'
    today_str = datetime.today().strftime(time_format)
    cols = list(findf.columns)
    findf["biogrid_creation_date"] = today_str
    findf["biogrid_updated_date"] = today_str

    # Since biogrid does not supply the dates, we have to do it by ourselves.
    if (previous is not None) and all(column in previous.columns for column in ['biogrid_creation_date', 'biogrid_updated_date']):
        common_idx = findf.index.intersection(previous.index)
        # Compare only matching rows
        matches = (
            findf.loc[common_idx, cols].astype(str)
            == previous.loc[common_idx, cols].astype(str)
        ).all(axis=1)

        # For matching rows, assign df1['date']
        findf.loc[common_idx, 'biogrid_creation_date'] = previous.loc[common_idx, 'biogrid_creation_date']
        findf.loc[common_idx[matches], 'biogrid_updated_date'] = previous.loc[common_idx[matches], 'biogrid_updated_date']

    return findf

def handle_and_split_save(df: pd.DataFrame, folder_path: str, normcols: list[str], swcols: list[str]) -> None:
    swcols.extend(['biological_role_interactor_a',
    'annotation_interactor_a',
    'biological_role_interactor_b',
    'annotation_interactor_b'])
    swcols.sort(key = lambda x: x[-1])
    dfcols = [
        'uniprot_id_a',
        'uniprot_id_b',
        'uniprot_id_a_noiso',
        'uniprot_id_b_noiso',
        'isoform_a',
        'isoform_b',
    ]  + normcols + swcols + ['source_database', 'notes', 'update_time']

    datarows = {}
    for row in df.itertuples(index=False):
        new_rows = [ 
            [getattr(row, col) for col in [
                f'uniprot_id_{n1}',
                f'uniprot_id_{n2}',
                f'uniprot_id_{n1}_noiso',
                f'uniprot_id_{n2}_noiso',
                f'isoform_{n1}',
                f'isoform_{n2}',   
                ]
            ] + [
                getattr(row,f'{c}_processed') for c in normcols
            ] + [
                getattr(row,f'{c[:-1]}{n1}_processed') for c in swcols[:3]
            ] + [
                getattr(row,f'{c[:-1]}{n2}_processed') for c in swcols[:3]
            ] + [
                getattr(row,n) for n in ['source_database','notes','update_time']
            ]
            for n1, n2 in [('a', 'b'), ('b', 'a')]
        ]
        for n in new_rows:
            if n[1] == '-':
                continue
            index = f'{n[0]}_-_{n[1]}'
            keys = dfcols
            datarows.setdefault(index, {v: set() for v in keys})
            for i, k in enumerate(keys):
                if isinstance(n[i], (list, set)):
                    datarows[index][k]|=set(n[i])
                elif isinstance(n[i],str):
                    datarows[index][k]|= set(n[i].split(';'))
                else:
                    datarows[index][k]|={n[i]}
    
    # Create DataFrame in one go
    findf = pd.DataFrame.from_dict({ind: {key: ';'.join(sorted([str(x) for x in val])).strip(';') for key, val in ind_dict.items()} for ind, ind_dict in datarows.items()},orient='index')
    findf['publication_identifier'] = findf['publication_identifier'].str.lower()
    findf.index.name = 'interaction'
    split_and_save_by_prefix(findf.reset_index(), 'interaction', 3, folder_path)

def split_and_save_by_prefix(df: pd.DataFrame, column: str, num_chars: int, output_dir: str, index: bool = False, sep: str = '\t') -> None:
    os.makedirs(output_dir, exist_ok=True)
    for (prefix, result) in df.groupby(df[column].str[:num_chars]):
        filename = os.path.join(output_dir, f"{prefix}.tsv")
        write_header = not os.path.exists(filename)
        result.to_csv(filename, mode='a', header=write_header, index=index, sep=sep)

# super inefficient, but run rarely, so good enough.
def generate_pandas(file_path:str, uniprots_to_get:set|None, organisms: set|None = None, current_version: str|None = None) -> None:
    """
    Inefficiently generates a pandas dataframe from a given biogrid file (downloaded  by update()) and writes it to a .tsv file with the same name as input file path.

    :param file_path: path to the downloaded .tab3 file
    :param current_version: current version of the biogrid file
    :param uniprots_to_get: set of which uniprots should be included in the written .tsv file. If None, all uniprots will be included.
    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    """
    folder_path: str = file_path.replace('.txt','')
    if not current_version:
        current_version = ''
    for chunk in pd.read_csv(file_path,sep='\t', chunksize=10000):
        chunk, normcols, swcols = filter_chunk(chunk, uniprots_to_get, organisms)
        if chunk.shape[0] > 0:
            #handle_and_split_save(chunk, temp_dir)
            handle_and_split_save(chunk, folder_path, normcols, swcols)
    
    for fname in os.listdir(folder_path):
        if fname.endswith('.tsv'):
            previous = None
            if os.path.exists(os.path.join(current_version, fname)):
                previous = pd.read_csv(os.path.join(current_version, fname),sep='\t', index_col='interaction')
            chunk_df = pd.read_csv(os.path.join(folder_path, fname),sep='\t', index_col='interaction')
            findf: pd.DataFrame = get_final_df(chunk_df, previous = previous)
            findf.to_csv(os.path.join(folder_path, fname), index=True, sep='\t')

def do_update(save_dir:str, save_zipname: str, latest_zip_url: str, uniprots_to_get:set|None, organisms: set|None = None, current_version: str|None = None) -> None:
    """
    Handles practicalities of updating the biogrid tsv file on disk

    :param save_dir: directory where the datafiles will be put
    :param save_zipname: filename for the zipfile that will be downloaded
    :param latest_zip_url: url for the zip to download from BioGRID
    :param uniprots_to_get: a set of which uniprots should be retained. If None, all uniprots will be retained.
    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    """
    urlretrieve(latest_zip_url,os.path.join(save_dir, save_zipname))
    datafile_path = os.path.join(save_dir, save_zipname.replace('.zip','.txt'))
    if not os.path.exists(datafile_path):
        with ZipFile(os.path.join(save_dir, save_zipname), 'r') as zip_ref:
            zip_ref.extractall(save_dir)
    generate_pandas(datafile_path, uniprots_to_get, organisms, current_version)
    os.remove(os.path.join(save_dir, save_zipname))
    os.remove(datafile_path)

def read_file_chunks(filepath: str, organisms: set|None = None, since_date:str|None = None) -> pd.DataFrame:
    iter_csv = pd.read_csv(filepath, iterator=True, sep='\t', chunksize=1000)
    df_parts = []
    for chunk in iter_csv:
        if organisms:
            chunk = chunk.loc[chunk['organism_interactor_a'].isin(organisms) | chunk['organism_interactor_b'].isin(organisms)]
        df_parts.append(chunk)
    return pd.concat(df_parts)

def read_folder_chunks(folderpath: str, organisms: set|None = None, subset_letter: str|None = None, since_date:str|None = None) -> pd.DataFrame:
    df_parts = []
    for file in os.listdir(folderpath):
        if subset_letter and not file.startswith(subset_letter):
            continue
        df_parts.append(read_file_chunks(os.path.join(folderpath, file), organisms, since_date))
    if len(df_parts) > 0:
        return pd.concat(df_parts)
    else:
        heads = pd.read_csv(os.path.join(folderpath, file), nrows=1,sep='\t')
        return pd.DataFrame(columns=heads.columns)

# TODO: implement since_date
def get_latest(organisms: set|None = None, subset_letter: str|None = None, name_only: bool = False, since_date:str|None = None) -> pd.DataFrame|str :
    """
    Fetches the latest data from disk

    :param name_only: if True, returns the filepath of the latest file instead of the dataframe.
    :returns: Pandas dataframe of the latest BioGRID data.
    """
    current_version: str = apitools.get_newest_file(apitools.get_save_location('BioGRID'))
    filepath: str = os.path.join(apitools.get_save_location('BioGRID'), current_version)
    if name_only:
        return filepath
    if not os.path.exists(filepath):
        return pd.DataFrame()
    else:
        return read_folder_chunks(filepath, organisms, subset_letter, since_date)

def get_available() -> list[str]:
    filepath: str = get_latest(name_only=True) # type: ignore
    return [f.split('.')[0] for f in os.listdir(filepath) if f.endswith('.tsv')]


#TODO: check uniprots in should_update bool check too, not just version. Also organisms should be checked too.
def update(uniprots_to_get:set|None = None, organisms: set|None = None) -> None:
    """
    Updates the database, if required

    :param uniprots_to_get: uniprots to retain in the database. if None, all uniprots will be retained.
    :param organisms: organisms to filter the data by. This set should contain the organism IDs as strings. If None, all data will be included.
    """
    url = 'https://downloads.thebiogrid.org/BioGRID/Release-Archive'
    r = get(url).text.split('\n')
    r = [rr.strip().split('href=')[1].split('\' title')[0].strip().strip('\'') for rr in r if 'https://downloads.thebiogrid.org/BioGRID/Release-Archive/' in rr]
    latest = sorted(r,reverse=True)[0]
    latest_zipname = f'{latest.replace(".org/",".org/Download/")}{latest.rsplit("/",maxsplit=2)[-2].replace("BIOGRID","BIOGRID-ALL")}.tab3.zip'
    uzip = latest_zipname.rsplit('/',maxsplit=1)[1]
    save_location:str = apitools.get_save_location('BioGRID')
    current_file = apitools.get_newest_file(save_location)
    if os.path.exists(os.path.join(save_location, current_file)):
        should_update = uzip.rsplit('.',maxsplit=1)[0] != current_file
    else:
        should_update = True
    if should_update:
        do_update(save_location, uzip, latest_zipname, uniprots_to_get, organisms, current_file)
        
def get_version_info() -> str:
    """
    Returns version info for the newest available biogrid version.
    """
    nfile: str = apitools.get_newest_file(apitools.get_save_location('BioGRID'))
    return f'Downloaded ({nfile.split("_")[0]})'

def get_method_annotation() -> dict:
    """
    Returns information regarding annotation for interaction identification methods used in BioGRID
    """
    legend = {
        'Affinity Capture-Luminescence': r'An interaction is inferred when a bait protein, tagged with luciferase, is enzymatically detected in immunoprecipitates of the prey protein as light emission. The prey protein is affinity captured from cell extracts by either polyclonal antibody or epitope tag.', 
        'Affinity Capture-MS': r'An interaction is inferred when a bait protein is affinity captured from cell extracts by either polyclonal antibody or epitope tag and the associated interaction partner is identified by mass spectrometric methods. Note that this in an in vivo experiment where all relevant proteins are co-expressed in the cell (e.g. PMID: 12150911).', 
        'Affinity Capture-RNA': r'An interaction is inferred when a bait protein is affinity captured from cell extracts by either polyclonal antibody or epitope tag and the associated RNA species is identified by Northern blot, RT-PCR, affinity labeling, sequencing, or microarray analysis. Note that this is an in vivo experiment where all relevant interactors are co-expressed in the cell (e.g. PMID: 10747033). If the protein-RNA interaction is detected in vitro, use “Protein-RNA” instead.', 
        'Affinity Capture-Western': r'An interaction is inferred when a bait protein is affinity captured from cell extracts by either polyclonal antibody or epitope tag and the associated interaction partner is identified by Western blot with a specific polyclonal antibody or second epitope tag (e.g. PMID: 11782448, Fig. 2). This category is also used if an interacting protein is visualized directly by dye stain or radioactivity. Note that this is an in vivo experiment where all relevant proteins are co-expressed in the cell. If the proteins are shown to interact outside of a cellular environment (such as lysates exposed to a bait protein for pull down) this should be considered in vitro and Reconstituted Complex should be used. This also differs from any co-purification experiment involving affinity capture in that the co-purification experiment involves at least one extra purification step to get rid of potential contaminating proteins.', 
        'Co-fractionation': r'An interaction is inferred from the presence of two or more protein subunits in a partially purified protein preparation (e.g. PMID: 11294905, Fig. 9). If co-fractionation is demonstrated between 3 or more proteins, then add them as a complex.', 
        'Co-localization': r'An interaction is inferred from two proteins that co-localize in the cell by indirect immunofluorescence only when in addition, if one gene is deleted, the other protein becomes mis-localized. This also includes co-dependent association of proteins with promoter DNA in chromatin immunoprecipitation experiments (write “ChIP” in qualification text box), and in situ proximity ligation assays (write “PLA” in qualification text box).', 
        'Proximity Label-MS': r'An interaction is inferred when a bait-enzyme fusion protein selectively modifies a vicinal protein with a diffusible reactive product, followed by affinity capture of the modified protein and identification by mass spectrometric methods, such as the BioID system PMID: 24255178. This system should not be used for in situ proximity ligation assays in which the interaction is measured by fluorescence, eg. PMID: 25168242, which should be captured as co-localization.', 
        'Co-purification': r'An interaction is inferred from the identification of two or more protein subunits in a purified protein complex, as obtained by several classical biochemical fractionation steps, or else by affinity purification and one or more additional fractionation steps. Note that a Western or mass-spec may also be used to identify the subunits, but that this differs from “Affinity Capture-Western” or “Affinity Capture-Mass Spec” because it involves at least one extra purification step to get rid of contaminants (e.g. PMID: 19343713). Typically, TAP-tag experiments are considered to be affinity captures and not co-purification experiments. If there is no obvious bait-hit directionality to the interaction, then the co-purifying proteins should be listed as a complex. If only co-fractionation is demonstrated, i.e. if the interaction is inferred from the presence of two or more protein subunits in a partially purified protein preparation (e.g. PMID: 11294905, Fig. 9), then use “Co-fractionation” instead.', 
        'FRET': r'An interaction is inferred when close proximity of interaction partners is detected by fluorescence resonance energy transfer between pairs of fluorophore-labeled molecules, such as occurs between CFP (donor) and YFP (acceptor) fusion proteins in vivo (e.g. PMID: 11950888, Fig. 4).', 
        'PCA': r'An interaction is inferred through the use of a Protein-Fragment Complementation Assay (PCA) in which a bait protein is expressed as a fusion to either an N- or C- terminal peptide fragment of a reporter protein and a prey protein is expressed as a fusion to the complementary C- or N- terminal fragment, respectively, of the same reporter protein. Interaction of bait and prey proteins bring together complementary fragments, which can then fold into an active reporter, e.g. the split-ubiquitin assay (e.g. PMID: 12134063, Figs. 1,2), bimolecular fluorescent complementation (BiFC). More examples of PCAs are discussed in this paper.', 
        'Two-hybrid': r'An interaction is inferred when a bait protein is expressed as a DNA binding domain (DBD) fusion, a prey protein is expressed as a transcriptional activation domain (TAD) fusion and the interaction is measured by reporter gene activation (e.g. PMID: 9082982, Table 1).', 
        'Biochemical Activity': r'An interaction is inferred from the biochemical effect of one protein upon another in vitro, for example, GTP-GDP exchange activity or phosphorylation of a substrate by a kinase (e.g. PMID: 9452439, Fig. 2). The “bait” protein executes the activity on the substrate “hit” protein. A Modification value is recorded for interactions of this type with the possible values Phosphorylation, Ubiquitination, Sumoylation, Dephosphorylation, Methylation, Prenylation, Acetylation, Deubiquitination, Proteolytic Processing, Glucosylation, Nedd(Rub1)ylation, Deacetylation, No Modification, Demethylation.', 
        'Co-crystal Structure': r'An interaction is directly demonstrated at the atomic level by X-ray crystallography (e.g. PMID: 12660736). This category should also be used for NMR or Electron Microscopy (EM) structures, and for each of these cases, a note should be added indicating that it\'s an NMR or EM structure. If there is no obvious bait-hit directionality to the interaction involving 3 or more proteins, then the co-crystallized proteins should be listed as a complex.', 
        'Far Western': r'An interaction is inferred when a bait protein is immobilized on a membrane and a prey protein that is incubated with the membrane localizes to the same membrane position as the bait protein. The prey protein could be provided as a purified protein probe (e.g. PMID: 12857883, Fig. 7).', 
        'Protein-peptide': r'An interaction is inferred between a protein and a peptide derived from an interaction partner. A variety of techniques could be employed including phage display experiments (e.g. PMID: 12706896). Depending on the experimental details, either the protein or the peptide could be the “bait”.', 
        'Protein-RNA': r'An interaction is inferred using a variety of techniques between a protein and an RNA in vitro. By way of contrast, note that “Affinity Capture-RNA” involves protein and RNA that are co-expressed in vivo.', 
        'Reconstituted Complex': r'An interaction is inferred between proteins in vitro. This can include proteins in recombinant form or proteins isolated directly from cells with recombinant or purified bait. For example, GST pull-down assays where a GST-tagged protein is first isolated and then used to fish interactors from cell lysates are considered reconstituted complexes (e.g. PMID: 14657240, Fig. 4A or PMID: 14761940, Fig. 5). This can also include gel-shifts and surface plasmon resonance experiments. The bait-hit directionality may not be clear for 2 interacting proteins. In these cases the directionality is up to the discretion of the curator.Direction of Interactions (Bait/Hit). If there is no obvious bait-hit directionality to an interaction involving 3 or more proteins, then the proteins in the reconstituted complex should be entered as a complex. ', 
    }
    return legend

def methods_text() -> str:
    """
    Generates a methods text for used biogrid data
    
    :returns: a tuple of (readable reference information (str), PMID (str), biogrid description (str))
    """
    short,long,pmid = apitools.get_pub_ref('BioGRID')
    return '\n'.join([
        'BioGRID',
        f'Interactions were mapped with BioGRID (https://thebiogrid.org) {short}',
        f'{get_version_info()}',
        pmid,
        long
    ])

# TODO: this should not be set here.
odir = os.path.join('components','api_tools','annotation')
    