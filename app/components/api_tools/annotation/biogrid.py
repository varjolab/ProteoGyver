import sys
import os
from zipfile import ZipFile
from datetime import datetime
import pandas as pd
import numpy as np
from requests import get
from urllib.request import urlretrieve

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools

# super inefficient, but run rarely, so good enough.
def generate_pandas(file_path:str, uniprots_to_get:set) -> None:
    """
    Inefficiently generates a pandas dataframe from a given biogrid file (downloaded  by update()) and writes it to a .tsv file with the same name as input file path.

    :param file_path: path to the downloaded .tab3 file
    :param uniprots_to_get: set of which uniprots should be included in the written .tsv file
    """
    newpath:str = file_path.replace('.txt','.tsv')
    os.rename(file_path, newpath)
    df = pd.read_csv(newpath,sep='\t', low_memory=False)
    df = df.replace('-',np.nan)
    df = df[['Experimental System', 'Experimental System Type', 'Publication Source', 'Organism ID Interactor A', 
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
    for c in df.columns:
        df[c] = [str(v).replace(';',',') for v in df[c].values]

    df = df[~df['Experimental System'].isin(not_used)]
    df = df.dropna(how='all',axis=1)
    #df['Experimental system details'] = [legend[x] for x in df['Experimental System'].values]
    df['Score'] = df['Score'].astype(float)
    df = df.drop(columns=[c for c in df.columns if 'Ontology Term' in c])
    vals_needed = ['SWISS-PROT Accessions Interactor A', 'SWISS-PROT Accessions Interactor B']
    cols = [c for c in df.columns if c not in vals_needed]
    ndata = {}
    for _, row in df.iterrows():
        ids_a = [row['SWISS-PROT Accessions Interactor A']]
        ids_b = [row['SWISS-PROT Accessions Interactor B']]
        ids_a = [i for i in ids_a if i in uniprots_to_get]
        ids_b = [i for i in ids_b if i in uniprots_to_get]
        if len(ids_a) == 0:continue
        if len(ids_b) == 0:continue
        for i in ids_a:
            if i not in ndata:
                ndata[i] = {}
            for j in ids_b:
                if j not in ndata[i]:
                    ndata[i][j] = {c: set() for c in cols}
                for c in cols:
                    for v in str(row[c]).split('|'):
                        ndata[i][j][c].add(v)
        for i in ids_b:
            if i not in ndata:
                ndata[i] = {}
            for j in ids_a:
                if j not in ndata[i]:
                    ndata[i][j] = {c: set() for c in cols}
                for c in cols:
                    flipped_c = c.replace(' A',' B')
                    if c.endswith(' B'):
                        flipped_c = c.replace(' B',' A')
                    for v in str(row[c]).split('|'):
                        ndata[i][j][flipped_c].add(v)
    dodata = []
    for id1, idi in ndata.items():
        if '-' in id1:
            iso1: str = id1
            id1: str = id1.split('-')[0]
        else:
            iso1 = ''
        for id2, cdic in idi.items():
            if '-' in id2:
                iso2: str = id2
                id2: str = id2.split('-')[0]
            else:
                iso2 = ''
            dodata.append([id1,id2,iso1,iso2])
            for c in cols:
                cval = []
                for val in cdic[c]:
                    try:
                        cval.extend(val.split('|'))
                    except AttributeError:
                        cval.append(val)
                try:
                    dodata[-1].append(';'.join(cval))
                except TypeError:
                    vs = set(cval)
                    dodata[-1].append(list(vs)[0])
    docols = ['uniprot_id_a', 'uniprot_id_b', 'isoform_a','isoform_b'] + cols
    findf = pd.DataFrame(data=dodata, columns=docols)
    findf = findf.rename(columns = {
        'UPID A': 'uniprot_id_a',
        'UPID B': 'uniprot_id_b',
        'Experimental system': 'BioGRID experiment',
        'Source Database': 'source_database',
        'Experimental System Type': 'interaction_type',
        'Experimental System': 'interaction_detection_method',
        'Publication Source': 'publication_identifier',
        'Score': 'confidence_value'
    })
    nc = []
    id1c = []
    id2c = []
    for _,row in findf.iterrows():
        nc.append(f'Experiment throughput from BioGRID: {";".join(list(set(row["Throughput"].split(";"))))}')
        if pd.notna(row["Qualifications"]):
            if len(row["Qualifications"].strip().strip(";"))>0:
                nc[-1] += f'+Q:{row["Qualifications"].strip().strip(";")}'
        if pd.notna(row['Modification']):
            nc[-1] += f'+Mod:{row["Modification"]}'
        id1 = row['uniprot_id_a']
        id2 = row['uniprot_id_b']
        if '-' in id1:
            id1: str = id1.split('-')[0]
        if '-' in id2:
            id2: str = id2.split('-')[0]
        id1c.append(id1)
        id2c.append(id2)
    findf['notes'] = nc
    findf['update_time'] = 'BioGRID:'+str(datetime.today()).split()[0]
    findf['uniprot_id_a_noiso'] = id1c
    findf['uniprot_id_b_noiso'] = id2c
    findf['interaction'] = findf['uniprot_id_a'] + '_-_' + findf['uniprot_id_b']
    findf['experimental_role_interactor_a'] = np.nan
    findf['biological_role_interactor_a'] = np.nan
    findf['annotation_interactor_a'] = np.nan
    findf['experimental_role_interactor_b'] = np.nan
    findf['biological_role_interactor_b'] = np.nan
    findf['annotation_interactor_b'] = np.nan

    findf = findf[[
        'interaction','uniprot_id_a', 'uniprot_id_b', 'uniprot_id_a_noiso', 'uniprot_id_b_noiso',
        'isoform_a', 'isoform_b', 'publication_identifier', 
        'interaction_detection_method', 'interaction_type', 'confidence_value',
        'source_database', 'experimental_role_interactor_a','experimental_role_interactor_b',
        'biological_role_interactor_a', 'biological_role_interactor_b','annotation_interactor_a',
        'annotation_interactor_b', 'notes','update_time'
    ]]
    for c in findf.columns:
        for repchar in ['|','__']:
            tmp = [v for v in findf[c].values if repchar in str(v)]
            if len(tmp)>0:
                findf[c] = [str(v).replace(repchar,';') for v in findf[c].values]
    for c in findf.columns:
        findf[c] = [str(v).replace('nan','').replace('None','').replace('-|-','|') for v in findf[c].values]
    for c in findf.columns:
        nvals = findf[c].values
        for nullval in  ['-',';','_','|','',' ','0']:
            nvals = [str(v).strip(nullval).strip() for v in nvals]
        if sum([len(v)==0 for v in nvals]) > 0:
            findf[c] = nvals
    findf = findf.replace('',np.nan)
    findf['publication_identifier'] = findf['publication_identifier'].str.lower()
    findf.to_csv(newpath, sep='\t', index=False)

def do_update(save_dir:str, save_zipname: str, latest_zip_url: str, uniprots_to_get:set) -> None:
    """
    Handles practicalities of updating the biogrid tsv file on disk

    :param save_dir: directory where the datafiles will be put
    :param save_zipname: filename for the zipfile that will be downloaded
    :param latest_zip_url: url for the zip to download from BioGRID
    :param uniprots_to_get: a set of which uniprots should be retained.
    """
    urlretrieve(latest_zip_url,os.path.join(save_dir, save_zipname))
    with ZipFile(os.path.join(save_dir, save_zipname), 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    generate_pandas(os.path.join(save_dir, save_zipname.replace('.zip','.txt')), uniprots_to_get)

def get_latest() -> pd.DataFrame:
    """
    Fetches the latest data from disk

    :returns: Pandas dataframe of the latest BioGRID data.
    """
    current_version: str = apitools.get_newest_file(apitools.get_save_location('BioGRID'),namefilter='.tsv')
    return pd.read_csv(
        os.path.join(apitools.get_save_location('BioGRID'), current_version),
        index_col = 'interaction',
        sep='\t',
        low_memory=False
    )

#TODO: check uniprots in should_update bool check too, not just version.
def update(uniprots_to_get:set) -> None:
    """
    Updates the database, if required

    :param uniprots_to_get: uniprots to retain in the database    
    """
    url = 'https://downloads.thebiogrid.org/BioGRID/Release-Archive'
    r = get(url).text.split('\n')
    r = [rr.strip().split('href=')[1].split('\' title')[0].strip().strip('\'') for rr in r if 'https://downloads.thebiogrid.org/BioGRID/Release-Archive/' in rr]
    latest = sorted(r,reverse=True)[0]
    latest_zipname = f'{latest.replace(".org/",".org/Download/")}{latest.rsplit("/",maxsplit=2)[-2].replace("BIOGRID","BIOGRID-ALL")}.tab3.zip'
    uzip = latest_zipname.rsplit('/',maxsplit=1)[1]
    save_location:str = apitools.get_save_location('BioGRID')
    should_update:bool = False
    try:
        should_update = uzip != apitools.get_newest_file(save_location)
    except ValueError:
        should_update = True
    if should_update:
        do_update(save_location, uzip, latest_zipname, uniprots_to_get)
        
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
    