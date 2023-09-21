
import sys
import os
import re
import warnings
from typing import Union
import requests
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
import json

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import apitools


# TODO: move to a separate file
def get_uniprot_column_map() -> dict:
    """Returns currently understood map of uniprot columns in a dict."""
    return {'Entry': 'accession',
            'Absorption': 'absorption',
            'Annotation': 'annotation_score',
            'Activity regulation': 'cc_activity_regulation',
            'Allergenic properties': 'cc_allergen',
            'Alternative products': 'cc_alternative_products',
            'Biotechnological use': 'cc_biotechnology',
            'Catalytic activity': 'cc_catalytic_activity',
            'Caution': 'cc_caution',
            'Cofactor': 'cc_cofactor',
            'Developmental stage': 'cc_developmental_stage',
            'Involvement in disease': 'cc_disease',
            'Disruption phenotype': 'cc_disruption_phenotype',
            'Domain[CC]': 'cc_domain',
            'Function [CC]': 'cc_function',
            'Induction': 'cc_induction',
            'Interacts with': 'cc_interaction',
            'Mass spectrometry': 'cc_mass_spectrometry',
            'Miscellaneous [CC]': 'cc_miscellaneous',
            'Pathway': 'cc_pathway',
            'Pharmaceutical use': 'cc_pharmaceutical',
            'Polymorphism': 'cc_polymorphism',
            'Post-translational modification': 'cc_ptm',
            'RNA editing': 'cc_rna_editing',
            'Sequence caution': 'cc_sequence_caution',
            'Subcellular location[CC]': 'cc_subcellular_location',
            'Subunit structure[CC]': 'cc_subunit',
            'Tissue specificity': 'cc_tissue_specificity',
            'Toxic dose': 'cc_toxic_dose',
            'Comment Count': 'comment_count',
            'Date of creation': 'date_created',
            'Date of last modification': 'date_modified',
            'Date of last sequence modification': 'date_sequence_modified',
            'EC number': 'ec',
            'Erroneous gene model prediction': 'error_gmodel_pred',
            'Features': 'feature_count',
            'Fragment': 'fragment',
            'Active site': 'ft_act_site',
            'Binding site': 'ft_binding',
            # this field name causes problems 2022-08-22
            # 'Calcium binding': 'ft_ca_bind',
            'Glycosylation': 'ft_carbohyd',
            'Chain': 'ft_chain',
            'Coiled coil': 'ft_coiled',
            'Compositional bias': 'ft_compbias',
            'Sequence conflict': 'ft_conflict',
            'Cross-link': 'ft_crosslnk',
            'Disulfide bond': 'ft_disulfid',
            'DNA binding': 'ft_dna_bind',
            'Domain[FT]': 'ft_domain',
            'Helix': 'ft_helix',
            'Initiator methionine': 'ft_init_met',
            'Intramembrane': 'ft_intramem',
            'Lipidation': 'ft_lipid',
            # this field name causes problems 2022-08-22
            # 'Metal binding': 'ft_metal',
            'Modified residue': 'ft_mod_res',
            'Motif': 'ft_motif',
            'Mutagenesis': 'ft_mutagen',
            'Non-adjacent residues': 'ft_non_cons',
            'Non-standard residue': 'ft_non_std',
            'Non-terminal residue': 'ft_non_ter',
            # this field name causes problems 2022-08-22
            # 'Nucleotide binding': 'ft_np_bind',
            'Peptide': 'ft_peptide',
            'Propeptide': 'ft_propep',
            'Region': 'ft_region',
            'Repeat': 'ft_repeat',
            'Signal peptide': 'ft_signal',
            'Site': 'ft_site',
            'Beta strand': 'ft_strand',
            'Topological domain': 'ft_topo_dom',
            'Transit peptide': 'ft_transit',
            'Transmembrane': 'ft_transmem',
            'Turn': 'ft_turn',
            'Sequence uncertainty': 'ft_unsure',
            'Alternative sequence': 'ft_var_seq',
            'Natural variant': 'ft_variant',
            'Zinc finger': 'ft_zn_fing',
            'Gene names': 'gene_names',
            'Gene names (ordered locus)': 'gene_oln',
            'Gene names (ORF)': 'gene_orf',
            'Gene names (primary)': 'gene_primary',
            'Gene names (synonym)': 'gene_synonym',
            'Gene ontology (GO)': 'go',
            'Gene ontology (cellular component)': 'go_c',
            'Gene ontology (molecular function)': 'go_f',
            'Gene ontology IDs': 'go_id',
            'Gene ontology (biological process)': 'go_p',
            'Entry name': 'id',
            'Keywords': 'keyword',
            'Keyword ID': 'keywordid',
            'Kinetics': 'kinetics',
            'Length': 'length',
            'Taxonomic lineage': 'lineage',
            'Taxonomic lineage (IDs)': 'lineage_ids',
            'PubMed ID': 'lit_pubmed_id',
            'Mass': 'mass',
            'Gene encoded by': 'organelle',
            'Organism ID': 'organism_id',
            'Organism': 'organism_name',
            'pH dependence': 'ph_dependence',
            'Protein existence': 'protein_existence',
            'Protein families': 'protein_families',
            'Protein names': 'protein_name',
            'Redox potential': 'redox_potential',
            'Reviewed': 'reviewed',
            'Rhea ID': 'rhea',
            'Sequence': 'sequence',
            'Sequence version': 'sequence_version',
            '3D': 'structure_3d',
            'Temperature dependence': 'temp_dependence',
            'Tools': 'tools',
            'UniParc': 'uniparc_id',
            'Entry version': 'version',
            'Virus hosts': 'virus_hosts',
            'ABCD': 'xref_abcd',
            'Allergome': 'xref_allergome',
            'AlphaFoldDB': 'xref_alphafolddb',
            'Antibodypedia': 'xref_antibodypedia',
            'ArachnoServer': 'xref_arachnoserver',
            'Araport': 'xref_araport',
            'Bgee': 'xref_bgee',
            'BindingDB': 'xref_bindingdb',
            'BioCyc': 'xref_biocyc',
            'BioGRID': 'xref_biogrid',
            'BioGRID-ORCS': 'xref_biogrid-orcs',
            'BioMuta': 'xref_biomuta',
            'BMRB': 'xref_bmrb',
            'BRENDA': 'xref_brenda',
            'CarbonylDB': 'xref_carbonyldb',
            'CAZy': 'xref_cazy',
            'CCDS': 'xref_ccds',
            'CDD': 'xref_cdd',
            'CGD': 'xref_cgd',
            'ChEMBL': 'xref_chembl',
            'ChiTaRS': 'xref_chitars',
            'CLAE': 'xref_clae',
            'CleanEx': 'xref_cleanex',
            'CollecTF': 'xref_collectf',
            'ComplexPortal': 'xref_complexportal',
            'COMPLUYEAST-2DPAGE': 'xref_compluyeast-2dpage',
            'ConoServer': 'xref_conoserver',
            'CORUM': 'xref_corum',
            'CPTAC': 'xref_cptac',
            'CPTC': 'xref_cptc',
            'CTD': 'xref_ctd',
            'dbSNP': 'xref_dbsnp',
            'DEPOD': 'xref_depod',
            'dictyBase': 'xref_dictybase',
            'DIP': 'xref_dip',
            'DisGeNET': 'xref_disgenet',
            'DisProt': 'xref_disprot',
            'DMDM': 'xref_dmdm',
            'DNASU': 'xref_dnasu',
            'DOSAC-COBS-2DPAGE': 'xref_dosac-cobs-2dpage',
            'DrugBank': 'xref_drugbank',
            'DrugCentral': 'xref_drugcentral',
            'EchoBASE': 'xref_echobase',
            'eggNOG': 'xref_eggnog',
            'ELM': 'xref_elm',
            'EMBL': 'xref_embl',
            'Ensembl': 'xref_ensembl',
            'EnsemblBacteria': 'xref_ensemblbacteria',
            'EnsemblFungi': 'xref_ensemblfungi',
            'EnsemblMetazoa': 'xref_ensemblmetazoa',
            'EnsemblPlants': 'xref_ensemblplants',
            'EnsemblProtists': 'xref_ensemblprotists',
            'EPD': 'xref_epd',
            'ESTHER': 'xref_esther',
            'euHCVdb': 'xref_euhcvdb',
            'EvolutionaryTrace': 'xref_evolutionarytrace',
            'ExpressionAtlas': 'xref_expressionatlas',
            'FlyBase': 'xref_flybase',
            'Gene3D': 'xref_gene3d',
            'GeneCards': 'xref_genecards',
            'GeneID': 'xref_geneid',
            'GeneReviews': 'xref_genereviews',
            'GeneTree': 'xref_genetree',
            'Genevisible': 'xref_genevisible',
            'GeneWiki': 'xref_genewiki',
            'GenomeRNAi': 'xref_genomernai',
            'GlyConnect': 'xref_glyconnect',
            'GlyGen': 'xref_glygen',
            'Gramene': 'xref_gramene',
            'GuidetoPHARMACOLOGY': 'xref_guidetopharmacology',
            'HAMAP': 'xref_hamap',
            'HGNC': 'xref_hgnc',
            'HOGENOM': 'xref_hogenom',
            'HPA': 'xref_hpa',
            'IDEAL': 'xref_ideal',
            'IMGT_GENE-DB': 'xref_imgt_gene-db',
            'InParanoid': 'xref_inparanoid',
            'IntAct': 'xref_intact',
            'InterPro': 'xref_interpro',
            'iPTMnet': 'xref_iptmnet',
            'jPOST': 'xref_jpost',
            'KEGG': 'xref_kegg',
            'KO': 'xref_ko',
            'LegioList': 'xref_legiolist',
            'Leproma': 'xref_leproma',
            'MaizeGDB': 'xref_maizegdb',
            'MalaCards': 'xref_malacards',
            'MANE-Select': 'xref_mane-select',
            'MassIVE': 'xref_massive',
            'MaxQB': 'xref_maxqb',
            'MEROPS': 'xref_merops',
            'MetOSite': 'xref_metosite',
            'MGI': 'xref_mgi',
            'MIM': 'xref_mim',
            'MINT': 'xref_mint',
            'MoonDB': 'xref_moondb',
            'MoonProt': 'xref_moonprot',
            'neXtProt': 'xref_nextprot',
            'NIAGADS': 'xref_niagads',
            'OGP': 'xref_ogp',
            'OMA': 'xref_oma',
            'OpenTargets': 'xref_opentargets',
            'Orphanet': 'xref_orphanet',
            'OrthoDB': 'xref_orthodb',
            'PANTHER': 'xref_panther',
            'PathwayCommons': 'xref_pathwaycommons',
            'PATRIC': 'xref_patric',
            'PaxDb': 'xref_paxdb',
            'PCDDB': 'xref_pcddb',
            'PDB': 'xref_pdb',
            'PDBsum': 'xref_pdbsum',
            'PeptideAtlas': 'xref_peptideatlas',
            'PeroxiBase': 'xref_peroxibase',
            'Pfam': 'xref_pfam',
            'PharmGKB': 'xref_pharmgkb',
            'Pharos': 'xref_pharos',
            'PHI-base': 'xref_phi-base',
            'PhosphoSitePlus': 'xref_phosphositeplus',
            'PhylomeDB': 'xref_phylomedb',
            'PIR': 'xref_pir',
            'PIRSF': 'xref_pirsf',
            'PlantReactome': 'xref_plantreactome',
            'PomBase': 'xref_pombase',
            'PRIDE': 'xref_pride',
            'PRINTS': 'xref_prints',
            'PRO': 'xref_pro',
            'ProDom': 'xref_prodom',
            'ProMEX': 'xref_promex',
            'PROSITE': 'xref_prosite',
            'Proteomes': 'xref_proteomes',
            'ProteomicsDB': 'xref_proteomicsdb',
            'PseudoCAP': 'xref_pseudocap',
            'Reactome': 'xref_reactome',
            'REBASE': 'xref_rebase',
            'RefSeq': 'xref_refseq',
            'REPRODUCTION-2DPAGE': 'xref_reproduction-2dpage',
            'RGD': 'xref_rgd',
            'RNAct': 'xref_rnact',
            'SABIO-RK': 'xref_sabio-rk',
            'SASBDB': 'xref_sasbdb',
            'SFLD': 'xref_sfld',
            'SGD': 'xref_sgd',
            'SignaLink': 'xref_signalink',
            'SIGNOR': 'xref_signor',
            'SMART': 'xref_smart',
            'SMR': 'xref_smr',
            'STRING': 'xref_string',
            'SUPFAM': 'xref_supfam',
            'SWISS-2DPAGE': 'xref_swiss-2dpage',
            'SwissLipids': 'xref_swisslipids',
            'SwissPalm': 'xref_swisspalm',
            'TAIR': 'xref_tair',
            'TCDB': 'xref_tcdb',
            'TIGRFAMs': 'xref_tigrfams',
            'TopDownProteomics': 'xref_topdownproteomics',
            'TreeFam': 'xref_treefam',
            'TubercuList': 'xref_tuberculist',
            'UCD-2DPAGE': 'xref_ucd-2dpage',
            'UCSC': 'xref_ucsc',
            'UniCarbKB': 'xref_unicarbkb',
            'UniLectin': 'xref_unilectin',
            'UniPathway': 'xref_unipathway',
            'VectorBase': 'xref_vectorbase',
            'VEuPathDB': 'xref_veupathdb',
            'VGNC': 'xref_vgnc',
            'WBParaSite': 'xref_wbparasite',
            'WBParaSiteTranscriptProtein': 'xref_wbparasitetranscriptprotein',
            'World-2DPAGE': 'xref_world-2dpage',
            'WormBase': 'xref_wormbase',
            'Xenbase': 'xref_xenbase',
            'ZFIN': 'xref_zfin'
            }

def get_default_uniprot_column_map() -> dict:
    """Returns a sensible, yet extensive default map of uniprot columns in a dict."""
    return {'Entry': 'accession',
            'Post-translational modification': 'cc_ptm',
            'Comment Count': 'comment_count',
            'Organism': 'organism_name',
            'Modified residue': 'ft_mod_res',
            'Gene names': 'gene_names',
            'Gene names (primary)': 'gene_primary',
            'Gene ontology (cellular component)': 'go_c',
            'Gene ontology (molecular function)': 'go_f',
            'Gene ontology (biological process)': 'go_p',
            'Entry name': 'id',
            'Keywords': 'keyword',
            'Length': 'length',
            'PubMed ID': 'lit_pubmed_id',
            'Mass': 'mass',
            'Protein names': 'protein_name',
            'Reviewed': 'reviewed',
            'Sequence': 'sequence',
            'Entry version': 'version',
            'Reactome': 'xref_reactome',
            'Cofactor': 'cc_cofactor',
            'Developmental stage': 'cc_developmental_stage',
            'Involvement in disease': 'cc_disease',
            'Disruption phenotype': 'cc_disruption_phenotype',
            'Domain[CC]': 'cc_domain',
            'Function [CC]': 'cc_function',
            'Induction': 'cc_induction',
            'Interacts with': 'cc_interaction',
            'Miscellaneous [CC]': 'cc_miscellaneous',
            'Sequence caution': 'cc_sequence_caution',
            'Subcellular location[CC]': 'cc_subcellular_location',
            'Date of last sequence modification': 'date_sequence_modified',
            'Organism ID': 'organism_id',
            'PhosphoSitePlus': 'xref_phosphositeplus',
            'Sequence version': 'sequence_version',
            'RefSeq': 'xref_refseq',
            'STRING': 'xref_string',
            'Ensembl': 'xref_ensembl',
            'IntAct': 'xref_intact',
            'InterPro': 'xref_interpro',
            'PANTHER': 'xref_panther',
            'SUPFAM': 'xref_supfam',
            'Pfam': 'xref_pfam',
            'PRO': 'xref_pro',
            'PROSITE': 'xref_prosite',
            }


def __get_uniprot_batch(batch_url: str, session: requests.Session) -> tuple:
    """Retrieves batches from UniProt, results response, and total number of results.

    Yields:
    a tuple of (requests.models.Response, total:str)
    """
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        try:
            total = response.headers["x-total-results"]
        except KeyError:
            total = 1
        yield (response, total)
        batch_url: str = __get_uniprot_next_link(response.headers)


def __get_uniprot_next_link(headers: requests.structures.CaseInsensitiveDict) -> str:
    """Returns link to the next page of results in a UniProt query, or None is no link is found.
    """
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)
    return None


def download_uniprot_chunks(progress: bool = False, organism: int = 9606,
                            fields: list = None, reviewed_only: bool = True) -> pd.DataFrame:
    """Downloads whole uniprot for a given organism using pagination.

    Entry -column will always be the first column and used as the index in the output dataframe.

    Parameters:
    progress: Progress report printing
    organism: human by default, otherwise specify organism ID (e.g. human is 9606)
    fields: uniprot field labels for fields to retrieve. Refer to \
        https://www.uniprot.org/help/return_fields for help with field Labels \
        (from the label column). If None, download a default selection.
    """
    if reviewed_only:
        reviewed:str = '%29%20AND%20%28reviewed%3Atrue'
    else:
        reviewed = ''

    retries: Retry = Retry(total=5, backoff_factor=0.25,
                    status_forcelist=[500, 502, 503, 504])
    session: requests.Session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    fieldstr: list = []
    headers: list = []
    output_format: str = 'tsv'
    if not fields:
        field_dict: dict = get_default_uniprot_column_map()
        fields: list = [
            'Entry', 'Entry version', 'Reactome', 'Entry name', 'Gene names',
            'Protein names', 'Reviewed', 'Organism', 'Comment Count', 'Keywords',
            'Modified residue', 'PubMed ID', 'Post-translational modification',
            'Gene ontology (cellular component)', 'Gene ontology (molecular function)',
            'Gene ontology (biological process)', 'Length', 'Mass', 'Sequence'
        ]
        for field in fields:
            fieldstr.append(field_dict[field])
            headers.append(field)
        have: set = set(headers)
        for name, field in field_dict.items():
            if name not in have:
                fieldstr.append(field)
                headers.append(name)
    else:
        field_dict = get_uniprot_column_map()
        fieldstr.append(field_dict['Entry'])
        headers.append('Entry')
        for field in fields:
            if field == 'Entry':
                continue
            if field not in field_dict:
                warnings.warn((
                    f'Field {field} not found in UniProt. Refer to '
                    f'https://www.uniprot.org/help/return_fields for help with field Labels.'
                ), UserWarning)
            else:
                fieldstr.append(field_dict[field])
                headers.append(field)
    assert len(fieldstr) > 0, (
        'Empty UniProt label list. Refer to '
        'https://www.uniprot.org/help/return_fields for help with field Labels.'
    )
    fieldstr = '%2C'.join(fieldstr)
    pagination_url: str = (
        f'https://rest.uniprot.org/uniprotkb/search?fields={fieldstr}&format={output_format}&'
        f'query=%28taxonomy_id%3A{organism}{reviewed}%29&size=500'
    )
    url: str = pagination_url
    alltext: list = []
    index: list = []
    for batch, total in __get_uniprot_batch(url, session):
        for line in batch.text.splitlines()[1:]:
            line: list = line.split('\t')
            alltext.append(line[1:])
            index.append(line[0])
        if progress:
            print(f'{len(alltext)} / {total}')
    return pd.DataFrame(columns=headers[1:], data=alltext, index=pd.Series(index, name='Entry'))


def retrieve_uniprot(uniprotfile: str = 'Full human uniprot.tsv', **kwargs) -> pd.DataFrame:
    """Downloads full uniprot (reviewed entries only) to a file and returns the dataframe.

    Parameters:
    uniprotfile: path of the output file
    kwargs: kwargs to pass down to download_full_uniprot_for_organism, e.g. to specify which\
        organism or if progress should be reported.
    """
    if os.path.isfile(uniprotfile):
        updf:pd.DataFrame = pd.read_csv(uniprotfile, sep='\t')
    else:
        updf = download_full_uniprot_for_organism(**kwargs)
        updf.to_csv(uniprotfile, sep='\t',encoding = 'utf-8')
    updf.index = updf['Entry']
    updf = updf.drop(columns=['Entry'])
    return updf


def download_full_uniprot_for_organism(organism: Union[list, int] = None,
                                       columns=None, progress: bool = False,
                                       overall_progress=False, reviewed_only:bool = True) -> pd.DataFrame:
    """Downloads the full uniprot database EXCLUDING isoforms in a .tsv format for a \
        given organism.

    Parameters:
    organism: list of integer ID of the desired organism, e.g. human is 9606. If none, defaults\
        to human.
    progress: Print progress reports of how each batch download is going
    overall_progress: Print progress reports when each batch is finished"""
    if not organism:
        organism: list = [9606]
    elif isinstance(organism, int):
        organism: list = [organism]
    dfs_to_merge: list = []
    if not columns:
        columns: set = set(get_default_uniprot_column_map().keys())
    need: set = columns
    lim: int = 40
    next_batch: list = ['Entry']
    while len(need) > 0:
        if overall_progress:
            print(f'{len(need)} columns to go in {1+int(len(need)/lim)} batches.')
        next_batch.extend(list(need)[:(lim-1)])
        for organism_id in organism:
            dfs_to_merge.append(download_uniprot_chunks(fields=next_batch,
                                                        organism=organism_id, progress=progress, reviewed_only=reviewed_only))
        need = {k: v for k, v in need.items() if k not in next_batch}
        next_batch = ['Entry']
    return pd.concat(dfs_to_merge, axis=1)

def update(organism = 9606,progress=False) -> None:
    outdir: str = apitools.get_save_location('Uniprot')
    if is_newer_available(apitools.get_newest_file(outdir, namefilter=str(organism))):
        today: str = apitools.get_timestamp()
        df: pd.DataFrame = download_full_uniprot_for_organism(organism=organism,progress=progress,overall_progress=progress)
        outfile: str = os.path.join(outdir, f'{today}_Uniprot_{organism}')
        df.to_csv(f'{outfile}.tsv',encoding = 'utf-8',sep='\t')
        name_dict:dict = {}
        length_dict:dict = {}
        for _,row in df.iterrows():
            name_dict[row['Entry']] = row['Gene names (primary)']
            length_dict[row['Entry']] = row['Length']
        with open(f'{outfile}_protein_names.json','w',encoding='utf-8') as fil:
            json.dump(name_dict,fil)
        with open(f'{outfile}_protein_lengths.json','w',encoding='utf-8') as fil:
            json.dump(length_dict,fil)


def is_newer_available(newest_file: str, organism: int = 9606) -> bool:
    uniprot_url: str = f"https://rest.uniprot.org/uniprotkb/search?\
        query=organism_id:{organism}&format=fasta"
    uniprot_response: requests.Response = requests.get(uniprot_url)
    newest_version:str = uniprot_response.headers['X-UniProt-Release'].replace('_','-')
    vals: list =  newest_version.split('-')
    newest_y: int = int(vals[0])
    newest_m: int = int(vals[1])
    vals =  newest_file.split('_',maxsplit=1)[0].split('-')
    if len(vals) < 2:
        vals = [-1,-1]
    current_y: int = vals[0]
    current_m: int = vals[1]
    ret: bool = False
    if current_y < newest_y:
        ret = True
    elif current_y == newest_y:
        if current_m < newest_m:
            ret = True
    return ret

def get_version_info(organism:int=9606) -> str:
    nfile: str = apitools.get_newest_file(apitools.get_save_location('Uniprot'), namefilter=str(organism))
    return f'Downloaded ({nfile.split("_")[0]})'

def methods_text(organism=9606) -> str:
    short:str
    long: str
    pmid: str
    short,long,pmid = apitools.get_pub_ref('uniprot')
    return '\n'.join([
        f'Protein annotations were mapped from UniProt (https://uniprot.org) {short}',
        f'{get_version_info(organism)}',
        pmid,
        long
    ])

