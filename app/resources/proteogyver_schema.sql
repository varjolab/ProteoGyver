CREATE TABLE proteins (
    uniprot_id TEXT PRIMARY KEY,
    is_reviewed INTEGER NOT NULL,
    gene_name TEXT,
    entry_name TEXT NOT NULL,
    all_gene_names TEXT,
    organism TEXT NOT NULL,
    length INTEGER NOT NULL,
    sequence TEXT NOT NULL,
    is_latest INTEGER NOT NULL,
    entry_source TEXT NOT NULL,
    version_update_time TEXT NOT NULL,
    prev_version TEXT
);

CREATE TABLE contaminants (
    uniprot_id TEXT PRIMARY KEY,
    is_reviewed INTEGER NOT NULL,
    gene_name TEXT,
    entry_name TEXT NOT NULL,
    all_gene_names TEXT,
    organism TEXT NOT NULL,
    length INTEGER NOT NULL,
    sequence TEXT,
    entry_source TEXT NOT NULL,
    contamination_source TEXT NOT NULL,
    version_update_time TEXT NOT NULL,
    prev_version TEXT
);

CREATE TABLE known_interactions (
    annotation_interactor_a TEXT,
    annotation_interactor_b TEXT,
    biogrid_creation_date TEXT,
    biogrid_updated_date TEXT,
    biological_role_interactor_a TEXT,
    biological_role_interactor_b TEXT,
    confidence_value TEXT,
    experimental_role_interactor_a TEXT,
    experimental_role_interactor_b TEXT,
    intact_creation_date TEXT,
    intact_update_date TEXT,
    interaction TEXT PRIMARY KEY,
    interaction_detection_method TEXT,
    interaction_type TEXT,
    isoform_a TEXT,
    isoform_b TEXT,
    modification TEXT,
    notes TEXT,
    ontology_term_categories TEXT,
    ontology_term_ids TEXT,
    ontology_term_names TEXT,
    ontology_term_qualifier_ids TEXT,
    ontology_term_qualifier_names TEXT,
    organism_interactor_a TEXT,
    organism_interactor_b TEXT,
    prev_version TEXT,
    publication_count TEXT,
    publication_identifier TEXT,
    qualifications TEXT,
    source_database TEXT NOT NULL,
    throughput TEXT,
    uniprot_id_a TEXT NOT NULL,
    uniprot_id_a_noiso TEXT NOT NULL,
    uniprot_id_b TEXT NOT NULL,
    uniprot_id_b_noiso TEXT NOT NULL,
    update_time TEXT,
    version_update_time TEXT
);

CREATE TABLE msmicroscopy (
    Interaction TEXT PRIMARY KEY,
    Bait TEXT NOT NULL,
    Prey TEXT NOT NULL,
    Bait_norm REAL NOT NULL,
    Bait_sumnorm REAL NOT NULL,
    Loc TEXT NOT NULL,
    Unique_to_loc REAL NOT NULL,
    Loc_norm REAL NOT NULL,
    Loc_sumnorm REAL NOT NULL,
    MSMIC_version TEXT NOT NULL,
    Version_update_time TEXT NOT NULL,
    Prev_version TEXT
);

CREATE TABLE common_proteins (
    uniprot_id TEXT PRIMARY KEY,
    gene_name TEXT,
    entry_name TEXT,
    all_gene_names TEXT,
    organism TEXT,
    protein_type TEXT NOT NULL,
    version_update_time TEXT NOT NULL,
    prev_version TEXT
);

CREATE TABLE ms_plots (
    internal_run_id TEXT PRIMARY KEY,
    BPC_auc REAL,
    BPC_maxtime REAL,
    BPC_max_intensity REAL,
    BPC_mean_intensity REAL,
    BPC_trace TEXT,
    MSn_auc REAL,
    MSn_maxtime REAL,
    MSn_max_intensity REAL,
    MSn_mean_intensity REAL,
    MSn_trace TEXT,
    TIC_auc REAL,
    TIC_maxtime REAL,
    TIC_max_intensity REAL,
    TIC_mean_intensity REAL,
    TIC_trace TEXT
);

CREATE TABLE update_log (
    update_id TEXT,
    timestamp TEXT,
    update_type TEXT,
    modification_type TEXT,
    tablename TEXT,
    count INTEGER
);

CREATE TABLE data_versions (
            dataset TEXT,
            version TEXT,
            last_update_check TEXT
        );

CREATE TABLE "ms_runs"(
  internal_run_id TEXT,
  data_type TEXT,
  file_name TEXT,
  file_size INT,
  parsed_date TEXT,
  sample_id TEXT,
  file_name_clean TEXT,
  sample_name TEXT,
  run_date TEXT,
  run_start_time REAL,
  run_end_time REAL,
  run_last_scan_number INT,
  inst_model TEXT,
  inst_serial_no TEXT,
  inst_name TEXT
);

