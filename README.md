# Proteogyver

Proteogyver (PG) is a low-threshold, web-based platform for proteomics and interactomics data analysis. It provides tools for quality control, data visualization, and statistical analysis of mass spectrometry-based proteomics data. These should be used as rapid ways to get preliminary data (or in simple cases, publishable results) out of manageable chunks of MS rundata. PG is not intended to be a full-featured analysis platform, but rather a quick way to identify issues, characterize results, and move on to more detailed analysis. The additional tools of PG can be used for inspecting how MS is performing across a sample set (MS Inspector), and for generating colocalization heatmaps from microscopy data (Microscopy Colocalizer).

## Security

The app is insecure as it is. It is intended to be run on a network that is not exposed to the public internet, and contain data only accessible to trusted users.

## QC and quick analysis toolset

### Core Analysis Workflows
- **Proteomics Analysis**
  - Missing value handling and imputation
  - Data normalization
  - Statistical analysis and visualization
  - Differential abundance analysis with volcano plots
  - Enrichment analysis

- **Interactomics Analysis**
  - SAINT analysis integration
  - CRAPome filtering
  - Protein-protein interaction network visualization
  - MS-microscopy analysis
  - Known interaction mapping


### Usage

1. Access the web interface at `http://localhost:8050`
2. Upload your data and sample tables
3. Choose your workflow (Proteomics or Interactomics)
4. Configure analysis parameters
5. Export results in various formats (HTML, PNG, PDF, TSV)

### Input Data Format
- Sample table must include:
  - "Sample name" column
  - "Sample group" column
  - "Bait uniprot" column (for interactomics)
- Supported data formats:
  - Interactomics:
    - FragPipe (combined_prot.tsv, reprint.spc)
    - Generic matrix format
  - Proteomics:
    - FragPipe (combined_prot.tsv)
    - DIA-NN (pg_matrix.tsv, report.tsv (discouraged due to size))
    - Generic matrix format

## Additional Tools
- **MS Inspector**: Interactive visualization and analysis of MS performance through TIC graphs
- **Microscopy Image Colocalizer**: Analysis tool for .lif image files

### Microscopy Colocalizer

The Microscopy Colocalizer is a tool for analyzing spatial relationships between different fluorescent channels in microscopy images.

#### Features
- Multi-channel image visualization
- Colocalization analysis and colocalization map generation
- Support for .lif (Leica), other formats may be supported in the future

#### Usage
1. Upload your microscopy file (only .lif is supported for now)
2. Select channels for analysis
3. Select the Z-stack for analysis
4. Generate colocalization maps
5. Export results as merged channel visualizations

### MS Inspector

The MS Inspector is a tool for visualizing and analyzing Mass Spectrometry (MS) performance through chromatogram graphs and related metrics.

#### Features
- Interactive TIC visualization with animation controls
- Multiple trace types support (TIC, BPC)
- Supplementary metrics tracking:
  - Area Under the Curve (AUC)
  - Mean intensity
  - Maximum intensity
- Sample filtering by:
  - Date range
  - Sample type
  - Run IDs
- Data export in multiple formats:
  - HTML interactive plots
  - PNG images
  - PDF documents
  - TSV data files

#### Usage
1. Select MS instrument from dropdown
2. Choose analysis period
3. Filter by sample type(s) or input specific run IDs
4. Click "Load runs by selected parameters"
5. Use animation controls to explore TIC graphs as a time series:
   - Start/Stop: Toggle automatic progression
   - Previous/Next: Manual navigation
   - Reset: Return to first run
6. Switch between TIC and BPC metrics using dropdown
7. Export visualizations and data using "Download Data"

#### Notes
- Maximum of 100 runs can be loaded at once
- Multiple traces are displayed with decreasing opacity for temporal comparison
- Supplementary metrics are synchronized with TIC visualization
- For switching to a different run set, reload the page to ensure clean state
- Prerequisite for the use of the tool, as well as chromatogram visualization in the QC workflow, is the pre-analysis of MS rawfiles and their inclusion into the PG database with the bundled database updater tool. See the "Updating the database" section for more information.

## Installation

### Docker Installation (recommended use case)

Build the Docker images, run the preparation script, and run the PG updater to generate a database
**!!NOTE!!: the preparation script will assume that PG_DATA_DIR environment variable is set to the base directory for PG data. By default, it will use /data. As this can vary between systems, it is recommended to set the PG_DATA_DIR environment variable to a more suitable location, or modify the script if needed.**
```
docker build -t proteogyver:1.0 -f dockerfiles/dockerfile .
docker build -t pg_updater:1.0 -f dockerfiles/dockerfile_pg_updater .
bash utils/preparations.sh
bash utils/run_updater.sh
```
Run the docker stack
```
docker compose -f dockerfiles/docker-compose.yaml up
```

### Run PG locally (unsupported, but possible)

Runnin gPG locally is possible, especially for testing and development use. However, it is not a supported use case.
On a windows computer, it is recommended to use WSL. These instructions apply only to linux systems.

#### Requirements and setup:
- conda
- Python 3.10+
- redis-cli
- celery

```
# Create the PG conda environment and install dependencies
conda env create -f resources/environment.yml
# Generate a database. This can take a while.
conda activate PG
python database_admin.py
```

#### Run PG:
```
conda activate PG
bash startup.sh
```


## Updating the database
To update the database, use the updater container. E.g. docker run proteogyver_updater.
On the first run, it will create a new database file in the specified db directory (specified in parameters.toml), if the file does not exist. In other cases, it will update the existing database. For updates, data will be added to existing tables from the update files directory (specified in parameters.toml). If it does not exist, the updater will create it, as well as example files for each database table. Crapome and control set table examples will not be created, because they would clutter up the output. For each of these tables, lines in them represent either new data rows, or modifications to existing rows. Deletions are handled differently, and are described below.

IF the files handed to updater contain columns, that are not in the existing tables, the updater will add them. However, the column names will be sanitized to only contain lowercase letters, numbers, and underscores, and any consecutive underscores will be removed. E.g. "gene name" will be changed to "gene_name". If a column starts with a number, "c" will be added to the beginning of the name. E.g. "1.2.3" will be changed to "c1_2_3".

When updating existing entries in the database, if the update file does not contain a column that is present in the database, or a row of the update file has no value for a column,the updater will impute values from the existing entries in the database.

Keep in mind that the updater will delete the files from the db_updates directories after it has finished running.

### Update running order:
1) External data is updated first.
2) Deletions are handled next.
3) Additions and replacements are handled next. 
4) Finally other modifications are applied.

If the tools that provide the external data provide ANY new columns that do not already exist in the database, the new columns will need to be manually added to the database FIRST. Otherwise the updater will throw an error.

### MS run data pre-analysis
MS run data needs to be pre-analyzed. As it may not be desirable to present run files directly to the server PG is running on, PG assumes that run file pre-analysis .json files are present in the directory specified in parameters.toml at "database creation"."MS runs information"."MS run data dir for updater". The script for pre-analysis is provided as parse_tims_data.py. Currently it is tailored for the data generated by the TimsTOF Pro2, but it should be easily adaptable for other platforms. As long as all the keys have values in the correct format in the .json files, the data will be parsed correctly. See MS run data pre-analysis for more information.

### Adding new crapome or control sets:
Two files per set are needed:
1) The crapome/control overall table needs an update, and for that the control_sets.tsv or crapome_sets.tsv example file can be added to, and then put into the db_updates/crpaome_sets or db_updates/control_sets directory.
2) The individual crapome/control set needs its own update file added to the db_updates/add_or_replace directory. The file should have the same columns, as existing crapome/control set tables (specified in parameters.toml at "database creation"."control and crapome db detailed columns"). The column types can be found in "database creation"."control and crapome db detailed types". 

### Adding other new tables:
In order to add any other new tables, two updates and two files are needed:
1) .tsv file in the "add_or_replace" directory. Column names MUST NOT contain any spaces. Otherwise the updater will throw an error.
2) .txt file in the "add_or_replace" directory with the exact same name as the .tsv file, except it MUST have a .txt extension. This file contains the column types for the new table. One line per column, in the same order as the columns in the .tsv file. It should contain only the types. For example, if the .tsv file has the following columns: "uniprot_id", "gene_name", "description", "spectral_count", the .txt file should have the following lines: "TEXT PRIMARY KEY", "TEXT", "TEXT", "INTEGER". Empty lines and lines starting with '#' are ignored.
3) The new table needs to be added to the ["Database updater"."Update files"] list with the same name as the .tsv file, but without the .tsv extension.
4) In order to generate an empty template file for future updates, the ["Database updater"."Database table primary keys"] list in parameters.toml needs to be added to. 

### Deleting data
To delete data rows, the syntax is different. Each file in the remove_data -directory should be named exactly the same as the table it is deleting from + .tsv. E.g. to delete from table "proteins", name the file proteins.tsv. One row should contain one criteria in the format of "column_name, value\tcolumn_name2, value2", without quotes. The tab separates criterias from one another, and all criteria of a row will have to match for the deletion. E.g.
uniprot_id, UPID1\tgene_name, GENE12
will match the rows in the database where uniprot_id is UPID1 and gene_name is GENE12.

Empty lines and lines starting with '#' are ignored.

Deleting columns from tables is not supported this way, nor is deleting entire tables. These need to be done manually. The database is sqlite3, and thus easy to work with. Please make a backup first.

### Update logging
Updates will be logged to the update_log table.

## Advanced use cases

### Embedding other websites as tabs within Proteogyver
To embed another website/tool within Proteogyver, add a line to embed_pages.tsv, and run the embedded_page_updater.py script. Preferably these will be things hosted on the same server, but this is not required. Current example is proteomics.fi (hosted externally). Keep in mind that most websites ban browsers from accessing if they are embedded in an html.Embed element.

### Adding custom tools as tabs to Proteogyver
Adding custom tools to Proteogyver is supported as pages in the app/pages folder. Here the following rules should be followed:
- Use dash.register_page to register the page (register_page(__name__, path='/YOUR_PAGE_NAME') )
- Use GENERIC_PAGE from element_styles.py for styling starting point. Mostly required from this is the offset on top of the page to fit the navbar

### Accessing the database from other tools
Other tools can access the database, preferably using methods contained within the db_functions.py file. Writes to the database should not require any specific precautions. However, please check that the database is not locked, and another transaction is not in progress. Other scenarios when one should not write to the database include if it is in the process of being backed up (currently not implemented, but planned), or while the updater is actively running.

## MS run data pre-analysis
# TODO: add documentation.

## License

[Add your license information here]

## Citation

If you use Proteogyver or a part of it in your research, please cite:
[Add citation information here]
