# Proteogyver

Proteogyver (PG) is a low-threshold, web-based platform for proteomics and interactomics data analysis. It provides tools for quality control, data visualization, and statistical analysis of mass spectrometry-based proteomics data. These should be used as rapid ways to get preliminary data (or in simple cases, publishable results) out of manageable chunks of MS rundata. PG is not intended to be a full-featured analysis platform, but rather a quick way to identify issues, characterize results, and move on to more detailed analysis. The additional tools of PG can be used for inspecting how MS is performing across a sample set (MS Inspector), and for generating colocalization heatmaps from microscopy data (Microscopy Colocalizer).

## Table of contents:

## Security and compatibility

The app is insecure as it is. It is intended to be run on a network that is not exposed to the public internet. PG is designed to contain only nonsensitive data. Besides public databases, PG will optionally contain information about sample MS runs (run IDs, sample names, TIC/BPC etc.)
ProteoGyver is supplied as a docker container. It is only tested routinely on a ubuntu server, however should work just fine on other platforms as well. ARM-based systems may require a rebuild of the container.

## Example use cases
[Example use cases](./example%20use%20cases/) cover both interactomics and proteomcis workflows, as well as typical use of MS inspector. The examples are rudimentary, but they do showcase the outputs of the tools quite well.

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

- **Pipeline mode**
  - Any of the above workflows can be run in the background in automated fashion through the pipeline module.
  - Accessible via API or folder watcher script

### Usage
Example files are downloadable from the sidebar of the main interface. These include example data files and sample tables for interactomics, and proteomics workflows.

1. Access the web interface (host:port, e.g. localhost:8050, if running locally)
2. Upload your data and sample tables
3. Choose your workflow (Proteomics or Interactomics)
4. Choose analysis parameters
5. Export results in various formats (HTML, PNG, PDF, TSV)

### Input Data Format
- Sample table must include:
  - "Sample name" column
  - "Sample group" column
  - "Bait uniprot" column (for interactomics)
- Supported data formats:
  - Interactomics:
    - FragPipe (combined_prot.tsv, reprint.spc)
    - Generic matrix format (one row = one protein, one column = one sample)
  - Proteomics:
    - FragPipe (combined_prot.tsv)
    - DIA-NN (pg_matrix.tsv)
    - Generic matrix format (one row = one protein, one column = one sample)

### Algorithm details
ProteoGyver relies mostly on proven software. However, some choices have been made:
- p-values for differential abundance are adjusted with the Benjamini-Hochberg procedure
- log2 fold chance thresholding is also used for the abundance in differential tests. This relies on "raw" log2 fold change (1 = 2-fold change).

### PTM workflow
Currently the PTM workflow is not ready for deployment. However, PTMs can be analyzed in a rudimentary way with the proteomics workflow. In this case, the input data table should be a generic matrix, where the first column is the ID column specifying the protein and modification site, and all other columns represent intensity values of e.g. the identified peptide or site. In this case, the first column could contain values such as Protein12-siteY32, or similar. As long as each entry is unique, PG will ingest the file happily. The workflow will then produce the same plots, e.g. counts, intensity distributions, missing values, volcano plots etc. 

Alternatively, you can use e.g. the [DiaNN R package](https://github.com/vdemichev/diann-rpackage) to recalculate MaxLFQ on protein level based only on modified (e.g. phosphorylated) precursors. And then run the usual proteomics workflow. See the example R file in [utils/scripts/diann-phospho-requant.R](./utils/scripts/diann-phospho-requant.R). Since the DiaNN R package is a bit out of date, you will first need to convert the .parquet report to .tsv. This can be done e.g. via python:
> import pandas as pd
> df  = pd.read_parquet('report.parquet')
> df.to_csv('report.tsv',sep='\t')
as long as pandas, pyarrow, and fastparquet are installed via e.g. pip. With large reports, you will need to read/write in chunks.
The report.tsv is then ready for the script, and the resulting matrix ready for PG. Do keep in mind that R may change the column headers if special characters or spaces are present.

### Pipeline mode for automated processing
The pipeline module features an ingest directory (see parameters.toml [Pipeline module.Input watch directory], by default data/Server_input/Pipeline_input ). While the Proteogyver container is running, a watcher script will detect newly created folders in this directory, and launch the analysis in the background for each.

Each input folder should contain:
- pipeline.toml file
- Data table
- Sample table
Of these, the data table and sample table can be in a subdirectory, if so specified in the toml file.

For example files, see pipeline example inputs -directory

Once the analysis is done, output will be generated in the same directory, as the input. IF errors occur, ERRORS.txt will be generated, and reanalysis will not be performed.

To trigger reanalysis, the input folder must not contain either the error file, nor the "PG output" folder.

#### Pipeline input toml

The pipeline module is set to watch the /proteogyver/data/Server_input/Pipeline_input directory in the **container** by default. This should be mapped to a host path, where pipeline input files can be placed either automatically or manually. In the [docker compose](./dockerfiles/proteogyver/docker-compose.yaml) the host directory /data/PG/input is mounted at /proteogyver/data/Server_input, so the pipeline module will watch /data/PG/input/Pipeline_input for new directories. 

Each new directory represents a dataset to analyze. Each directory should contain three, optionally four, files:
- data file
- sample table
- pipeline input toml
- (proteomics comparisons)

Examples of these are available in the [example files](./app/data/PG%20example%20files/), or in the download zip that is obtained from the download example files button in the web GUI.

Full available parameters can be seen in the [default files](./app/data/Pipeline%20module%20default%20tomls/), which are split into common.toml, interactomics.toml, and proteomics.toml. The common has parameters available for all workflows, while the workflow specific ones deal with parameters for the workflows.

The toml file contains three sections (file to see for full list of parameters):
1) pipeline (common.toml)
2) general (common.toml)
3) workflow-specific (proteomics.toml/interactomics.toml)

The **only** section that is **absolutely mandatory** is this:
> [general]
> workflow = "interactomics" # OR "proteomics"
> data = "path/to/data/table.tsv"
> "sample table" = "path/to/sample/table.tsv"

The parameter file **needs** to be named something.toml. Preferably something_pipeline.toml.
Note that the data and sample tables do not need to be in the same directory, but the path specified needs to be relative to the .toml file, and they need to be accessible for the docker container. For example, it might be clearer to put the tables into "data" directory in the input directory.

At the end, this is an example of the file structure you should have on the HOST for PG to initiate the pipeline successfully:
/data/PG/input/Pipeline_input/AnalysisDir/pipeline.toml
/data/PG/input/Pipeline_input/AnalysisDir/data.tsv
/data/PG/input/Pipeline_input/AnalysisDir/sample table.tsv

and the pipeline.toml should contain:
> [general]
> workflow = "interactomics" 
> data = "data.tsv"
> "sample table" = "sample table.tsv"

#### Parameters not in the input toml
Since the input .toml can be very minimal, for all parameters that are NOT in it, PG will use values from the [default files](./app/data/Pipeline%20module%20default%20tomls/). For this reason, it is advisable to always specify things like additional controls, control sample groups, and crapome sets for interactomics, and control groups for proteomics. 

#### Initiating pipeline analysis via API
An alternative to direct file system access to the server is to use the API. The API is on the same port as the GUI, under /api/upload-pipeline-files. 
Here is a complete usage example via python:
>datafile = 'datafile.tsv'
>sample_table = 'sampletable.tsv'
>pipeline = 'pipeline.toml'
>server_url = 'proteogyver.server.com'
>server_port = 8050
>
>files = {
>    'data_table': open(datafile, 'rb'),
>    'sample_table': open(sample_table, 'rb'),
>    'pipeline_toml': open(pipeline, 'rb'),
>}
>
>response = requests.post(f"http://{server_url}:{server_port}/api/upload-pipeline-files", files=files)
>upload_dir_name = response.json()['upload_directory_name']
>----------------------------------------------
># Check status later
>status = requests.get(
>    f"http://{server_url}:{server_port}/api/pipeline-status",
>    params={'upload_directory_name': upload_dir_name}
>).json()
>print(f'Status: {status["status"]}, message: {status["message"]}')
># The status also includes upload directory name
>if status['status'] == 'error':
>    print(f"Error: {status['error_message']}")
>----------------------------------------------
># Download the output zip file
>response = requests.get(
>    f"http://{server_url}:{server_port}/api/download-output",
>    params={'upload_directory_name': upload_dir_name},
>    stream=True
>)
>
>if response.status_code == 200:
>    # Save the zip file
>    with open('PG output.zip', 'wb') as f:
>        for chunk in response.iter_content(chunk_size=8192):
>            f.write(chunk)
>    print("Downloaded PG output.zip")
>else:
>    print(f"Error: {response.json()}")

## Additional Tools
- **MS Inspector**: Interactive visualization and analysis of MS performance through TIC graphs
- **Microscopy Image Colocalizer**: Analysis tool for .lif image files

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
- Unfortunately the proteogyver container needs a restart to detect new runs in the database in MS inspector. This limitation will be fixed in the next update.
- Maximum of 100 runs can be loaded at once
- Multiple traces are displayed with decreasing opacity for temporal comparison
- Supplementary metrics are synchronized with TIC visualization
- For switching to a different run set, reload the page to ensure clean state
- Prerequisite for the use of the tool, as well as chromatogram visualization in the QC workflow, is the pre-analysis of MS rawfiles and their inclusion into the PG database with the bundled database updater tool. See the "Updating the database" section for more information.

### Microscopy Image Colocalizer

The Microscopy image colocalizer is a simple tool to generate colocalization images from multichannel .lif files from confocal microscopes. The tool lets the user choose the image in the series, the timepoint, and the level in the z-stack, as well as colormap for the individual channels and colocalization image. The tool allows zooming into the location of interest, and seamless export in .png format.

## Installation

### MS run data pre-analysis
This is optional, but highly recommended. In order for the MS-inspector to have data to work with, or for QC to display chromatograms, information about MS runs needs to be included in the database.

MS run data needs to be pre-analyzed. As it may not be desirable to present run files directly to the server PG is running on, PG assumes that rawfile pre-analysis .json files are present in the directory specified in parameters.toml at "Maintenance"."MS run parsing"."Input files". The parser script is provided in utils folder (MSParser subdir), with its own venv requirements.txt file. MSParser.py can handle timsTOF and thermo data currently. Tested MS systems so far include the TimsTOF Pro, Pro2, QExactive, Orbitrap Elite, Astral, and Astral Zoom.

to run, set up the MSParser venv:
```
cd utils/MSParser
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
And then run it. MSParser expects three inputs: path to a raw file (e.g. something.raw, or something.d), path to output **directory**, and path to an error file:
```
python3 MSParser.py /path/to/rawfile.d /path/to/output_dir/for/jsons/ /path/to/errorfile.txt
python3 MSParser.py /path/to/rawfile.raw /path/to/output_dir/for/jsons/ /path/to/errorfile.txt
```
It will parse the rawfile, and produce a .json file in the output directory, which is understood by MS_run_json_parser.py (run on a schedule by the proteogyver main container). The parser runs in the background, and will digest files in the directory specified in parameters at Maintenance.MS run parsing.Input files. By default, it will move the jsons afterwards to the directory specified in parameters at Maintenance.MS run parsing.Move done jsons into subdir. If the latter parameter is empty, files will be deleted after parsing. The parsed json files, if kept, will also be compressed (zip), when they accumulate.

### Docker Installation (recommended)


TODO: dockerhub/zenodo here.


#### Running the docker images

Next modify docker-compose NOW to suit your local system if needed.

For production use, the updater is required for external data to stay up to date. It is encouraged to run the updater container as a periodical service, and adjust the intervals between e.g. external updates via the parameters.toml file (see below). On the first, run, the updater will create a database, if one doesn't yet exist.

Building the updater container should take around a minute. Running the updater can take a long time, especially on the first run.
**All commands should be run from the proteogyver root folder**


Then run the updater to generate a database:
```
cd dockerfiles/pg_updater && docker compose up
```

##### Changing parameters
In order to keep the parameters.toml in sync with PG and the updater container, it is copied into path specified in the docker-compose.yaml. The file needs to be edited in that location ONLY, in order for the updated parameters to be applied to existing docker container, and the updater (e.g. different database name, or modified update intervals). When pg_updater or proteogyver is run the first time, it will copy the default parameters.toml into config folder.

#### Run the container
- Modify the dockerfiles/docker-compose.yaml file to suit your environment, and then deploy the container:
```
cd dockerfiles/proteogyver && docker compose up
```

##### Volume paths
PG will generate data on disk in the form of tempfiles when a dataset is requested for download, and when certain functions are used (e.g. imputation). As such, it is suggested that the cache folder (/proteogyver/cache) is mounted from e.g. tmpfs (/tmp on most linux distros) or similar, for speed and latency.

/proteogyver/data/db is suggested to live on an externally mounted directory due to database size. 
/proteogyver/data/Server_input should contain the MS_rundata directory, which houses .json files for MS runs that should be included in the database by the updater.
/proteogyver/data/Server_output currently has no use, but will in the future be used for larger exports.
/proteogyver/cache Should live on a fast disk
/proteogyver/config should also be an external mount to sync parameters between proteogyver and pg_updater

## Creating and updating the database
To update the database, use the updater container
```
cd dockerfiles/pg_updater && docker compose up
```
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

#### Forcing an update
In some cases it is useful to force a full update of the database, even if the interval specified in the parameters.toml has not elapsed. In this case, add an environmental variable to the docker compose: FORCE_PG_DB_UPDATE: '1'

### Adding MS run data
See the [MS run data pre-analysis](#ms-run-data-pre-analysis) section of the install instructions.

### Adding new crapome or control sets:
Two files per set are needed:
1) The crapome/control overall table needs an update, and for that the control_sets.tsv or crapome_sets.tsv example file can be added to, and then put into the db_updates/crapome_sets or db_updates/control_sets directory.
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

As an example, utils/database_control_set_purger_examplescript.py is provided.
It will take as input the path to the database file, and delete ALL control sets from it. 


### Update logging
Updates will be logged to the update_log table.

## Building the docker images:
If you want to build the docker images locally, start by cloning the repo:
```
git clone https://github.com/varjolab/Proteogyver/
cd Proteogyver
```
### Build the Docker images and run the PG updater
These commands may need sudo depending on the system.
PG updater is used to generate a database. A small test database is provided, and that works well with the example files that can be downloaded from the PG interface. The test database contains scrambled data, and is thus not recommended as a base for a production database. Proper database should be built before real use.

#### Prerequisites:
- Download SAINTexpress from https://saint-apms.sourceforge.net/Main.html and place the **linux** executables into app/external/SAINTexpress:
  - Folder structure should contain:
    app/external/SAINTexpress/SAINTexpress-int
    app/external/SAINTexpress/SAINTexpress-spc
  - These will be registered as executables and put into the path of the PG container during the container creation (see dockerfile)
- IF you want to use the CRAPome repository data, download it from https://reprint-apms.org/?q=data
  - Afterwards, you need to format the data into a format usable by pg_updater, see [Creating and updating the database](#creating-and-updating-the-database) for details

#### Used external data
During database building, PG downloads data from several sources:
- Known interactions are downloaded from [IntACT](https://www.ebi.ac.uk/intact/home) and [BioGRID](https://thebiogrid.org/)
- Protein data is downloaded from [UniProt](https://www.uniprot.org/)
- Common contaminants are downloaded from [Global proteome machine](https://thegpm.org/), [MaxQuant](https://www.maxquant.org/), and a publication by Frankenfield et al., 2022 (PMID: 35793413).
- MS-microscopy data is from a previous publication (PMID: 29568061)
Some are included in the files already.

#### Build the main docker image.
**NOTE:** docker commands in particular may require superuser rights (sudo).
This should take around 15 minutes, but can take much longer, mostly due to R requirements. 
```
docker build -t pg_updater:1.5 -f dockerfiles/dockerfile_updater .
docker build -t proteogyver:1.5 -f dockerfiles/dockerfile .
```

## Rare use cases

### Embedding other websites as tabs within Proteogyver
To embed another website/tool within Proteogyver, add a line to embed_pages.tsv, and run the embedded_page_updater.py script. Preferably these will be things hosted on the same server, but this is not required. Current example is proteomics.fi (hosted externally). Keep in mind that most websites ban browsers from accessing if they are embedded in an html.Embed element.

### Adding custom tools as tabs to Proteogyver
Adding custom tools to Proteogyver is supported as pages in the app/pages folder. Here the following rules should be followed:
- Use dash.register_page to register the page (register_page(__name__, path='/YOUR_PAGE_NAME') )
- Use GENERIC_PAGE from element_styles.py for styling starting point. Mostly required from this is the offset on top of the page to fit the navbar

### Accessing the database from other tools
Other tools can access the database. Writes to the database should not require any specific precautions. However, please check that the database is not locked, and another transaction is not in progress. Other scenarios when one should not write to the database include if it is in the process of being backed up, or while the updater is actively running.

## How to cite

If you use ProteoGyver, a part of it, or tools based on it, please cite:
[Add citation information here]
