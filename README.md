# Proteogyver

Proteogyver is a comprehensive web-based platform for proteomics and interactomics data analysis. It provides tools for quality control, data visualization, and statistical analysis of mass spectrometry-based proteomics data.

## Features

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

### Additional Tools
- **MS Inspector**: Interactive visualization and analysis of MS performance through TIC graphs
- **Microscopy Image Colocalizer**: Analysis tool for .lif image files

### Microscopy Colocalizer Guide

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

### MS Inspector Guide

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

## Installation

### Docker Installation

Build the Docker image
```
docker build -t proteogyver .
```
Run the container
```
docker run -p 8050:8050 -p 8090:8090 proteogyver
```

## Usage

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
  - FragPipe (combined_prot.tsv)
  - DIA-NN (pg_matrix.tsv, report.tsv (discouraged due to size))
  - Generic matrix format

## Documentation

Detailed documentation for each module and workflow is available in the application:
- QC and analysis guide
- MS Analytics dashboard guide
- Windowmaker guide
- Output file documentation


## Advanced use cases

### Embedding other tools as tabs within Proteogyver
To embed another tool within Proteogyver, add a line to embed_pages.tsv, and run the embedded_page_updater.py script. Preferably these will be things hosted on the same server, but this is not required. Current examples include jupyterhub (locally hosted in the container), and proteomics.fi (hosted externally).

### Adding custom workflows to Proteogyver
Adding custom workflows is supported as pages in the app/pages folder. Here the following rules should be followed:
- Use dash.register_page to register the page (register_page(__name__, path='/YOUR_PAGE_NAME') )
- Use GENERIC_PAGE from element_styles.py for styling starting point. Mostly required from this is the offset on top of the page to fit the navbar

### Updating the database
To update the database, use the updater container (supported use case), or run the database_updater.py script.

## License

[Add your license information here]

## Citation

If you use Proteogyver or a part of it in your research, please cite:
[Add citation information here]
