# ProteoGyver Batch Pipeline

The ProteoGyver batch pipeline allows you to run proteomics and interactomics analyses non-interactively, making it ideal for automated workflows, bulk processing, and reproducible analyses.

## Overview

The batch pipeline consists of two main scripts:

1. **`pipeline_batch.py`** - Direct command-line interface with arguments
2. **`pipeline_from_toml.py`** - Configuration file-based interface using TOML files

Both scripts support:
- **Proteomics workflow**: NA filtering, normalization, imputation, PCA, differential abundance analysis
- **Interactomics workflow**: SAINT analysis, network generation, enrichment analysis, MS-microscopy

## Usage

### Method 1: Command Line Interface

```bash
# Proteomics example
python app/pipeline_batch.py \
    --data "data/PG example files/proteinGroups.tsv" \
    --samples "data/PG example files/experimental_design.tsv" \
    --workflow proteomics \
    --outdir "results_proteomics" \
    --control "Control" \
    --fc 2.0 \
    --p 0.05

# Interactomics example
python app/pipeline_batch.py \
    --data "data/PG example files/interactomics_data.tsv" \
    --samples "data/PG example files/interactomics_design.tsv" \
    --workflow interactomics \
    --outdir "results_interactomics" \
    --additional-controls "VL GFP MAC3 10min AP" \
    --crapome-sets "Nesvilab" \
    --saint-bfdr 0.05 \
    --enrichments "GO_BP" "KEGG"
```

### Method 2: Configuration File Interface

1. Create a TOML configuration file (see examples: `batch_settings_proteomics_example.toml` and `batch_settings_interactomics_example.toml`)

2. Run the pipeline:
```bash
python app/pipeline_from_toml.py batch_settings.toml
```

## Command Line Arguments

### Common Arguments
- `--data`: Path to data file (required)
- `--samples`: Path to experimental design file (required)
- `--workflow`: Analysis type ("proteomics" or "interactomics", default: "proteomics")
- `--outdir`: Output directory (default: "batch_out")

### Proteomics-specific Arguments
- `--control`: Control group name for comparisons
- `--comparisons`: Path to custom comparisons file
- `--na`: NA filter percentage (default: 70)
- `--na-type`: NA filter type ("sample-group" or "sample-set", default: "sample-group")
- `--norm`: Normalization method ("no_normalization", "Median", "Quantile", "Vsn", default: "no_normalization")
- `--imp`: Imputation method ("QRILC", "minProb", "gaussian", "minValue", default: "QRILC")
- `--fc`: Fold change threshold (default: 1.5)
- `--p`: P-value threshold (default: 0.05)
- `--test`: Statistical test type ("independent" or "paired", default: "independent")

### Interactomics-specific Arguments
- `--uploaded-controls`: List of control sample names from your data
- `--additional-controls`: List of built-in control sets to use
- `--crapome-sets`: List of CRAPome datasets for filtering
- `--proximity-filtering`: Enable proximity filtering for controls
- `--n-controls`: Number of controls to keep (default: 3)
- `--saint-bfdr`: SAINT BFDR threshold (default: 0.05)
- `--crapome-pct`: CRAPome percentage threshold (default: 20)
- `--crapome-fc`: CRAPome fold change threshold (default: 2)
- `--rescue`: Enable rescue filtering
- `--enrichments`: List of enrichment analyses to perform

## Configuration File Format

See the example files for detailed TOML configuration structure:
- `batch_settings_proteomics_example.toml` - Proteomics workflow example
- `batch_settings_interactomics_example.toml` - Interactomics workflow example

## Output

The pipeline creates JSON files containing analysis results in the specified output directory:

### Common Outputs
- `01_data_dictionary.json` - Processed input data
- `02_replicate_colors.json` - Color assignments for plots
- `03_qc_artifacts.json` - Quality control analysis results

### Proteomics Outputs
- `10_na_filtered.json` - NA-filtered data
- `11_normalized.json` - Normalized data
- `12_imputed.json` - Imputed data
- `13_pca.json` - PCA analysis results
- `14_volcano.json` - Differential abundance results

### Interactomics Outputs
- `20_saint_dict.json` - SAINT input data
- `21_saint_output_raw.json` - Raw SAINT results
- `22_saint_with_crapome.json` - SAINT results with CRAPome data
- `23_saint_filtered.json` - Filtered SAINT results
- `24_network_elements.json` - Network visualization data
- `25_pca.json` - PCA analysis results
- `26_enrichment_data.json` - Enrichment analysis results (if requested)
- `27_msmic_data.json` - MS-microscopy analysis results

## Error Handling

The pipeline includes robust error handling:
- Validates input files and parameters
- Checks for required data types (spectral counts for interactomics)
- Handles SAINT analysis failures gracefully
- Provides informative error messages in the output JSON

## Requirements

- Valid `parameters.toml` file in the application directory
- Database file (`proteogyver.db`) with required reference data
- Input data files in supported formats (MaxQuant, FragPipe, etc.)
- For interactomics: SAINTexpressSpc executable (optional, will create dummy data if missing)

## Integration with Docker

This batch pipeline is designed to work with the ProteoGyver Docker container for fully automated analyses.
