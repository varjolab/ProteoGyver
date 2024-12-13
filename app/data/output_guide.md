# Proteogyver QC and preliminary analysis output guide
## Description of output files
Output consists of files in the zip base directory, as well as several subdirectories. For figures, Html format retains figure legends, and the functionalities to zoom figures, see only specific sample groups etc. For examining and discussing Html format is recommended, while pdf is typically best for publication. Png can be used e.g. for presentations or, in the worst case, publication.

Some of the outputs produced depend on the workflow and other choices utilized.
### Base directory:
- Options used in analysis.txt lists all the input options chosen during ProteoGyver run
- README.html this file.
### Data:
This directory contains output and input data, and data for figures.
- Enrichment information contains whatever info the used APIs offer about each enrichment
#### Enrichment:
Tables of unfiltered enrichment results
#### Input data tables
Tables used for input
#### Interactomics/Proteomics data tables
Workflow specific data tables, e.g. Saint output (filtered and unfiltered) for interactomics.
#### Summary data
Tables for QC metrics
## Debug:
Contains information useful for debugging. 
## Enrichment figures:
Figures of all chosen enrichments
## MS microscopy:
MS microscopy polar plots for all baits, as well as a heatmap of all baits (if more than one bait)
## Proteomics figures:
Contains proteomics-specific figures in a few different formats each.
## QC Figures:
Quality control figures
## Volcano plots:
All generated volcano plots


