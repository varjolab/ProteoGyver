## Description of output files
Output consists of files in the zip base directory, as well as several subdirectories. For figures, Html format retains figure legends, and the functionalities to zoom figures, see only specific sample groups etc. For examining and discussing Html format is recommended, while pdf is typically best for publication. Png can be used e.g. for presentations or, in the worst case, publication.

Some of the outputs produced depend on the workflow and other choices utilized.
### Base directory:
- Options used in analysis.txt lists all the input options chosen during ProteoGyver run
- README.html this file.
### Data:
This directory contains output and input data, and data for figures
- Commonality data.json: data for shared identifications
- Enrichment.xlsx: Enrichment results in excel form, one sheet per enrichment.
- Enrichment information files: data from each enrichment API utilized.
- Figure data.xlsx: Data for multiple figures
- Input data tables.xlsx: Data tables that were uploaded to proteogyver
- Interactomics data tables.xlsx: step by step interactomics data tables
- MS microscopy results.xlsx: MS-microscopy data in matrix form
- SAINT input.xlsx: input files for SAINTexpress
- Proteomics data tables.xlsx: step by step proteomics data tables:
  - NA-Norm-imputed data: final data used for analysis. Values have been NA filtered, normalized, and imputed
  - NA-Normalized data: NA filtered and normalized data
  - NA filtered data: NA filtered data
- Reproducibility data.json: data for identification coverage
- Significant differences between sample groups:  Excel with differential abundance comparisons. Per comparison there is one sheet with only significant values, and one sheet with all comparisons.
- Summary data.xlsx: Data for QC figures describing summary stats, like protein counts and intensity sums etc.
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
