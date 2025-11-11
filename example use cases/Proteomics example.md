# Example usecases for ProteoGyver using public datasets
This guide documents the end to end QC analysis of a public proteomics dataset.

## Software and data utilized
- Proteomics dataset used was from a (rprevious publication)[https://doi.org/10.1113/JP288104], provided by the corresponding author for quality control demonstration use. 
- 
Since the goal is not to demo how to use Dia-NN, default workflows will be used, with minimal consideration given to e.g. parameters of different mass spectrometers. As long as the library used is from the same organism, we will identify enough proteins to demo PG. In a real use scenario, parameters in Dia-NN in particular should be set according to the MS settings. 


## ProteoGyver
1) Upload the pg_matrix.tsv as the data file
2) Upload the sample table.tsv as the sample table
- Both indicators should turn green at this point to indicate successful preliminary data parse. 
3) Select proteomics from the workflow dropdown
4) Click begin analysis

