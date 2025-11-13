# ProteoGyver User Guide
## QC and data analysis
### Data upload
Uploaded data consists of two files: the sample table, and the data table.
#### Sample table
Sample table contains information about your experiments: which sample is a part of what sample group. The file needs to have two columns for proteomics, three for interactomics: "Sample name" and "Sample group" are required for any workflow, while interactomics also needs "Bait uniprot" -column. It is highly encouraged to avoid using sample group names beginning with a number, or especially consisting only of numbers, due to lack of testing.

The bait uniprot column can have any name for the bait, but if a valid uniprot ID is supplied, it's used for mapping known interactions automatically. It is also necessary in order to discard the bait from the identifications in the interactomics analysis. Otherwise enrichment etc. will have skewed results.
##### Example Proteomics sample table
| Sample name | Sample group | 
| --------------- | ---------------- | 
| S1_replicate1 | S1 |
| S1_replicate2 | S1 |
| S1_replicate3 | S1 |
| B1_replicate1 | B1 |
| B1_replicate2 | B1 |
| B1_replicate3 | B1 |
##### Example Interactomics sample table

| Sample name | Sample group | Bait uniprot |
| --------------- | ---------------- | ------------- |
| S1_replicate1 | S1 | UniprotID1 |
| S1_replicate2 | S1 | UniprotID1 |
| S1_replicate3 | S1 | UniprotID1 |
| B1_replicate1 | B1 | UniprotID2 |
| B1_replicate2 | B1 | UniprotID2 |
| B1_replicate3 | B1 | UniprotID2 |

Sample names or column names may have the full path of the input raw file in them. Only the part after the last path separator will be utilized. 

#### Data table
Data table can be direct input from FragPipe or Dia-NN. From FragPipe the combined_prot.tsv table should be used, and from DIA-NN the pg_matrix.tsv table. 

Alternatively, you can use a generic matrix with either only spectral counts, or only intensities. In this case the columns should match the sample names (from sample table) exactly, and a column names "Protein.Group" should be in the table. If there is no Protein.Group -column, the first column of the data table is assumed to contain protein uniprot IDs.

Protein length column for should also exist, and named either PROTLEN, Protein Length , or Protein.Length. This is especially important for interactomics, where any protein which does not have a length either in the table, or in the limited ProteoGyver database is **discarded.**

The data table should **NOT** contain any duplicate proteins. Duplicates should not exist if you run your FragPipe, Dia-NN etc. well. If duplicates are found, only one will be kept, and its value will be the **SUM** of all duplicates. This can lead to **major data integrity problems**. 

### Running ProteoGyver
1. Upload data and sample table files to their corresponding places (either drag and drop, or click on the area and choose)
- If the sample table has a bait uniprot column, the indicator will turn blue, indicating that PG thinks you are uploading interactomics data.
2. Choose workflow from the dropdown
3. (optional) change figure theme, if you want
4. (optional) disable contaminant removal, if you know why you're doing that
5. Press the button to start the analysis

After initial QC plots have been generated, you may use the discard samples-button to get rid of any samples or groups of samples that you do not want to include in the final analysis.

#### QC results
**Most figures** support selecting a subregion for closer study, zooming in, or toggling on and off different groups. For example, clicking once on on the name of a sample group removes it from the plot, while clicking twice disables all other sample groups. Selecting a region on the plot zooms into that region, and double clicking restores the view.

After starting PG, the first thing you will see is a bunch of QC plots. These describe sample quality, and may indicate problems with your runs. Figure legends will elaborate on each figure. 
#### Workflow specific figures
These figures represent QC metrics specific to the workflow. In addition, some more specialized plots are described in the context of the appropriate workflow below.
#### Workflow-specific input
After checking through the QC results, the workflow specific results will appear. On the top of the page, there will be a new tab named after the workflow you chose.
##### Proteomics
Proteomics-specific settings consist:
- Missing value slider
- Imputation choice
- Normalization choice
- Control sample dropdown or comparison table upload
- Fold change threshold
- Adjusted p-value threshold

###### Missing value slider
On the top of the proteomics page, you can choose how to deal with missing values. Using the slider you can choose what percentage of samples of at least one sample group must have a value for a protein in order to keep it. The number of non-missing values per sample group required is the smallest whole number higher or equal to (sample group size * (percentage/100)).

e.g. with a 60 % threshold, at least two out of three samples in at least one sample group needs to have a value for a protein. For example, if we have a protein A and two sample groups with three samples each, with a 60 % setting, protein A needs to be found in two samples in either sample group 1, or sample group 2, or both. If it is found in only one sample in both of the sample groups, it will be discarded as a low-quality identification, one-hit wonder, or a random contamination. 

###### Example with 60 % filter (default):

| samples per group | Minimum required to pass 60% filter |
| ---------------- | ---------------- | 
| 1 | 1 |
| 2 | 2 |
| 3 | 2 |
| 4 | 3 |
| 5 | 3 |
| 6 | 4 |
| 7 | 5 |
| 8 | 5 |
| 9 | 6 |
| 10 | 6 |
| 11 | 7 |

Replicates should have more or less the same protein content, and therefore discarding one-hit-wonders ensures we only continue with the proteins we feel confident about identifying in the sample material, and not due to e.g. introduction into one replicate during the sample processing pipeline. Technical replicates should be nearly identical, and biological replicates extremely similar.
###### Imputation choice
Use QRILC. It's currently the best imputation method available on PG. Only change this is you know why you're doing it.
###### Normalization choice
Usually not needed, since MaxLFQ is already a normalized intensity value. However, median works well for most situations. If the post-normalization value distribution of your data looks like there are large differences between sample groups, try vsn or quantile.  
###### Comparisons
From the control dropdown, you can choose, which of your sample groups is the control against which all the other ones will be compared as volcano plots on the bottom of the page. The second alternative is to upload a comparison file:

| sample group | control group | 
| --------------- | ---------------- | 
| S1 | B1 |
| S2 | B1 |
| S1 | B2 |
You can also both choose a control group, and upload a file.
###### Thresholds
log2 fold change and adjusted p-value thresholds are usually fine with default values. If in volcano plots you see clouds of proteins near where thresholds intersect, consider tightening. Fold change threshold is presented on log2 scale, meaning a value of 1 means a two-fold increase compared to control, and -1 means halving the intensity compared to control.

For p-value adjustment FDR method (using multipletests from python statsmodels.stats.multitest package, method fdr_bh).
##### Interactomics
Input options consist of
- Controls from uploaded sample groups
- Inbuilt controls
- Crapomes
- Enrichments
- Rescue
- Nearest-n controls
After running SAINTexpress, you will be presented with more options:
- Saint BFDR Threshold
- Crapome Filtering percentage
- SPC fold change vs crapome threshold for rescue
###### Initial input
PG will attempt to identify controls from your sample groups. If any are found, it suggests to use them. Always double check. In addition, you may use inbuilt control sets, paying attention to the type of experiment, localization tag, etc. Crapomes offered are the same as control sets (same MS runs), but utilized differently for filtering of your data. Controls are used for SAINTexpress, while crapomes are utilized for crude filtering.

Rescue option means that if enabled, any interactors that have passed the selected filters and thresholds with ANY bait are classified as high-confidence interactors with ALL baits with which they are detected.

Choosing the nearest-n controls is currently not advisable.
###### Post-SAINTexpress input
Saint BFDR threshold can be thought of as a p-value threshold. 0.05 is the maximum you should consider, 0.01 is recommended. With 0.01 threshold, values of 0.01 or lower will be let through the filter.
Crapome filtering is calculated for each crapome. Protein is considered junk, if it's present in over the specified percentage of crapome samples **unless** the sample spectral count is higher than crapome average multiplied by the specified fold change.

Interactors that pass all of the filters are considered high-confidence interactions.

If bait uniprot is supplied, the bait is removed from the preys prior to any analysis. The bait will still be kept in the "Saint output" -sheet of the output excel.

## MS Analytics dashboard
The analytics dashboard is meant for inspection of MS performance over a series of runs. You can choose runs based on dates and sample types, or input a list of run ID numbers (the number in the beginning of the name of each run file) (this feature will be implemented soon hopefully). The TIC graph is generally what you would be looking at, but MSn graphs can be useful as well. The supplementary metrics, AUC, mean, and max intensity, can be used to judge the progress of a sample series, and things like gradual changes in MS performance.

## Colocalizer
Colocalizer is a tool for quickly generating colocalization heatmaps from confocal microscopy images. Currently only .LIF files are supported. Once a file is uploaded, you can choose which z-slice is utilized, and what channels should be used. The tool offers both multiplication and addition based generation of the colocalization image. For nearly all applications, multiplication is preferred. Using this mode, bright pixels in the resulting colocalization image = pixels that had a high value in both of the channel images of that z-slice.