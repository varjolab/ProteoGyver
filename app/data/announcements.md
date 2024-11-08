## Announcements and abbreviated changelog
##### 2024.11.08: version 1.45 available on test server (:8060), 1.4a on stable (:8050)
- [x] Plenty of minor bugfixes
- [x] All column names are now transformed into str before any work is done to reduce the possibility of bugs. It is still not recommended to use only numbers as sample, group, or run names
- [x] Input file information now included in output outside of json files
- [x] Cache folder now lives as an external mount to reduce storage overhead
- [x] Added multiple resources to the other tools page
- [x] Multiple sample table and data file uploads were decided to not be implemented, due to implications to MBR during search etc. Interactomics workload standalone might happen at some point
- [x] Added a microscopy image colocalizer page. Only works with .lif images

##### 2024.04.23: version 1.3 alpha available on test server
- [x] Added a tickbox for the user to force PG todisplay supervenn instead of a heatmap when number of sample groups is over 6
- [x] Multiple bait uniprots now supported. In input files, the "Bait uniprot" -column can contain a list of bait uniprot IDs, separated by ;
- [x] Removed empty columns from export tables
- [x] PCA hover now shows sample name and color matches sample gorup
- [x] Can now choose groups shown in commonality plot, and can force the use of supervenn plot instead of heatmap, when more than 6 sample groups are present
- [x] Common proteins, such as keratin etc have been expanded to all organisms
- [x] Added empty option to control group dropdown
- [x] The user can now deselect all enrichments
- [x] More common protein groups have been added
- [x] Enrichment and MS-microscopy figures exported from interactomics workflow are no longer all just PCA
- [x] CV plot now only shows unimputed values
- [x] TIC graph now has a better x-axis upper bound
- [x] Volcano plot hover information now includes both gene name and protein ID, and other information
- [x] Added gene names to:
    - volcano output
    - volcano hover
- [x] Added sample names to: pca hover
- [x] Sidebar input section now folds away, and the table of contents is scrollable if needed
- [x] Differential abundance heatmaps are now generated into the volcano plots -directory
- [x] A SAINT "feature" that sets spectral counts of bait to 0, if the sample group name matches the bait gene name now has a workaround
- [x] Added multiple resources to the other tools page
##### 2024.02.15
- Proteogyver 1.2 now available on test server