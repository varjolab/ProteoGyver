from dash.html import P

leg_dict: dict = {
    'qc': {
        'count-plot': 'Protein counts per sample. Replicates (both biological and technical) should have similar numbers of proteins. Darker area at the bottom of each bar is the proportion of contaminants identified inthe sample.',
        'common-protein-plot': 'Some proteins are often copurified due to sample handling or workflow in general. These are expected, and if samples were handled in a robust way, the signal we get from these proteins should be rather stable across all samples and sample groups. This plot shows the sum of the signal (intensity by default, spectral count if intensity is not available) of each protein group or combination of groups. :::!NOTE!:::The protein groupings shown is very crude: uniprot was searched with the name of the common protein group, and reviewed protein IDs were downloaded. No further processing was done. This means that proteins might be only tangentially related to the specified function. This will be improved in the future.',
        'coverage-plot': 'Most proteins should be identified in all of your samples. In interactomics, we expect to see relatively high values on both ends of the scale. Common contaminants/interactors on the left, and highly-specific interactors on the right. In proteomics, we expect a peak on the left, and a descending series towards the right.',
        'reproducibility-plot': 'The plot describes how far from the average of the sample group values of individual runs are. With some exceptions, all values should be very nearly identical between biological and especially technical replicates, regardless of what kind of an experiment is happening.',
        'missing_values-plot': 'Missing values are mostly a problem in proteomics and phosphoproteomics workflows. In interactomics, depending on your specific bait set, we can expect to see either very few or very many of them. In particular, if you include controls, e.g. GFP, in your analysis set (as you should), it will inevitably raise the number of missing values. Missing values in interactomics only matter, if comparing protein abundance between different baits.',
        'value_sum-plot': 'The total intensity sum (or sum of spectral counts, depending on the data) should be roughly equal across sample groups for comparable data.',
        'value_mean-plot': 'The mean intensity (or spectral counts, depending on the data) should be roughly equal across sample groups for comparable data.',
        'value_dist-plot': 'Value distribution of the identifications. The specifics can be different across sample groups, but especially replicates should look very similar.',
        'shared_id-plot-hm': 'Shared identifications across samples. The color corresponds to the number of shared identifications between row and column divided by the number of unique proteins identified across the two sample groups.',
        'shared_id-plot-sv': 'Supervenn diagram of the shared identifications. One row is one sample group. Numbers on the right show the number of identifications (no filtering) in each sample group. One figure "column" shows how many proteins are shared by each combination of sample gorups, and the number on the bottom shows the number of proteins in the shared group, while the number on top shows the number of sample groups in the shared group.',
        'tic': '''Chromatogram of the input MS runs. Check your curves, and consider if they're comparable. TIC is usually enough to identify problems, but you should also check the base peak.'''
    },
    'proteomics': {
        'na_filter': 'Unfiltered and filtered protein counts in samples. Proteins identified in fewer than FILTERPERC percent of the samples of at least one sample group were discarded as low-quality identifications, contaminants, or one-hit wonders.',
        'normalization': 'Runs were normalized with the chosen method, in order to reduce the effects of run-to-run variation, stemming from e.g. LC performance, MS performance, and especially variation in protein concentration measurements.',
        'missing-in-other-samples': 'This plot shows the distribution of intensities for all of the proteins, that had missing values in any sample, and intensities of proteins without any missing values. It should be very similar to the imputation histogram. Proteins at the low-end of the value range are most likely low abundance proteins that are either under detection threshold or missing due to expectable losses. Proteins with missing values on the high end of the range are most likely either differentially expressed, errors in sample prep, or chance events.',
        'imputation': 'Missing values were imputated using the chosen imputation method. This is important, because otherwise calculating p-values for detected changes is impossible. With missing values, it means that either the protein is not in the sample, or the level in the sample is below the detection method. We do not know which of these is the case, so we assume their values are around the low end of the detection range. Note that protein can be missing from a sample for reasons related to both biology and sample handling, as well as sampling method etc.',
        'pca': 'PCA of the individual sample runs. Sample groups should cluster nicely. If there are large gaps between replicates, check how much of each axis explains of the variation between the samples, and how similar the samples are otherwise (TIC, protein counts, etc).',
        'clustermap': 'Correlation is another metric for how similar samples are. Similarly to above, samples should form clusters of each sample group.',
        'pertubation-bar': 'Z-score based pertubation score against CONTROLSTRING. Based on the molecular degree of pertubation R-package, but not quite the same. Z-score is calculated based on the mean of the control group, instead of the full sample panel. Higher values mean samples that are more different from the control group. Calculated using BARVALS.',
        'pertubation-bar-2': 'Z-score based pertubation score against CONTROLSTRING. Calculated using BARVALS.',
        'pertubation-matrix': 'Z-score based pertubation score against CONTROLSTRING for TOPN most different proteins across all samples.',
        'cv': 'Coefficient of variation is the standard deviation of a protein divided by the mean of the protein across each sample group. One protein therefore is represented by as many entries as there are sample groups in this plot. Calculating CV per sample group instead of across all sample groups is better, when we are interested in how stable the abundance is within-group. Given that each sample group consists of replicates, variance should be minimal. In this plot, percentage of CV is plotted against the protein abundance (raw intensity values, not log2 transformed like elsewhere). CVs under 20% qualify as good quality data, and most of them should be under 10%, however be cautious when interpreting CVs alone (https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/pmic.202300090).'
    },
    'interactomics': {
        'pca': 'Spectral count -based PCA. Missing values have been imputed with zeroes. This is not a publication-worthy plot, but does indicate how similar baits are to one another.',
        'saint-histo': 'Distribution of SAINT BFDR values. There should be a spike on the high end of the range, and a smaller one on the low end.',
        'filtered-saint-counts': 'Preys have been filtered based on the selected thresholds. BFDR threshold means anything with the selected value or lower passes the filter. Preys that passed through the filter in at least one bait were $RESCUE from other baits. Bait-bait interactions have been discarded, IF bait uniprots are in the input file.',
        'known': 'Known interactor preys (if any) are shown in a darker color on the bottom of each bar, previously unidentified HCIs make up the rest of the bar.',
        'ms-microscopy-single': 'MS microscopy results for BAITSTRING. Values are not "real", but instead 100 = best match per bait, and the rest are scaled appropriately based on how much of shared signal originates from Preys specific to each localization.',
        'ms-microscopy-all': 'MS microscopy results for all baits. Values are not "real", but instead 100 = best match per bait, and the rest are scaled appropriately based on how much of shared signal originates from Preys specific to each localization.',
    }
}

QC_LEGENDS: dict = {key: P(id=f'qc-legend-{key}', children=val)
                    for key, val in leg_dict['qc'].items()}
PROTEOMICS_LEGENDS: dict = {key: P(id=f'proteomics-legend-{key}', children=val)
                            for key, val in leg_dict['proteomics'].items()}
INTERACTOMICS_LEGENDS: dict = {key: P(id=f'interactomics-legend-{key}', children=val)
                               for key, val in leg_dict['interactomics'].items()}

def leg_rep(legend, replace, rep_with) -> P:
    return P(id=legend.id, children = legend.children.replace(replace, rep_with))

def volcano_plot_legend(sample, control, id_prefix) -> P:
    return P(id=f'{id_prefix}-volcano-plot-{sample}-{control}', children=f'{sample} vs {control} volcano plot. Significant values are marked with the name and different color, and the lines represent significance thresholds in fold change and q-value dimensions.')

def saint_legend(rescued: bool) -> P:
    return leg_rep(INTERACTOMICS_LEGENDS['filtered-saint-counts'], '$RESCUE', 'rescued' if rescued else 'not rescued')

def enrichment_legend(clean_enrichment_name, enrichment_name, fc_threshold, fc_col, p_value_name, p_threshold):
    return P(id=f'{clean_enrichment_name}-enrichment-legend', children=f'{enrichment_name} enrichment using {fc_col} filter of {fc_threshold} and {p_value_name} filter of {p_threshold}.')

def volcano_heatmap_legend(control, id_prefix) -> P:
    return P(id=f'{id_prefix}-volcano-heatmap-{control}', children=f'All comparisons vs vs {control} volcano plot. Only significantly different proteins shown')
