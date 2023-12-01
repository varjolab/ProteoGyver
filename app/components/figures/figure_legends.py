from dash.html import P

leg_dict: dict = {
    'qc': {
        'count-plot': 'Protein counts per sample. Replicates (both biological and technical) should have similar numbers of proteins. Darker area at the bottom of each bar is the proportion of contaminants identified inthe sample.',
        'coverage-plot': 'Most proteins should be identified in all of your samples. In interactomics, these are (depending on the bait set) common contaminants, and possibly common interactors. In interactomics, we expect to see relatively high values on both ends of the scale. Common contaminants/interactors on the left, and highly-specific interactors on the right. In proteomics, we expect a peak on the left, and a descending series towards the right.',
        'reproducibility-plot': 'The plot describes how far from the average of the sample group values of individual runs are. With some exceptions, all values should be very nearly identical between biological and especially technical replicates, regardless of what kind of an experiment is happening.',
        'missing_values-plot': 'Missing values are mostly a problem in proteomics and phosphoproteomics workflows. In interactomics, depending on your specific bait set, we can expect to see either very few or very many of them. In particular, if you include controls, e.g. GFP, in your analysis set (as you should), it will inevitably raise the number of missing values. Missing values in interactomics only matter, if comparing protein abundance between different baits.',
        'value_sum-plot': 'The total intensity sum (or sum of spectral counts, depending on the data) should be roughly equal across sample groups for comparable data.',
        'value_mean-plot': 'The mean intensity (or spectral counts, depending on the data) should be roughly equal across sample groups for comparable data.',
        'value_dist-plot': 'Value distribution of the identifications. The specifics can be different across sample groups, but especially replicates should look very similar.',
        'shared_id-plot': 'Shared identifications across samples. The color corresponds to the number of shared identifications between row and column divided yb the number of unique proteins identified across the two sample groups.',
    },
    'proteomics': {
        'na_filter': 'Unfiltered and filtered protein counts in samples. Proteins identified in fewer than FILTERPERC percent of the samples of at least one sample group were discarded as low-quality identifications, contaminants, or one-hit wonders.',
        'comparative-violin-plot': '',
        'imputation': '',
        'pca': '',
        'clustermap': ''
    },
    'interactomics': {
        'pca': 'Spectral count -based PCA. Missing values have been imputed with zeroes. This is not a publication-worthy plot, but does indicate how similar baits are to one another.',
        'saint-histo': 'Distribution of SAINT BFDR values. There should be a spike on the high end of the range, and a smaller one on the low end.',
        'filtered-saint-counts': 'Preys have been filtered based on the selected thresholds. Preys that passed through the filter in at least one bait were also rescued from other baits. Bait-bait interactions have been discarded, IF bait uniprots are in the input file.',
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


def enrichment_legend(clean_enrichment_name, enrichment_name, fc_threshold, fc_col, p_value_name, p_threshold):
    return P(id=f'{clean_enrichment_name}-enrichment-legend', children=f'{enrichment_name} enrichment using {fc_col} filter of {fc_threshold} and {p_value_name} filter of {p_threshold}.')
