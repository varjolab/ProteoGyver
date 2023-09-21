from dash.html import P

leg_dict: dict = {
    'qc':{
        'count-plot': 'Protein counts per sample. Replicates (both biological and technical) should have similar numbers of proteins. Darker area at the bottom of each bar is the proportion of contaminants identified inthe sample.',
        'coverage-plot': '',
        'reproducibility-plot': '',
        'missing_values-plot': '',
        'value_sum-plot': '',
        'value_mean-plot': '',
        'value_dist-plot': '',
        'shared_id-plot': '',
    },
    'proteomics': {
        'na_filter': 'Unfiltered and filtered protein counts in samples. Proteins identified in fewer than FILTERPERC percent of the samples of at least one sample group were discarded as low-quality identifications, contaminants, or one-hit wonders.',
        'comparative-violin-plot': '',
        'imputation': '',
        'pca': '',
        'clustermap': ''
    },
    'interactomics': {
        'pca': 'Spectral count -based PCA. Missing values have been imputed with zeroes. This is not a publication-worthy plot, but does indicate, how similar baits are to one another.',
        'saint-histo': 'Distribution of SAINT BFDR values. There should be a spike on the high end of the range, and a smaller one on the low end.',
        'filtered-saint-counts': 'Preys have been filtered based on the selected thresholds. Preys that passed through the filter in at least one bait were also rescued from other baits. Bait-bait interactions have been discarded.',
        'known': 'Known interactor preys (if any) are shown in a darker color on the bottom of each bar, previously unidentified HCIs make up the rest of the bar.'
    }
}

QC_LEGENDS: dict = {key: P(id=f'qc-legend-{key}', children=val) for key, val in leg_dict['qc'].items()}
PROTEOMICS_LEGENDS: dict = {key: P(id=f'proteomics-legend-{key}', children=val) for key, val in leg_dict['proteomics'].items()}
INTERACTOMICS_LEGENDS: dict = {key: P(id=f'interactomics-legend-{key}', children=val) for key, val in leg_dict['interactomics'].items()}

def volcano_plot_legend(sample, control, id_prefix) -> P: 
    return P(id=f'{id_prefix}-volcano-plot-{sample}-{control}', children=f'{sample} vs {control} volcano plot. Significant values are marked with the name and different color, and the lines represent significance thresholds in fold change and q-value dimensions.')