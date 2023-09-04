from pandas import DataFrame
from pandas import read_json as pd_read_json
from dash import html
from dash.dcc import Graph
from components.figures import before_after_plot, comparative_violin_plot, imputation_histogram, scatter, heatmaps, volcano_plot
from components import matrix_functions, summary_stats, stats

def na_filter(input_data_dict, filtering_percentage,figure_defaults, title:str=None) -> tuple:
    data_table: DataFrame = pd_read_json(
            input_data_dict['data tables']['intensity'],
            orient = 'split'
        )
    original_counts:DataFrame = matrix_functions.count_per_sample(
                data_table, input_data_dict['sample groups']['rev'])
    filtered_data: DataFrame = matrix_functions.filter_missing(
        data_table,
        input_data_dict['sample groups']['norm'],
        filtering_percentage,
    )
    filtered_counts:DataFrame = matrix_functions.count_per_sample(
                filtered_data, input_data_dict['sample groups']['rev'])
    return (
        html.Div(
            id='proteomics-na-filter-plot-div',
            children=[
                html.H4(id='proteomics-na-filter-header',children='Missing value filtering'),
                before_after_plot.make_graph(figure_defaults, original_counts, filtered_counts, 'proteomics-na-filter-plot', title=title)
            ]
        ),
        filtered_data.to_json(orient='split')
    )

def normalization(filtered_data_json, normalization_option, defaults, title:str=None) -> tuple:
    data_table: DataFrame = pd_read_json(filtered_data_json, orient='split')
    normalized_table: DataFrame = matrix_functions.normalize(data_table, normalization_option)
    sample_groups_rev: dict = {
        column_name: 'Before normalization' for column_name in data_table.columns
    }
    for column_name in normalized_table.columns:
        new_column_name:str = f'Normalized {column_name}'
        data_table[new_column_name] = normalized_table[column_name]
        sample_groups_rev[new_column_name] = 'After normalization'

    names: list
    comparative_data: dict
    names, comparative_data= summary_stats.get_comparative_data(
        data_table, sample_groups_rev
    )

    plot_colors: dict = {
        'sample groups': {
            'Before normalization': 'rgb(235,100,50)',
            'After normalization': 'rgb(50,100,235)'
        }
    }

    return (
        html.Div(
            id='proteomics-normalization-plot-div',
            children = [
                html.H4(id='proteomics-normalization-header',children='Normalization'),
                comparative_violin_plot.make_graph(
                    'proteomics-normalization-plot',
                    comparative_data, 
                    defaults, 
                    names=names, 
                    title=title, 
                    replicate_colors=plot_colors
                )
            ]
        ),
        normalized_table.to_json(orient='split')
    )
def imputation(filtered_data_json, imputation_option, defaults, title:str=None) -> tuple:

    data_table: DataFrame = pd_read_json(filtered_data_json, orient='split')
    imputed_table: DataFrame = matrix_functions.impute(data_table, imputation_option)

    return (
        html.Div(
            id='proteomics-imputation-plot-div',
            children = [
                html.H4(id='proteomics-imputation-header',children='Imputation'),
                imputation_histogram.make_graph(
                    data_table,
                    imputed_table,
                    defaults,
                    id_name = 'proteomics-imputation-plot',
                    title=title
                )
            ]
        ),
        imputed_table.to_json(orient='split')
    )

def pca(imputed_data_json: dict, sample_groups_rev: dict, defaults: dict) -> tuple:
    data_table: DataFrame = pd_read_json(imputed_data_json, orient='split')
    pc1: str
    pc2: str
    pca_result: DataFrame
    # Compute PCA of the data
    pc1, pc2, pca_result = matrix_functions.do_pca(data_table, sample_groups_rev, n_components = 2) 
    pca_result.sort_values(by=pc1,ascending=True,inplace=True)

    return (
        html.Div(
            id='proteomics-pca-plot-div',
            children = [
                html.H4(id='proteomics-pca-header',children='PCA'),
                scatter.make_graph(
                    'proteomics-pca-plot',
                    defaults,
                    pca_result,
                    pc1,
                    pc2,
                    'Sample group'
                ),
            ]
        ),
        pca_result.to_json(orient='split')
    )

def clustermap(imputed_data_json:dict, defaults: dict) -> tuple:
    """Draws a correltion clustergram figure from the given data_table.

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    id_name: name for the plot. will be used for the id of the returned dcc.Graph object.

    Returns: 
    dcc.Graph containing a dash_bio.Clustergram describing correlation between samples.
    """
    corrdata: DataFrame = pd_read_json(imputed_data_json, orient='split').corr()
    return (
        html.Div(
            id='proteomics-clustermap-plot-div',
            children = [
                html.H4(id='proteomics-clustermap-header',children='Sample correlation clustering'),
                Graph(
                    id= 'proteomics-clustermap-plot', 
                    config = defaults['config'],
                    figure = heatmaps.draw_clustergram(
                        corrdata, defaults
                    )
                ),
            ]
        ),
        corrdata.to_json(orient='split')
    )

def volcano_plots(imputed_data_json: dict, sample_groups: dict, comparisons: list, fc_thr: float, p_thr: float, defaults: dict) -> tuple:
    data: DataFrame = pd_read_json(imputed_data_json, orient='split')
    significant_data: DataFrame = stats.differential(data, sample_groups, comparisons, fc_thr=fc_thr, adj_p_thr=p_thr)
    significant_data.to_csv('volcano_plots_debug.tsv',sep='\t')
    graphs_div: html.Div = volcano_plot.generate_graphs(significant_data, defaults, fc_thr, p_thr)
    return (
        html.Div(
            id='proteomics-volcano-div',
            children = [
                html.H4(id='proteomics-volcano-header',children='Volcano plots'),
                graphs_div
            ]
        ),
        significant_data.to_json(orient='split')
    )