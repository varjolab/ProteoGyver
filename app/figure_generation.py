import base64
import io
from typing import Any
from plotly import express as px
import plotly.graph_objects as go
import numpy as np
from utilitykit import plotting
from dash import dcc
import data_functions
import pandas as pd
from statsmodels.stats import multitest
from scipy.stats import ttest_ind
#import scipy.sparse as sp
from sklearn import manifold, decomposition
import plotly.graph_objs as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from dash import html
from sklearn.pipeline import make_pipeline
import dash_bio
from matplotlib import pyplot as plt
import matplotlib as mpl
from supervenn import supervenn as svenn

def add_replicate_colors(data_df, column_to_replicate) -> None:

    need_cols: int = list(
        {
            column_to_replicate[sname] for sname in
            data_df.index.unique()
            if sname in column_to_replicate
        }
    )
    colors: list = plotting.get_cut_colors(number_of_colors=len(need_cols))
    colors = plotting.cut_colors_to_hex(colors)
    colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
    color_column: list = []
    for sn in data_df.index.values:
        color_column.append(colors[column_to_replicate[sn]])
    data_df.loc[:, 'Color'] = color_column

def protein_coverage(data_table) -> dcc.Graph:
    return dcc.Graph(
        id='protein-coverage-{title}',
        figure=bar_plot(
                pd.DataFrame(data_table.notna()\
                .astype(int)\
                .sum(axis=1)\
                .value_counts(),columns=['Identified in # samples'])
                ,'Protein coverage',color=False)
            )

def bar_plot(value_df, title, y=0,color:bool=True) -> px.bar:
    if color:
        colorval: str = 'Color'
        cmapval: str = 'identity'
    else:
        colorval = None
        cmapval = None
    figure: px.bar = px.bar(
        value_df,
        x=value_df.index,  # 'Sample name',
        y=value_df.columns[y],
        #height=500,
        #width=750,
        title=title,
        color=colorval,
        color_discrete_map=cmapval
    )
    figure.update_xaxes(type='category')
    return figure


def comparative_violin_plot(sets: list, names: list = None, id_name: str = None, title: str = None, colors: list = None, barplot: bool = True) -> dcc.Graph:
    if id_name is None:
        id_name: str = 'comparative-violin-plot'
    if isinstance(colors, list):
        assert ((len(sets) == len(colors)) | (colors is None)
                ), 'Length of "sets" should be the same as length of "colors"'
    if isinstance(names, list):
        assert ((len(sets) == len(names))
                ), 'Length of "sets" should be the same as length of "names"'
    else:
        names: list = []
        for i in range(0, len(sets)):
            names.append(f'Set {i+1}')
    plot_data: list = np.array([])
    plot_legend: list = [[], []]
    for i, data_frame in enumerate(sets):
        for col in data_frame.columns:
            plot_data = np.append(plot_data, data_frame[col].values)
            plot_legend[0].extend([names[i]]*data_frame.shape[0])
            plot_legend[1].extend([f'{col} {names[i]}']*data_frame.shape[0])
    plot_df: pd.DataFrame = pd.DataFrame(
        {
            'Values': plot_data,
            'Column': plot_legend[1],
            'Name': plot_legend[0]
        }
    )
    if title is None:
        title: str = ''
    return dcc.Graph(id=id_name, figure=px.violin(
        plot_df, 
        y='Values', 
        x='Column', 
        color='Name', 
        box=True, 
        title=title, 
        color_discrete_sequence=colors,
        #height=500,
        #width=750,
        )
    )


def distribution_figure(data_table, color_dict, sample_groups, title: str = 'Value distribution', log2_transform=True) -> dcc.Graph:

    rev_sample_groups: dict = {}
    for k, v in sample_groups.items():
        if v not in rev_sample_groups:
            rev_sample_groups[v] = []
        rev_sample_groups[v].append(k)
    names: list = sorted(list(rev_sample_groups.keys()))
    colors: list = [color_dict[rev_sample_groups[k][0]] for k in names]
    data_frames: list = []
    for sample_group in names:
        data_frames.append(data_table[rev_sample_groups[sample_group]])
    id_str: str = 'value-distribution-figure'
    return(comparative_violin_plot(data_frames, names=names, title=title, id_name=id_str, colors=colors, barplot=False))


def before_after_plot(before: pd.Series, after: pd.Series, rev_sample_groups: dict, title: str = None) -> dcc.Graph:
    data: list = [['Before or after', 'Count', 'Sample']]
    for i in before.index:
        if i in after.index:
            data.extend([
                ['before', before[i], i],
                ['after', after[i], i]
            ])
    if title is None:
        title: str = ''
    dataframe: pd.DataFrame = pd.DataFrame(data=data[1:], columns=data[0])
    return dcc.Graph(
        id='before-and-after-na-filter-figure',
        figure=px.bar(dataframe,
                      x='Sample',
                      y='Count',
                      color='Before or after',
                      barmode='group',
                      title=title)
    )


def imputation_histogram(non_imputed, imputed, title: str = None) -> dcc.Graph:
    #x,y = sp.coo_matrix(non_imputed.isnull()).nonzero()
    non_imputed = non_imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'})
    imputed = imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'}).rename(columns={'value': 'log2 value'})
    imputed['Imputed'] = non_imputed['value'].isna()
    if title is None:
        title = 'Imputation'
    figure = px.histogram(
        imputed, x='log2 value', marginal='violin', color='Imputed', title=title
    )
    return dcc.Graph(id='missing-value-histogram', figure=figure)


def sum_value_figure(sum_data) -> dcc.Graph:
    sum_figure = bar_plot(sum_data, title='Value sum per sample')
    return dcc.Graph(id='value-sum-figure', figure=sum_figure)


def avg_value_figure(avg_data) -> dcc.Graph:
    avg_figure = bar_plot(avg_data, title='Value mean per sample')
    return dcc.Graph(id='value-sum-figure', figure=avg_figure)


def missing_figure(na_data) -> dcc.Graph:
    na_figure: px.bar = bar_plot(na_data, title='Missing values per sample')
    return dcc.Graph(id='protein-count-figure', figure=na_figure)


def protein_count_figure(count_data) -> dcc.Graph:
    count_figure: px.bar = bar_plot(count_data, title='Proteins per sample')
    return dcc.Graph(id='protein-count-figure', figure=count_figure)


def volcano_plots(data_table, sample_groups, control_group) -> list:
    control_cols: list = sample_groups[control_group]
    volcanoes = []
    for group_name, group_cols in sample_groups.items():
        if group_name == control_group:
            continue
        volcanoes.append(
            dcc.Graph(
                id=f'volcano-{group_name}-vs-{control_group}',
                figure=volcano_plot(
                    data_table,
                    group_name,
                    control_group,
                    group_cols,
                    control_cols
                )
            )
        )
    return volcanoes


def volcano_plot(data_table, sample_name, control_name, sample_columns, control_columns, p_threshold: float = 0.05, fc_threshold: float = 1, fc_axis_min_max:float=2) -> dcc.Graph:

    # Calculate log2 fold change for each protein between the two sample groups
    log2_fold_change: pd.Series = np.log2(data_table[sample_columns].mean(
        axis=1) / data_table[control_columns].mean(axis=1))

    # Calculate the p-value for each protein using a two-sample t-test
    p_value: float = data_table.apply(lambda x: ttest_ind(
        x[sample_columns], x[control_columns])[1], axis=1)

    # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
    _: Any
    p_value_adj: np.ndarray
    _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

    # Create a new dataframe containing the fold change and adjusted p-value for each protein
    result: pd.DataFrame = pd.DataFrame(
        {'fold_change': log2_fold_change, 'p_value_adj': p_value_adj, 'p_value': p_value})
    result['Name'] = data_table.index.values
    result['significant'] = ((result['p_value_adj'] < p_threshold) & (
        abs(result['fold_change']) >= 1))
    result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
    result['Highlight'] = [row['Name'] if row['significant']
                           else '' for _, row in result.iterrows()]
    # Draw the volcano plot using plotly express
    fig = px.scatter(result, x='fold_change', y='p_value_adj_neg_log10',
                     title=f'{sample_name} vs {control_name}', color='significant', text='Highlight')

    # Set yaxis properties
    p_thresh_val: float = -np.log10(p_threshold)
    pmax: float = max(result['p_value_adj_neg_log10'].max(), p_thresh_val)+0.5
    fig.update_yaxes(title_text='-log10 (q-value)', range=[0, pmax])
    # Set the x-axis properties
    fcrange: float = max(max(abs(result['fold_change'])), fc_threshold)
    if fcrange < fc_axis_min_max:
        fcrange = fc_axis_min_max
    fcrange += 0.25
    fig.update_xaxes(title_text='Fold change', range=[-fcrange, fcrange])

    # Add vertical lines indicating the significance thresholds
    fig.add_shape(type='line', x0=-fc_threshold, y0=0, x1=-
                  fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
    fig.add_shape(type='line', x0=fc_threshold, y0=0,
                  x1=fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
    # And horizontal line:
    fig.add_shape(type='line', x0=-fcrange, y0=p_thresh_val,
                  x1=fcrange, y1=p_thresh_val, line=dict(width=2, dash='dot'))

    # Show the plot
    return fig


def pca_plot(data_table: pd.DataFrame, rev_sample_groups: dict, n_components: int = 2, figname: str = None) -> dcc.Graph:
    """Untested code"""
    # Compute PCA of the data
    data_df: pd.DataFrame = data_table.T
    pca = decomposition.PCA(n_components=n_components)
    pca_result = pca.fit_transform(data_df)
    data_df['PCA one'] = pca_result[:, 0]
    data_df['PCA two'] = pca_result[:, 1]
    data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
    data_df['Sample name'] = data_df.index
    if figname is None:
        figname: str = ''
    else:
        figname = '-' + figname
    fig = px.scatter(
        data_df, x='PCA one', y='PCA two', title=f'PCA{figname}',color=data_df['Sample group'],
        text='Sample name')
    fig.update_traces(marker_size=15)
    return dcc.Graph(figure=fig, id=f'pca-plot{figname}')


def t_sne_plot(data_table: pd.DataFrame, rev_sample_groups: dict, perplexity: int = 15, n_components: int = 2, iterations: int = 5000, figname: str = None) -> dcc.Graph:
    """Untested code"""
    data_df: pd.DataFrame = data_table.T
    perplexity: int = min(data_df.shape[0]-1, perplexity)
    tsne = manifold.TSNE(n_components=n_components, verbose=0,
                         perplexity=perplexity, n_iter=iterations)
    tsne_results = tsne.fit_transform(data_df)
    data_df['t-SNE one'] = tsne_results[:, 0]
    data_df['t-SNE two'] = tsne_results[:, 1]
    data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
    data_df['Sample name'] = data_df.index
    if figname is None:
        figname: str = ''
    else:
        figname = '-' + figname
    fig = px.scatter(
        data_df, x='t-SNE one', y='t-SNE two', title=f't-SNE{figname}',color=data_df['Sample group'],
        text='Sample name')
    fig.update_traces(marker_size=15)
    return dcc.Graph(figure=fig, id=f'tsne-plot{figname}')

def coefficient_of_variation(row: pd.Series) -> float:
    return (row.std()/row.mean())*100


def df_coefficient_of_variation(data_table: pd.DataFrame) -> pd.DataFrame:
    new_df: pd.DataFrame = pd.DataFrame(index=data_table.index)
    new_df['%CV'] = data_table.apply(coefficient_of_variation, axis=1)
    new_df['mean intensity'] = data_table.mean(axis=1)
    return new_df

def coefficient_of_variation_plot(data_table: pd.DataFrame, plotname: str = 'coefficient-of-variation', title:str = None) -> dcc.Graph:
    data_table: pd.DataFrame = df_coefficient_of_variation(data_table)
    col1, col2 = data_table.columns[:2]
    data_table = data_table.sort_values(by=col2,ascending=True,inplace=False)
    fig = px.scatter(data_table, x=col2, y=col1, title=title)

    plotx: pd.Series = data_table['mean intensity'].values[:,np.newaxis]
    modely: pd.Series = data_table['%CV']
    model = make_pipeline(SplineTransformer(n_knots=4, degree=2), Ridge(alpha=1e-3))
    model.fit(plotx,modely)
    y_plot: pd.Series = pd.Series(model.predict(plotx))
    plot_df: pd.DataFrame = pd.DataFrame({'x': plotx.flatten(),'y': y_plot})
    fig2 = px.line(plot_df, x='x', y='y')
    if title is None:
        title = ''
    fig = go.Figure(data=fig.data + fig2.data)
    fig.update_layout(
        title = 'Coefficients of variation',
        xaxis_title = 'Log2 normalized imputed mean intensity',
        yaxis_title = '%CV'
    )
    return dcc.Graph(figure=fig, id=f'scatter-{plotname}')


def clustergram(plot_data:pd.DataFrame, color_map: list = None, **kwargs):
    if color_map is None:
        color_map = [
            [0.0, '#FFFFFF'],
            [1.0, '#EF553B']
        ]
    return dash_bio.Clustergram(
        data=plot_data,
        column_labels=list(plot_data.columns.values),
        row_labels=list(plot_data.index),
        #height=800,
       # width=750,
        color_map= color_map,
        link_method = 'average',
        **kwargs
    )

def correlation_clustermap(data_table: pd.DataFrame, plotname: str = None) -> dcc.Graph:
    if plotname is None: 
        plotname: str = 'correlation-clustermap'
    corr_df: pd.DataFrame = data_table.corr()
    fig = clustergram(
        corr_df,
        )
    #fig.update_zaxes(range=[0,1])
    # kokeile vielä layoutin kautta zmin ja zmax parametrejä
    # tai fig.update_layout(coloraxis=dict(cmax=6, cmin=3))
    return dcc.Graph(figure=fig,id=plotname)

def full_clustermap(data_table:pd.DataFrame,plotname:str = None) -> dcc.Graph:
    if plotname is None: 
        plotname: str = 'full-clustermap'
    fig = clustergram(
        data_table,
        hidden_labels=['row'],
        #The cluster parameter should be "column", but that does not work. According to source code, "col" is the correct usage, but it might get changed to reflect documentation later.
        cluster='col',
        center_values=False
        )
    return dcc.Graph(figure=fig,id=plotname)

def supervenn(data_table: pd.DataFrame, rev_sample_groups: dict) -> html.Img:
    """Draws a super venn plot for the input data table.

    See https://github.com/gecko984/supervenn for details of the plot.
    Parameters:
    sets: dictionary containing the data sets. {setname1: set1(), setname2: set2()}
    figure_name: name for the figure title, as well as saved file
    save_figure: whether the figure should be saved as pdf
    out_dir: place the saved figure in this directory

    Returns:
    tuple of (fig, axes), or None if either too few or too many sets were given.
    """    
    group_sets: dict = {}
    for column in data_table.columns:
        col_proteins: set = set(data_table[[column]].dropna().index.values)
        group_name: str = rev_sample_groups[column]
        if group_name not in group_sets:
            group_sets[group_name] = set()
        group_sets[group_name] |= col_proteins

    # Buffer for use
    buffer: io.BytesIO = io.BytesIO()
    fig: mpl.figure
    axes: mpl.Axes
    fig, axes = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)
    
    plot_sets: list = []
    plot_setnames: list = []
    for set_name, set_proteins in group_sets.items():
        plot_sets.append(set(set_proteins))
        plot_setnames.append(set_name)
    svenn(
        plot_sets,
        plot_setnames,
        ax=axes,
        rotate_col_annotations=True,
        col_annotations_area_height=1.2,
        widths_minmax_ratio=0.1
        )
    plt.xlabel('Shared proteins')
    plt.ylabel('Sample group')
    plt.savefig(buffer, format = "png")
    plt.close()
    data: str = base64.b64encode(buffer.getbuffer()).decode("utf8") # encode to html elements
    buffer.close()
    return html.Img(id='supervennfigure',src=f'data:image/png;base64,{data}')


def missing_clustermap(data_table,plotname:str = None) -> dcc.Graph:
    if plotname is None:
        plotname: str = 'missing-value-clustermap'
    figure = clustergram(
        data_table.notna().astype(int),
        cluster='col',
        hidden_labels=['row'],
        color_map= [
            [0.0, '#000000'],
            [1.0, '#FFFFFF']
        ],
        center_values=False
    )
    figure.update_layout(title='Missing values')
    return dcc.Graph(
        figure = figure,
        id=plotname
    )
    

""" NEeds testing"""


def get_volcano_df(data_table: pd.DataFrame, sample_columns, control_columns, p_th) -> pd.DataFrame:
     # Calculate log2 fold change for each protein between the two sample groups
    log2_fold_change: pd.Series = np.log2(data_table[sample_columns].mean(
        axis=1) / data_table[control_columns].mean(axis=1))

    # Calculate the p-value for each protein using a two-sample t-test
    p_value: float = data_table.apply(lambda x: ttest_ind(
        x[sample_columns], x[control_columns])[1], axis=1)

    # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
    _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

    # Create a new dataframe containing the fold change and adjusted p-value for each protein
    result: pd.DataFrame = pd.DataFrame(
        {'fold_change': log2_fold_change, 'p_value_adj': p_value_adj, 'p_value': p_value})
    result['Name'] = data_table.index.values
    #result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
    return result

def volcano_plot2(data_table) -> html.Div: 
    
    html.Div([
    'log2FC:',
    dcc.RangeSlider(
        id='range-slider',
        min=-3,
        max=3,
        step=0.05,
        marks={i: {'label': str(i)} for i in range(-3, 3)},
        value=[-0.5, 1]
    ),
    html.Br(),
    html.Div(
        dcc.Graph(
            id='graph',
            figure=dashbio.VolcanoPlot(
                dataframe=df
            )
        )
    )
])

def volcano_plots2(data_table, sample_groups, control_group) -> list:
    control_cols: list = sample_groups[control_group]
    volcanoes = []
    for group_name, group_cols in sample_groups.items():
        if group_name == control_group:
            continue
        volcanoes.append(
            dcc.Graph(
                id=f'volcano-{group_name}-vs-{control_group}',
                figure=volcano_plot(
                    data_table,
                    group_name,
                    control_group,
                    group_cols,
                    control_cols
                )
            )
        )
    return volcanoes