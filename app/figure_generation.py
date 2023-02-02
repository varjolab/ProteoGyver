import base64
import io
import numpy as np
import pandas as pd
from typing import Any
from plotly import express as px
import plotly.graph_objects as go
from dash import html, dcc
import dash_bio
from utilitykit import plotting
from statsmodels.stats import multitest
from scipy.stats import ttest_ind
from sklearn import manifold, decomposition
import matplotlib as mpl
from matplotlib import pyplot as plt
from supervenn import supervenn as svenn


def add_replicate_colors(data_df: pd.DataFrame, group_dict: dict) -> None:
    """Adds "Color" column to long-format data_df based on group_dict dictionary.
    Each group will get its own color, and data_df.index should correspond to sample_names, which are found in group_dict as keys.
    
    Parameters:
    data_df: long data frame with groups in index
    group_dict: dictionary of {sample_name: group_name}
    """
    need_cols: int = list(
        {
            group_dict[sname] for sname in
            data_df.index.unique()
            if sname in group_dict
        }
    )
    colors: list = plotting.get_cut_colors(number_of_colors=len(need_cols))
    colors = plotting.cut_colors_to_hex(colors)
    colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
    color_column: list = []
    for sample_name in data_df.index.values:
        color_column.append(colors[group_dict[sample_name]])
    data_df.loc[:, 'Color'] = color_column


def comparative_violin_plot(sets: list, names: list = None, id_name: str = None, title: str = None, colors: list = None) -> dcc.Graph:
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
        # height=500,
        # width=750,
    )
    )


def distribution_figure(data_table, color_dict, sample_groups, title: str = 'Value distribution') -> dcc.Graph:

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
    return(comparative_violin_plot(data_frames, names=names, title=title, id_name=id_str, colors=colors))


def before_after_plot(before: pd.Series, after: pd.Series, title: str = None) -> dcc.Graph:
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


def histogram(data_table: pd.DataFrame, x_column: str, title: str) -> dcc.Graph:
    figure: go.Figure = px.histogram(data_table, x=x_column, title=title)
    return dcc.Graph(
        id=title.lower().replace('.', '-').replace('_', '-').replace(' ', '-'),
        figure=figure
    )


def imputation_histogram(non_imputed, imputed, title: str = None) -> dcc.Graph:
    #x,y = sp.coo_matrix(non_imputed.isnull()).nonzero()
    non_imputed: pd.DataFrame = non_imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'})
    imputed: pd.DataFrame = imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'}).rename(columns={'value': 'log2 value'})
    imputed['Imputed'] = non_imputed['value'].isna()
    if title is None:
        title: str = 'Imputation'
    figure: go.Figure = px.histogram(
        imputed, x='log2 value', marginal='violin', color='Imputed', title=title
    )
    return dcc.Graph(id='missing-value-histogram', figure=figure)


def bar_plot(value_df: pd.DataFrame, title: str, x_name: str = None, y_name: str = None, y_idx: int = 0, barmode:str = 'relative', color: bool = True, color_col: str = None, hide_legend=False, color_discrete_map=False, color_discrete_map_dict:dict=None) -> px.bar:
    """Draws a bar plot from the given input.
    
    Parameters:
    value_df: dataframe containing the plot data
    title: title for the figure
    x_name: name of the column to use for x-axis values
    y_name: name of the column to use for y-axis values
    y_idx: index of the column to use for y-axis values
    barmode: see https://plotly.com/python-api-reference/generated/plotly.express.bar
    color: True(default) if a column called "Color" contains color values for the plot
    color_col: name of color information containing column, see px.bar reference
    hide_legend: True, if legend should be hidden
    color_discrete_map: if True, color_discrete_map='identity' will be used with the plotly function.
    """
    colorval: str
    if color_col is not None:
        colorval = color_col
    elif color:
        colorval = 'Color'
    else:
        colorval = None

    cdm_val: dict = None
    if color_discrete_map_dict is not None:
        cdm_val = color_discrete_map_dict
    else:
        if color_discrete_map:
            cdm_val = 'identity'
    if y_name is None:
        y_name: str = value_df.columns[y_idx]
    if x_name is None:
        x_name: str = value_df.index
    figure: px.bar = px.bar(
        value_df,
        x=x_name,  # 'Sample name',
        y=y_name,
        title=title,
        color=colorval,
        color_discrete_map=cdm_val
    )
    figure.update_xaxes(type='category')
    if hide_legend:
        figure.update_layout(showlegend=False)
    return figure


def contaminant_figure(data_table: pd.DataFrame, contaminant_list: list) -> dcc.Graph:

    contaminant_list: list = list(set(contaminant_list) & set(data_table.index.values))
    plot_data: list = []
    plot_index: list = []
    for column in data_table.columns:
        plot_index.append(column)
        plot_index.append(column)
    #plot_index:pd.Index = data_table.columns
    contaminants: pd.DataFrame = data_table.loc[contaminant_list]
    non_contaminants: pd.DataFrame = data_table.loc[~data_table.index.isin(contaminant_list)]

    for i in range(0, len(plot_index), 2):
        sample_name: str = plot_index[i]
        plot_data.append([contaminants[sample_name].sum(), 'Contaminants'])
        plot_data.append([non_contaminants[sample_name].sum(), 'Non-contaminants'])

    return dcc.Graph(
        id='contaminant-bar-plot',
        figure=bar_plot(
            pd.DataFrame(
                data=plot_data,
                index=plot_index,
                columns=['Sum value','Contaminant'],
            ),
            title='Contaminants',
            color_col='Contaminant',
            color_discrete_map_dict = {'Contaminants': 'Red','Non-contaminants': 'Blue'}
        )
    )

# TODO: Merge these six functions into one. Or rather, delete them and just call bar_plot from data analysis.py
def sum_value_figure(sum_data) -> dcc.Graph:
    sum_figure = bar_plot(
        sum_data, title='Value sum per sample', color_discrete_map=True)
    return dcc.Graph(id='value-sum-figure', figure=sum_figure)


def avg_value_figure(avg_data) -> dcc.Graph:
    avg_figure = bar_plot(
        avg_data, title='Value mean per sample', color_discrete_map=True)
    return dcc.Graph(id='value-sum-figure', figure=avg_figure)


def missing_figure(na_data) -> dcc.Graph:
    na_figure: px.bar = bar_plot(
        na_data, title='Missing values per sample', color_discrete_map=True)
    return dcc.Graph(id='protein-count-figure', figure=na_figure)


def protein_count_figure(count_data) -> dcc.Graph:
    """Generates a bar plot of given data"""
    count_figure: px.bar = bar_plot(
        count_data, title='Proteins per sample', color_discrete_map=True)
    return dcc.Graph(id='protein-count-figure', figure=count_figure)

def protein_coverage(data_table) -> dcc.Graph:
    return dcc.Graph(
        id='protein-coverage-{title}',
        figure=bar_plot(
            pd.DataFrame(data_table.notna()
                         .astype(int)
                         .sum(axis=1)
                         .value_counts(), columns=['Identified in # samples']), 'Protein coverage', color=False),

    )


def volcano_plots(data_table, sample_groups, control_group) -> list:
    """Generates volcano plots of all sample groups vs given control group in data_table.

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    sample_groups: dictionary of {sample_group_name: [sample_columns]}
    control_group: name of the control group
    
    Returns: 
    a list of [dcc.Graph] volcano plots.
    """
    control_cols: list = sample_groups[control_group]
    volcanoes: list = []
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


def volcano_plot(data_table, sample_name, control_name, sample_columns, control_columns, adj_p_threshold: float = 0.05, fc_threshold: float = 1, fc_axis_min_max: float = 2) -> dcc.Graph:
    """Draws a Volcano plot of the given data_table

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    sample_name: Name for the test sample
    control_name: Name for control
    sample_columns: columns corresponding to test sample data
    control_columns: columns corresponding to control data
    adj_p_threshold: threshold of significance for the calculated adjusted p value (Default 0.05)
    fc_threshold: threshold of significance for the log2 fold change. Proteins with fold change of <-fc_threshold or >fc_threshold are considered significant (Default 1)
    fc_axis_min_max: minimum for the maximum value of fold change axis. Default of 2 is used to keep the plot from becoming ridiculously narrow
    
    Returns: 
    dcc.Graph containing a go.Figure of the Volcano plot.
    """

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
    result['significant'] = ((result['p_value_adj'] < adj_p_threshold) & (
        abs(result['fold_change']) >= 1))
    result['p_value_adj_neg_log10'] = -np.log10(result['p_value_adj'])
    result['Highlight'] = [row['Name'] if row['significant']
                           else '' for _, row in result.iterrows()]
    # Draw the volcano plot using plotly express
    fig: go.Figure = px.scatter(result, x='fold_change', y='p_value_adj_neg_log10',
                     title=f'{sample_name} vs {control_name}', color='significant', text='Highlight')

    # Set yaxis properties
    p_thresh_val: float = -np.log10(adj_p_threshold)
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


def pca_plot(data_table: pd.DataFrame, rev_sample_groups: dict, n_components: int = 2, plot_name: str = None) -> dcc.Graph:
    """Draws a PCA plot of the given data_table

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    rev_sample_groups: dictionary of {sample_column_name: sample_group_name} containing all sample columns.
    n_components: how many components the PCA should have. Only first two will be used for the plot.
    plot_name: name for the plot. The id of the returned dcc.Graph will be f"pca-plot{plot_name}"
    
    Returns: 
    dcc.Graph containing a go.Figure of the PCA plot.
    """
    # Compute PCA of the data
    data_df: pd.DataFrame = data_table.T
    pca: decomposition.PCA = decomposition.PCA(n_components=n_components)
    pca_result: np.ndarray = pca.fit_transform(data_df)
    data_df['PCA one'] = pca_result[:, 0]
    data_df['PCA two'] = pca_result[:, 1]
    data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
    data_df['Sample name'] = data_df.index
    if plot_name is None:
        plot_name: str = ''
    else:
        plot_name = '-' + plot_name
    fig: go.Figure = px.scatter(
        data_df, x='PCA one', y='PCA two', title=f'PCA{plot_name}', color=data_df['Sample group'],
        text='Sample name')
    fig.update_traces(marker_size=15)
    return dcc.Graph(figure=fig, id=f'pca-plot{plot_name}')


def df_coefficient_of_variation(data_table: pd.DataFrame) -> pd.DataFrame:
    """Calculates coefficient of variation for a given dataframe with samples in columns and measurements in rows.

    Parameters:
    data_table: table of samples (columns) and measurements(rows)

    Returns:
    a pandas dataframe with three columns:
        "mean intensity": mean of each row in the data_table
        "CV": CV of each row in the data_table
        "%CV": CV*100
    """
    new_df: pd.DataFrame = pd.DataFrame(index=data_table.index)
    new_df['mean intensity'] = data_table.mean(axis=1)
    new_df['CV'] = data_table.std(axis=1)/new_df['mean intensity']
    new_df['%CV'] = new_df['CV']*100
    return new_df


def coefficient_of_variation_plot(data_table: pd.DataFrame, plot_name: str = None, title: str = None,draw_trendline:bool=True,trendline:str=None,trendline_options:dict=None) -> dcc.Graph:
    """Draws a CV plot of the given data_table with a trendline

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    plot_name: name for the plot. will be used for the id of the returned dcc.Graph object. default is "coefficient-of-variation"
    title: title for the figure
    draw_trendline: True(default) if trendline should be drawn
    trendline: Which type of trendline to draw (default lowess), see https://plotly.com/python-api-reference/generated/plotly.express.trendline_functions.html for options.
    trendline_options: options dict for the trendline

    Returns: 
    dcc.Graph containing a go.Figure of the CV plot.
    """

    if plot_name is None:
        plot_name:str = 'coefficient-of-variation'
    data_table: pd.DataFrame = df_coefficient_of_variation(data_table)
    int_col: str = 'mean intensity'
    cv_col: str = '%CV'
    data_table = data_table.sort_values(
        by=int_col, ascending=True, inplace=False)
    if draw_trendline:
        if trendline is None:
            trendline: str = 'lowess'
    else:
        trendline = None
    scatter_dots = px.scatter(
        data_table,
        x=cv_col,
        y=int_col,
        title=title,
        trendline=trendline,
        trendline_options = trendline_options,
        #trendline_options={'alpha': 0.1},
        trendline_color_override='black'
        #    color='hsl(210, 100%, 50%)'
    )
    if title is None:
        title: str = ''
    cv_str: str = f'Median CV: {data_table["CV"].median()}, '
    scatter_dots.update_layout(
        title=f'Coefficients of variation {cv_str}',
        xaxis_title='Log2 normalized imputed mean intensity',
        yaxis_title='% CV'
    )
    return dcc.Graph(figure=scatter_dots, id=f'scatter-{plot_name}')


def clustergram(plot_data: pd.DataFrame, color_map: list = None, **kwargs) -> dash_bio.Clustergram:
    """Draws a clustergram figure from the given plot_data data table.

    Parameters:
    plot_data: Clustergram data
    color-map: list of values and corresponding colors for the color map. default:  [[0.0, "#FFFFFF"], [1.0, "#EF553B"]]
    **kwargs: keyword arguments to pass on to dash_bio.Clustergram
    
    Returns: 
    dash_bio.Clustergram drawn with the input data.
    """
    if color_map is None:
        color_map: list = [
            [0.0, '#FFFFFF'],
            [1.0, '#EF553B']
        ]
    return dash_bio.Clustergram(
        data=plot_data,
        column_labels=list(plot_data.columns.values),
        row_labels=list(plot_data.index),
        # height=800,
        # width=750,
        color_map=color_map,
        link_method='average',
        **kwargs
    )


def correlation_clustermap(data_table: pd.DataFrame, plot_name: str = None) -> dcc.Graph:
    """Draws a correltion clustergram figure from the given data_table.

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    plot_name: name for the plot. will be used for the id of the returned dcc.Graph object.

    Returns: 
    dcc.Graph containing a dash_bio.Clustergram describing correlation between samples.
    """
    if plot_name is None:
        plot_name: str = 'correlation-clustermap'
    corr_df: pd.DataFrame = data_table.corr()
    fig: dash_bio.Clustergram = clustergram(
        corr_df,
    )
    # fig.update_zaxes(range=[0,1])
    # kokeile vielä layoutin kautta zmin ja zmax parametrejä
    # tai fig.update_layout(coloraxis=dict(cmax=6, cmin=3))
    return dcc.Graph(figure=fig, id=plot_name)


def full_clustermap(data_table: pd.DataFrame, plot_name: str = None) -> dcc.Graph:
    """Draws a clustermap figure from the given data_table.

    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    plot_name: name for the plot. will be used for the id of the returned dcc.Graph object.

    Returns: 
    dcc.Graph containing a dash_bio.Clustergram.
    """
    if plot_name is None:
        plot_name: str = 'full-clustermap'
    fig: dash_bio.Clustergram = clustergram(
        data_table,
        hidden_labels=['row'],
        # The cluster parameter should be "column", but that does not work. According to source code, "col" is the correct usage, but it might get changed to reflect documentation later.
        cluster='col',
        center_values=False
    )
    return dcc.Graph(figure=fig, id=plot_name)


def reproducibility_figure(data_table: pd.DataFrame, sample_groups: dict, title='Reproducibility plot') -> dcc.Graph:
    """Produces a graph describing reproducibility within sammple groups
    
    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    sample_groups: dictionary of {sample_group_name: [sample_columns]}
    title: title for the figure

    Returns: 
    dcc.Graph containing a go.Figure describing the reproducibility of the samples within sample groups.
    """
    results: list = []
    output_columns: list = None
    min_num_of_samples: int = None
    for _, sgcols in sample_groups.items():
        if len(sgcols) < 2:
            continue
        if min_num_of_samples is None:
            min_num_of_samples = len(sgcols)
        else:
            min_num_of_samples = min(min_num_of_samples, len(sgcols))
    ranked_proteins: list = list(data_table.sum(
        axis=1).sort_values(ascending=False).index)
    for groupname, groupcols in sample_groups.items():
        if len(groupcols) < 2:
            continue
        data_points: dict = {protein: {} for protein in data_table.index}
        for col in groupcols:
            index: int = 1
            for protein in ranked_proteins:
                intensity: float = data_table.loc[protein, col]
                data_points[protein][col] = [index, intensity]
                index += 1
        if min_num_of_samples == 2:
            for protein, pdic in data_points.items():
                results.append([
                    intensity for column, (index, intensity) in pdic.items()
                ][:2])
                results[-1].append(groupname)
        elif min_num_of_samples > 2:
            for protein, protein_dict in data_points.items():
                ranks: list = []
                intensities: list = []
                for _, values in protein_dict.items():
                    ranks.append(values[0])
                    intensities.append(values[1])
                int_series: pd.Series = pd.Series(intensities)
                if int_series.notna().sum() == 0:
                    continue
                avg_intensity: float = int_series.median()  # sum(intensities)/len(intensities)
               # avg_rank: float = sum(ranks)/len(ranks)

                summed_distance: float = 0.0
                for protein, (intensity, _) in protein_dict.items():
                    summed_distance += abs(intensity-avg_intensity)
                results.append(
                    [
                        avg_intensity, summed_distance, groupname
                    ]
                )
    if min_num_of_samples == 2:
        output_columns = ['Intensity1', 'Intensity2', 'Group name']
        plot_dataframe: pd.DataFrame = pd.DataFrame(
            data=results, columns=output_columns)
        groups: list = list(plot_dataframe['Group name'].unique())
        figure: go.Figure = go.Figure()
        for group in groups:
            df_group: pd.DataFrame = plot_dataframe[plot_dataframe['Group name'] == group]
            trace: go.Figure = go.Scatter(
                x=df_group[output_columns[0]],
                y=df_group[output_columns[1]],
                xaxis='x',
                yaxis='y',
                mode='markers',
                name=group,
                marker=dict(
                    size=3,
                    opacity=1
                )
            )
            figure.add_trace(trace)
        figure.update_layout(
            title=title,
            xaxis_title=output_columns[0],
            yaxis_title=output_columns[1],
            showlegend=True
        )
    else:
        output_columns = ['Average intensity',
                          'Summed distance to avg', 'Group name']
        plot_dataframe: pd.DataFrame = pd.DataFrame(
            data=results, columns=output_columns)
        groups: list = list(plot_dataframe['Group name'].unique())

        x: pd.Series = plot_dataframe[output_columns[0]]
        y: pd.Series = plot_dataframe[output_columns[1]]

        figure = go.Figure()
        figure.add_trace(go.Histogram2dContour(
            x=x,
            y=y,
            colorscale='Blues',
            xaxis='x',
            yaxis='y'
        ))
        for group in groups:
            df_group: pd.DataFrame = plot_dataframe[plot_dataframe['Group name'] == group]
            trace: go.Figure = go.Scatter(
                x=df_group[output_columns[0]],
                y=df_group[output_columns[1]],
                xaxis='x',
                yaxis='y',
                mode='markers',
                name=group,
                marker=dict(
                    size=3,
                    opacity=1
                )
            )
            figure.add_trace(trace)

        figure.add_trace(go.Histogram(
            y=y,
            xaxis='x2',
            marker=dict(
                color='rgba(0,0,0,1)'
            )
        ))
        figure.add_trace(go.Histogram(
            x=x,
            yaxis='y2',
            marker=dict(
                color='rgba(0,0,0,1)'
            )
        ))

        figure.update_layout(
            autosize=False,
            xaxis=dict(
                zeroline=False,
                domain=[0, 0.85],
                showgrid=False
            ),
            yaxis=dict(
                zeroline=False,
                domain=[0, 0.85],
                showgrid=False
            ),
            xaxis2=dict(
                zeroline=False,
                domain=[0.85, 1],
                showgrid=False
            ),
            yaxis2=dict(
                zeroline=False,
                domain=[0.85, 1],
                showgrid=False
            ),
            height=600,
            width=600,
            bargap=0,
            hovermode='closest',
            showlegend=False
        )

        figure.update_layout(
            title=title,
            xaxis_title=output_columns[0],
            yaxis_title=output_columns[1]
        )
    return dcc.Graph(figure=figure, id=f'{title.lower().replace(" ","-")}')


def supervenn(data_table: pd.DataFrame, rev_sample_groups: dict, save_figure:str=None, save_format:str='svg') -> html.Img:
    """Draws a super venn plot for the input data table.

    See https://github.com/gecko984/supervenn for details of the plot.
    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    rev_sample_groups: dictionary of {sample_column_name: sample_group_name} containing all sample columns.
    figure_name: name for the figure title, as well as saved file
    save_figure: directory to save the generated figure into. if None (default), figure will not be saved.
    save_format: format for the saved figure. default is svg.

    Returns:
    returns html.Img object containing the figure data in png form.
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
    plt.savefig(buffer, format="png")
    plt.close()
    data: str = base64.b64encode(buffer.getbuffer()).decode(
        "utf8")  # encode to html elements
    buffer.close()
    return html.Img(id='supervennfigure', src=f'data:image/png;base64,{data}')


def missing_clustermap(data_table: pd.DataFrame, plot_name: str = None, title: str = None) -> dcc.Graph:
    """Generates a clustergram of missing values in a data_table.
    
    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    plot_name: name for the output plot. will be used as the id value for the returned dcc.Graph object. If none, id will be "missing-value-clustergram"
    title: Plot title. If none, title will be "Missing values"

    Returns: 
    dcc.Graph containing a dash_bio.Clustergram of missing values.
    """
    if plot_name is None:
        plot_name: str = 'missing-value-clustermap'
    figure: dash_bio.Clustergram = clustergram(
        data_table.notna().astype(int),
        cluster='col',
        hidden_labels=['row'],
        color_map=[
            [0.0, '#000000'],
            [1.0, '#FFFFFF']
        ],
        center_values=False
    )
    if title is None:
        title: str = 'Missing values'
    figure.update_layout(title=title)
    return dcc.Graph(
        figure=figure,
        id=plot_name
    )


# TODO: NEeds testing
def get_volcano_df(data_table: pd.DataFrame, sample_columns, control_columns) -> pd.DataFrame:
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
                figure=dash_bio.VolcanoPlot(
                    dataframe=data_table
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


def t_sne_plot(data_table: pd.DataFrame, rev_sample_groups: dict, perplexity: int = 15, n_components: int = 2, iterations: int = 5000, figname: str = None) -> dcc.Graph:
    """Untested code"""
    data_df: pd.DataFrame = data_table.T
    perplexity: int = min(data_df.shape[0]-1, perplexity)
    tsne: manifold.TSNE = manifold.TSNE(n_components=n_components, verbose=0,
                                        perplexity=perplexity, n_iter=iterations)
    tsne_results: np.ndarray = tsne.fit_transform(data_df)
    data_df['t-SNE one'] = tsne_results[:, 0]
    data_df['t-SNE two'] = tsne_results[:, 1]
    data_df['Sample group'] = [rev_sample_groups[i] for i in data_df.index]
    data_df['Sample name'] = data_df.index
    if figname is None:
        figname: str = ''
    else:
        figname = '-' + figname
    fig: go.Figure = px.scatter(
        data_df, x='t-SNE one', y='t-SNE two', title=f't-SNE{figname}', color=data_df['Sample group'],
        text='Sample name')
    fig.update_traces(marker_size=15)
    return dcc.Graph(figure=fig, id=f'tsne-plot{figname}')