from plotly import express as px
import plotly.graph_objects as go
import numpy as np
from utilitykit import plotting
from dash import dcc
import data_functions
import pandas as pd
import plotly.express as px
from statsmodels.stats import multitest
from scipy.stats import ttest_ind

def add_replicate_colors(data_df, column_to_replicate):

    need_cols: int = list(
            {
                column_to_replicate[sname] for sname in \
                data_df.index.unique() \
                if sname in column_to_replicate
            }
        )
    colors: list = plotting.get_cut_colors(number_of_colors = len(need_cols))
    colors = plotting.cut_colors_to_hex(colors)
    colors = {sname: colors[i] for i, sname in enumerate(need_cols)}
    color_column:list = []    
    for sn in data_df.index.values:
        color_column.append(colors[column_to_replicate[sn]])
    data_df.loc[:,'Color'] = color_column


def bar_plot(value_df,title,y=0) -> px.bar:
    figure: px.bar = px.bar(
            value_df,
            x=value_df.index,#'Sample name',
            y=value_df.columns[y],
            height=500,
            width=750,
            title=title,
            color='Color',
            color_discrete_map='identity'
            )
    figure.update_xaxes(type='category')
    return figure

def sum_value_figure(sum_data) -> dcc.Graph:
    sum_figure = bar_plot(sum_data,title='Value sum per sample')
    return dcc.Graph(id='value-sum-figure', figure=sum_figure)

def avg_value_figure(avg_data) -> dcc.Graph:
    avg_figure = bar_plot(avg_data,title='Value mean per sample')
    return dcc.Graph(id='value-sum-figure', figure=avg_figure)

def missing_figure(na_data) -> dcc.Graph:
    na_figure: px.bar = bar_plot(na_data,title='Missing values per sample')
    return dcc.Graph(id='protein-count-figure',figure=na_figure)

def protein_count_figure(count_data) -> dcc.Graph:
    count_figure: px.bar = bar_plot(count_data,title='Proteins per sample')
    return dcc.Graph(id='protein-count-figure',figure=count_figure)


def distribution_figure(data_table, color_dict, sample_groups) -> dcc.Graph:
    data: list = []
    for col in data_table.columns:
        data.append(
            go.Violin(
                    x=np.log2(data_table[col]),
                    line_color = color_dict[col],
                    name=col,
                    legendgroup = sample_groups[col],
                    orientation = 'h',
                    side = 'positive',
                    width = 4,
                    points = False
                )
            )
    fheight = 40*len(data)
    layout: go.Layout = go.Layout(
        title = 'Value distribution',
        xaxis={
            'title': 'Log2 value',
            'showgrid': False,
            'zeroline': False
            },
        yaxis={
            'title': 'Sample',
            'showgrid': True,
            },
        legend_traceorder='grouped+reversed',
        height=fheight,
        width=750
    )

    fig: go.Figure = go.Figure(
            layout = layout,
            data=data,
            #height=500,
            #width=750,
        )

    return dcc.Graph(id='distribution-figure',figure=fig)

def volcano_plot(df, sample_columns, control_columns, p_threshold:float=0.05, fc_threshold:float=1) -> dcc.Graph:
    """ Untested code"""

    # Calculate the fold change for each protein between the two sample groups
    fold_change = np.log2(df[sample_columns].mean(axis=1) / df[control_columns].mean(axis=1))

    # Calculate the p-value for each protein using a two-sample t-test
    p_value: float = df.apply(lambda x: ttest_ind(x[sample_columns], x[control_columns])[1], axis=1)

    # Adjust the p-values for multiple testing using the Benjamini-Hochberg correction method
    _, p_value_adj, _, _ = multitest.multipletests(p_value, method='fdr_bh')

    # Create a new dataframe containing the fold change and adjusted p-value for each protein
    result: pd.DataFrame = pd.DataFrame({'fold_change': fold_change, 'p_value_adj': p_value_adj})

    # Filter the dataframe to only include proteins that have a p-value less than p_threshold and a significant fold change
    significant_proteins: pd.DataFrame = result[(result['p_value_adj'] < p_threshold) & ((result['fold_change'] > fc_threshold) | result['fold_change']<-fc_threshold)]

    # Draw the volcano plot using plotly express
    fig = px.scatter(result, x='fold_change', y=-1 * result.p_value_adj.apply(lambda x: -1 * np.log10(x)), title='Volcano plot')

    # Set the x-axis label
    fig.update_xaxes(title_text='Fold change')
    # Set the y-axis label
    fig.update_yaxes(title_text='-log10 (q-value)')


    # Add horizontal and vertical lines indicating the significance thresholds
    fig.add_shape(type='line', x0=-fc_threshold, y0=np.log10(p_threshold), x1=fc_threshold, y1=np.log10(p_threshold), line=dict(color='black', width=2, style='-'))
    fig.add_shape(type='line', x0=-fc_threshold, y0=0, x1=-fc_threshold, y1=np.log10(p_threshold), line=dict(color='black', width=2, style='-'))

    # Mark the significant proteins with a red color
    fig.add_trace(px.scatter(significant_proteins, x='fold_change', y=-1 * significant_proteins.p_value_adj.apply(lambda x: -1 * np.log10(x)), color='red'))

    # Show the plot
    fig.show()