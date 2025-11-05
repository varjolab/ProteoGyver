"""
Volcano plots and significant-difference heatmaps.

Provides a Plotly volcano plot builder and utilities to generate a set
of volcano graphs and an aggregate heatmap of significant changes.
"""
from pandas import DataFrame
from dash import html, dcc
import numpy as np
from plotly import graph_objects as go
from plotly import express as px
from components.figures.figure_legends import volcano_plot_legend, volcano_heatmap_legend
from components.figures import heatmaps


def volcano_plot(
    data_table, defaults, title: str = None, fc_axis_min_max: float = 2, highlight_only: list = None,
    adj_p_threshold: float = 0.01, fc_threshold: float = 1.0
) -> go.Figure:
    """Draw a volcano plot from a differential results table.

    :param data_table: Data from ``stats.differential``; only one comparison expected.
    :param defaults: Dict with ``height`` and ``width`` for the figure.
    :param title: Figure title.
    :param fc_axis_min_max: Minimum absolute FC range to avoid narrow plots.
    :param adj_p_threshold: Adjusted p-value threshold (q-value).
    :param fc_threshold: Absolute log2 fold change threshold for significance.
    :param highlight_only: Optional set/list; only significant points with names in this set are labeled.
    :returns: Plotly ``Figure``.
    """
    if highlight_only is None:
        highlight_only = set(data_table['Name'].values)
    data_table['Highlight'] = [row['Name'] if ((row['Significant']) & (row['Name'] in highlight_only))
                               else '' for _, row in data_table.iterrows()]
    # Draw the volcano plot using plotly express
    fig: go.Figure = px.scatter(
        data_table,
        x='fold_change',
        y='p_value_adj_neg_log10',
        title=title,
        color='Significant',
        text='Highlight',
        height=defaults['height'],
        width=defaults['width'],
        render_mode='svg',
        hover_data=['Name','Gene','Significant','p_value_adj_neg_log10','fold_change']
    )

    # Set yaxis properties
    p_thresh_val: float = -np.log10(adj_p_threshold)
    pmax: float = max(
        data_table['p_value_adj_neg_log10'].max(), p_thresh_val)+0.5
    fig.update_yaxes(title_text='-log10 (q-value)', range=[0, pmax])
    # Set the x-axis properties
    fcrange: float = max(abs(data_table['fold_change']).max(), fc_threshold)
    if fcrange < fc_axis_min_max:
        fcrange = fc_axis_min_max
    fcrange += 0.25
    fig.update_xaxes(title_text='Log2 fold change', range=[-fcrange, fcrange])

    # Add vertical lines indicating the significance thresholds
    fig.add_shape(type='line', x0=-fc_threshold, y0=0, x1=-
                  fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
    fig.add_shape(type='line', x0=fc_threshold, y0=0,
                  x1=fc_threshold, y1=pmax, line=dict(width=2, dash='dot'))
    # And horizontal line:
    fig.add_shape(type='line', x0=-fcrange, y0=p_thresh_val,
                  x1=fcrange, y1=p_thresh_val, line=dict(width=2, dash='dot'))

    # Return the plot
    return fig


def generate_graphs(significant_data: DataFrame, defaults: dict, fc_thr: float, p_thr: float, id_prefix: str) -> html.Div:
    """Generate volcano plots and a heatmap for significant changes.

    :param significant_data: Long-form DataFrame with at least ``Sample``, ``Control``, ``fold_change``, ``p_value_adj`` columns.
    :param defaults: Dict with figure defaults and component config.
    :param fc_thr: Absolute log2 fold change threshold for significance.
    :param p_thr: Adjusted p-value threshold.
    :param id_prefix: Prefix used for generated component IDs.
    :returns: ``dash.html.Div`` containing headers, plots, and legends.
    """
    return_div_contents: list = []
    significant_data = significant_data.sort_values(by=['Sample','Control'])
    for control in significant_data['Control'].unique():
        sigs = significant_data[significant_data['Control']==control]
        sigs = sigs[sigs['p_value_adj']<p_thr]
        sigs = sigs[abs(sigs['fold_change'])>fc_thr]
        sigs = sigs.pivot_table(columns='Name',index='Sample',values='fold_change')
        if sigs.shape[0] > 1:
            dlname = f'All significant differences vs {control}'
            return_div_contents.extend([
                html.H4(id=f'{id_prefix}-volcano-header-heatmap-{control}',
                        children=dlname),
                heatmaps.make_heatmap_graph(
                    sigs,
                    plot_name=f'volcano-significant-vs-{control.lower().strip()}',
                    value_name='log2 fold change',
                    defaults=defaults,
                    cmap='balance',
                    dlname=dlname,
                    autorange=True,
                    symmetrical=True,
                    cluster='columns'
                ),
                volcano_heatmap_legend(control, id_prefix)
            ])
    for _, row in significant_data[['Sample', 'Control']].drop_duplicates().iterrows():
        sample: str = row['Sample']
        control: str = row['Control']
        dlname = f'Volcano {sample} vs {control}'
        return_div_contents.append(
            html.H4(id=f'{id_prefix}-volcano-header-{sample}-vs-{control}',
                    children=dlname),
        )
        
        config = defaults['config'].copy()
        config['toImageButtonOptions'] = config['toImageButtonOptions'].copy()
        config['toImageButtonOptions']['filename'] = dlname
        return_div_contents.append(
            dcc.Graph(
                id=f'{id_prefix}-{sample}-vs-{control}-volcano',
                figure=volcano_plot(
                    significant_data[(significant_data['Sample'] == sample) & (
                        significant_data['Control'] == control)].copy(),
                    defaults, adj_p_threshold=p_thr, fc_threshold=fc_thr
                ),
                config=config
            )
        )
        return_div_contents.append(
            volcano_plot_legend(sample, control, id_prefix))
    return html.Div(
        children=return_div_contents
    )
