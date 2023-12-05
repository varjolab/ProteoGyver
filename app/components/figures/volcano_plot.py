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
) -> tuple:
    """Draws a Volcano plot of the given data_table

    :param data_table: data table from stats.differential. Should only contain one comparison.
    :param defaults: dictionary with height and width for the figure.
    :param title: Figure title
    :param fc_axis_min_max: minimum for the maximum value of fold change axis. Default of 2 is used to keep the plot from becoming ridiculously narrow
    :param adj_p_threshold: threshold of significance for the calculated adjusted p value (Default 0.01)
    :param fc_threshold: threshold of significance for the log2 fold change. Proteins with fold change of <-fc_threshold or >fc_threshold are considered significant (Default 1)

    :param highlight_only: only highlight significant ones that are also in this list

    :returns: volcano_plot: go.Figure
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
        render_mode='svg'
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
    fig.update_xaxes(title_text='Fold change', range=[-fcrange, fcrange])

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
    return_div_contents: list = []
    significant_data = significant_data.sort_values(by=['Sample','Control'])
    for control in significant_data['Control'].unique():
        sigs = significant_data[significant_data['Control']==control]
        sigs = sigs[sigs['p_value_adj']<p_thr]
        sigs = sigs[abs(sigs['fold_change'])>fc_thr]
        sigs = sigs.pivot_table(columns='Name',index='Sample',values='fold_change')
        print(sigs.shape)
        if sigs.shape[0] > 1:
            return_div_contents.extend([
                html.H4(id=f'{id_prefix}-volcano-header-heatmap-{control}',
                        children=f'All significant differences vs {control}'),
                heatmaps.make_heatmap_graph(
                    sigs,
                    plot_name=f'volcano-significant-vs-{control.lower().strip()}',
                    value_name='log2 fold change',
                    defaults=defaults,
                    cmap='balance',
                    autorange=True,
                    symmetrical=True
                ),
                volcano_heatmap_legend(control, id_prefix)
            ])
    significant_data.to_csv('SIGDATA.tsv',sep='\t')
    for _, row in significant_data[['Sample', 'Control']].drop_duplicates().iterrows():
        sample: str = row['Sample']
        control: str = row['Control']
        return_div_contents.append(
            html.H4(id=f'{id_prefix}-volcano-header-{sample}-vs-{control}',
                    children=f'Volcano {sample} vs {control}'),
        )
        return_div_contents.append(
            dcc.Graph(
                id=f'{id_prefix}-{sample}-vs-{control}-volcano',
                figure=volcano_plot(
                    significant_data[(significant_data['Sample'] == sample) & (
                        significant_data['Control'] == control)],
                    defaults, adj_p_threshold=p_thr, fc_threshold=fc_thr
                ),
                config=defaults['config']
            )
        )
        return_div_contents.append(
            volcano_plot_legend(sample, control, id_prefix))
    return html.Div(
        children=return_div_contents
    )
