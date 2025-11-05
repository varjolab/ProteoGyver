"""
Heatmap utilities, including clustered correlation heatmaps.

Provides a dendrogram-coupled clustergram builder and a simple imshow
heatmap factory for numeric matrices.
"""
from dash.dcc import Graph
from plotly import graph_objects as go
from plotly import express as px
from components import matrix_functions
from math import ceil
import numpy as np
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list



def draw_clustergram(plot_data, defaults, color_map:list|None = None, **kwargs) -> go.Figure:
    """Draw a clustered correlation heatmap with dendrograms.

    :param plot_data: Square DataFrame-like correlation matrix (symmetric).
    :param defaults: Dict with ``height`` and ``width``.
    :param color_map: Plotly colorscale list; defaults to white→red.
    :param kwargs: Optional ``zmin`` and ``zmax`` overrides.
    :returns: Plotly ``Figure`` containing the clustergram.
    :raises ValueError: If input is not square.
    """
    method: str = "average"
    if color_map is None:
        color_map = [
            [0.0, '#FFFFFF'],
            [1.0, '#EF553B']
        ]
    zmin: float = 0
    zmax: float = 1.0
    if 'zmin' in kwargs:
        zmin = kwargs['zmin']
    if 'zmax' in kwargs:
        zmax = kwargs['zmax']
    if plot_data.shape[0] != plot_data.shape[1]:
        raise ValueError("plot_data must be square (n x n) correlation matrix.")
    if not plot_data.index.equals(plot_data.columns):
        plot_data = plot_data.copy()
        plot_data.index = plot_data.columns

    labels = plot_data.columns.to_list()

    C = plot_data.copy().astype(float)
    np.fill_diagonal(C.values, 1.0)
    C = C.fillna(0.0)
    C = (C + C.T) / 2.0

    D = 1.0 - C
    np.fill_diagonal(D.values, 0.0)
    condensed = squareform(D.values, checks=False)

    col_link = linkage(condensed, method=method)
    row_link = linkage(condensed, method=method)
    col_order = leaves_list(col_link)
    row_order = leaves_list(row_link)

    corr_reordered = C.iloc[row_order, :].iloc[:, col_order]
    row_labels = [labels[i] for i in row_order]
    col_labels = [labels[i] for i in col_order]

    dendro_top = ff.create_dendrogram(
        corr_reordered.values, orientation="bottom", labels=col_labels,
        linkagefun=lambda _: col_link
    )
    for t in dendro_top['data']:
        t['yaxis'] = 'y2'

    dendro_left = ff.create_dendrogram(
        corr_reordered.values.T, orientation="right", labels=row_labels,
        linkagefun=lambda _: row_link
    )
    for t in dendro_left['data']:
        t['xaxis'] = 'x2'

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.18, 0.82],
        column_widths=[0.20, 0.80],   # left dendro column, heatmap column
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "xy"}, {"type": "heatmap"}]],
        horizontal_spacing=0.05,
        vertical_spacing=0.004,
        shared_xaxes=True
    )

    for trace in dendro_top['data']:
        fig.add_trace(trace, row=1, col=2)
    for trace in dendro_left['data']:
        fig.add_trace(trace, row=2, col=1)

    top_tickvals = dendro_top['layout']['xaxis']['tickvals']
    top_ticktext = dendro_top['layout']['xaxis']['ticktext']
    left_tickvals = dendro_left['layout']['yaxis']['tickvals']
    left_ticktext = dendro_left['layout']['yaxis']['ticktext']

    color_map: list = [
        [0.0, '#FFFFFF'],
        [1.0, '#EF553B']
    ]
    heatmap = go.Heatmap(
        z=corr_reordered.values,
        x=top_tickvals,
        y=left_tickvals,
        colorscale=color_map,
        zmin=zmin,
        zmax=zmax,
        xgap=0, ygap=0,                   # ← must be here
        colorbar=None,
        hovertemplate="row: %{customdata[0]}<br>col: %{customdata[1]}<br>r: %{z:.3f}<extra></extra>",
        customdata=np.dstack(np.meshgrid(left_ticktext, top_ticktext, indexing="ij"))
    )
    fig.add_trace(heatmap, row=2, col=2)

    # Axes for the heatmap
    fig.update_xaxes(
        row=2, col=2,
        tickmode="array",
        tickvals=top_tickvals,
        ticktext=top_ticktext,
        side="bottom",
        tickangle=90
    )
    fig.update_yaxes(
        row=2, col=2,
        tickmode="array",
        tickvals=left_tickvals,
        ticktext=left_ticktext,
        autorange="reversed",
        side="right",        # ← labels on the RIGHT
        automargin=True,     # ← allocate margin for long labels on the right
    )

    # Hide tick labels on dendrogram axes
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)
    fig.update_xaxes(visible=False, row=2, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)

    fig.update_layout(
        height=defaults['height'],
        width=defaults['width'],
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=70, t=50, b=100),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, ticks="")

    return fig

def make_heatmap_graph(matrix_df, plot_name:str, value_name:str, defaults: dict, cmap: str, dlname: str, autorange: bool = False, symmetrical: bool = True, cluster: str = None) -> Graph:
    """Create a simple heatmap as a Dash ``Graph``.

    :param matrix_df: DataFrame with numeric values to plot.
    :param plot_name: Name suffix for component ID.
    :param value_name: Colorbar label.
    :param defaults: Dict with ``height``, ``width``, ``config``.
    :param cmap: Plotly continuous color scale name.
    :param dlname: Name for the downloaded figure file.
    :param autorange: If ``True``, derive zmin from data with padding.
    :param symmetrical: If ``True``, use symmetric min/max around zero.
    :param cluster: If not ``None``, apply clustering via ``matrix_functions``.
    :returns: Dash ``Graph`` component with the heatmap figure.
    """
    zmi: int = 0
    if autorange:
        zmi = matrix_df.min().min()
        zmi = zmi - zmi*0.1
    #    zmi = -ceil(abs(zmi))
    zma: int = matrix_df.max().max()
    if cluster is not None:
        matrix_df = matrix_functions.hierarchical_clustering(matrix_df,cluster=cluster)
    zma = zma + zma*0.1
    zma = ceil(zma)
    if symmetrical:
        zma = max(zma, abs(zmi))
        zmi = -zma
    figure: go.Figure = px.imshow(
        matrix_df,
        aspect='auto',
        labels=dict(
            x=matrix_df.columns.name,
            y=matrix_df.index.name,
            color=value_name
        ),
        color_continuous_scale=cmap,
        height=defaults['height'],
        width=defaults['width'],
        zmin = zmi,
        zmax = zma,
    )
    config = defaults['config'].copy()
    config['toImageButtonOptions'] = config['toImageButtonOptions'].copy()
    config['toImageButtonOptions']['filename'] = dlname
    return Graph(config=config, figure=figure, id=f'heatmap-{plot_name}')