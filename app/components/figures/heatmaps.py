from dash_bio import Clustergram
from dash.dcc import Graph
from plotly import graph_objects as go
from plotly import express as px
from math import ceil

def draw_clustergram(plot_data, defaults, color_map:list = None, **kwargs) -> Clustergram:
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
    return Clustergram(
        data=plot_data,
        column_labels=list(plot_data.columns.values),
        row_labels=list(plot_data.index),
        color_map=color_map,
        link_method='average',
        height=defaults['height'],
        width=defaults['width'],
        **kwargs
    )

def make_heatmap_graph(matrix_df, plot_name:str, value_name:str, defaults: dict, cmap: str, autorange: bool = False, symmetrical: bool = True) -> Graph:
    zmi: int = 0
    if autorange:
        zmi = matrix_df.min().min()
        zmi = zmi = zmi*0.1
        zmi = -ceil(abs(zmi))
    zma: int = matrix_df.max().max()
    zma = zma + zma*0.1
    zma = ceil(zma)
    if symmetrical:
        zma = max(zma, abs(zmi))
        zmi = -zma
    figure: go.Figure = px.imshow(matrix_df,
                                    aspect='auto',
                                    labels=dict(
                                    x=matrix_df.columns.name,
                                    y=matrix_df.index.name,
                                    color=value_name),
                                    color_continuous_scale=cmap,
                                    height=defaults['height'],
                                    width=defaults['width'],
                                    zmin = zmi,
                                    zmax = zma)
    return Graph(config=defaults['config'], figure=figure, id=f'heatmap-{plot_name}')