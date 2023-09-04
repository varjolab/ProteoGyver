from pandas import DataFrame
from dash_bio import Clustergram
from dash.dcc import Graph

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
