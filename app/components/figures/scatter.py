from plotly.graph_objects import Figure
from pandas import DataFrame
from plotly.express import scatter as px_scatter
from dash.dcc import Graph
from components import figure_functions

def make_figure(data_table: DataFrame, x: str, y: str, color_col: str, defaults: dict, title: str = None, msize: int = 15, improve_text_pos: bool = True) -> Figure:
    figure: Figure = px_scatter(
        data_table, 
        x=x, y=y, color=color_col,
        height = defaults['height'], width = defaults['width']
    )
    figure.update_traces(marker_size=msize)
    if improve_text_pos:
        figure.update_traces(textposition = figure_functions.improve_text_position(data_table))
    return figure

def make_graph(id_name, defaults, *args) -> Graph:
    return Graph(
        id=id_name, 
        config = defaults['config'],
        figure = make_figure(*args, defaults)
    )