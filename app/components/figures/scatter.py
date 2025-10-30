"""
Scatter plot utilities with categorical coloring.

Offers a figure factory that preserves input categorical color order and
a Dash ``Graph`` wrapper.
"""
from plotly.graph_objects import Figure
from pandas import DataFrame
from plotly.express import scatter as px_scatter
from dash.dcc import Graph
from components import figure_functions

def make_figure(defaults: dict, data_table: DataFrame, x: str, y: str, color_col: str, name_col: str, msize: int = 15, improve_text_pos: bool = True, **kwargs) -> Figure:
    """Create a scatter plot with per-category colors and labels.

    :param defaults: Dict with ``height``, ``width``.
    :param data_table: Input DataFrame.
    :param x: Column name for x-axis.
    :param y: Column name for y-axis.
    :param color_col: Column containing color strings.
    :param name_col: Column used for categories and hover names.
    :param msize: Marker size.
    :param improve_text_pos: If ``True``, adjust text positions heuristically.
    :param kwargs: Additional keyword args forwarded to Plotly Express ``scatter``.
    :returns: Plotly ``Figure``.
    """
    color_seq = []
    cat_ord = {name_col: []}
    for _,row in data_table[[color_col, name_col]].drop_duplicates().iterrows():
        cat_ord[name_col].append(row[name_col])
        color_seq.append(row[color_col])
    figure: Figure = px_scatter(
        data_table, 
        x=x, y=y, 
        color=name_col,
        color_discrete_sequence = color_seq,
        category_orders = cat_ord,
        hover_name=name_col,
        height = defaults['height'], width = defaults['width'], 
        **kwargs
    )
    figure.update_traces(marker_size=msize)
    if improve_text_pos:
        figure.update_traces(textposition = figure_functions.improve_text_position(data_table))
    return figure

def make_graph(id_name, defaults, *args, **kwargs) -> Graph:
    """Wrap the scatter figure in a Dash ``Graph`` component.

    :param id_name: Component ID for the ``Graph``.
    :param defaults: Dict including a ``config`` key passed to the component.
    :param args: Positional args passed to ``make_figure``.
    :param kwargs: Keyword args passed to ``make_figure``.
    :returns: Dash ``Graph`` component.
    """
    return Graph(
        id=id_name, 
        config = defaults['config'],
        figure = make_figure(defaults, *args, **kwargs)
    )