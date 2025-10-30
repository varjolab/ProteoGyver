"""
Histogram figure utilities.

Provides a simple wrapper around Plotly Express histogram construction
with sensible defaults drawn from a ``defaults`` dictionary.
"""
import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame
def make_figure(data_table: DataFrame, x_column: str, title: str, defaults: dict,**kwargs) -> go.Figure:
    """Create a histogram figure using Plotly Express.

    :param data_table: Input DataFrame.
    :param x_column: Column name to plot on the x-axis.
    :param title: Figure title.
    :param defaults: Dict with ``height`` and ``width``.
    :param kwargs: Additional Plotly Express ``histogram`` kwargs; ``height``, ``width``, and ``nbins`` defaulted if missing.
    :returns: Plotly ``Figure``.
    """
    if 'height' not in kwargs:
        kwargs: dict = dict(kwargs,height=defaults['height'])
    if 'width' not in kwargs:
        kwargs = dict(kwargs,width=defaults['width'])
    if 'nbins' not in kwargs:
        kwargs = dict(kwargs,nbins=50)
    figure: go.Figure = px.histogram(
        data_table,
        x=x_column,
        title=title,
        **kwargs
    )
    return figure