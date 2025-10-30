"""
Histogram comparing imputed vs non-imputed distributions.

Builds a violin-marginal histogram and highlights imputed values for
visual comparison of value distributions.
"""
import numpy as np
from pandas import DataFrame
from dash.dcc import Graph
from plotly.graph_objects import Figure
from plotly.express import histogram

def make_graph(non_imputed, imputed, defaults, id_name: str = None, title:str = None, **kwargs) -> Graph:
    """Create a histogram comparing imputed vs non-imputed values.

    :param non_imputed: DataFrame before imputation.
    :param imputed: DataFrame after imputation.
    :param defaults: Dict with ``height``, ``width``, ``config``.
    :param id_name: Component ID for the ``Graph``.
    :param title: Figure title.
    :param kwargs: Additional keyword args forwarded to Plotly Express ``histogram``.
    :returns: Dash ``Graph`` containing the histogram figure.
    """
    #x,y = sp.coo_matrix(non_imputed.isnull()).nonzero()
    non_imputed: DataFrame = non_imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'})
    imputed: DataFrame = imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'}).rename(columns={'value': 'log2 value'})
    if id_name is None:
        id_name: str = 'imputation-histogram'
    imputed['Imputed'] = non_imputed['value'].isna()
    imputed.sort_values(by='Imputed', ascending=True, inplace=True)
    if 'height' not in kwargs:
        kwargs: dict = dict(kwargs,height=defaults['height'])
    if 'width' not in kwargs:
        kwargs = dict(kwargs,width=defaults['width'])
        
    figure: Figure = histogram(
        imputed,
        x='log2 value',
        marginal='violin',
        color='Imputed',
        title=title,
        **kwargs
    )
    figure.update_layout(
        barmode='overlay'
    )
    figure.update_traces(opacity=0.75) 
    config=dict(defaults['config'],displayModeBar = False)
    figure.update_layout(hovermode=False)
    figure.update_traces(hoverinfo='skip', hovertemplate=None)
    return Graph(config=config, id=id_name, figure=figure)
    