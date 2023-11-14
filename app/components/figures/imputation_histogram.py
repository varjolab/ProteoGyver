import numpy as np
from pandas import DataFrame
from dash.dcc import Graph
from plotly.graph_objects import Figure
from plotly.express import histogram

def make_graph(non_imputed, imputed, defaults, id_name: str = None, title:str = None, **kwargs) -> Graph:
    #x,y = sp.coo_matrix(non_imputed.isnull()).nonzero()
    non_imputed: DataFrame = non_imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'})
    imputed: DataFrame = imputed.melt(ignore_index=False).rename(
        columns={'variable': 'Sample'}).rename(columns={'value': 'log2 value'})
    if id_name is None:
        id_name: str = 'imputation-histogram'
    imputed['Imputed'] = non_imputed['value'].isna()
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
    return Graph(config=defaults['config'], id=id_name, figure=figure)