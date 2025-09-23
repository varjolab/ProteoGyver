
import plotly.graph_objects as go
import plotly.express as px
from pandas import DataFrame
def make_figure(data_table: DataFrame, x_column: str, title: str, defaults: dict,**kwargs) -> go.Figure:
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