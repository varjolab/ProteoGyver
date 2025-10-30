"""
Before/After grouped bar plot for paired counts.

Builds a simple grouped bar chart comparing two Series (before vs after)
per sample, returning a Dash ``Graph`` component.
"""
from pandas import DataFrame, Series
from plotly.express import bar
from dash.dcc import Graph
def make_graph(defaults:dict, before: Series, after: Series, graph_id:str, title: str = None) -> Graph:
    """Create a grouped bar chart comparing before vs after counts.

    :param defaults: Dictionary with ``config``, ``height``, ``width`` and related settings.
    :param before: Series of counts before, indexed by sample.
    :param after: Series of counts after, indexed by sample.
    :param graph_id: Component ID for the ``Graph``.
    :param title: Optional chart title.
    :returns: Dash ``Graph`` instance containing the grouped bar plot.
    """
    data: list = [['Before or after', 'Count', 'Sample']]
    for i in before.index:
        if i in after.index:
            data.extend([
                ['before', before[i], i],
                ['after', after[i], i]
            ])
    if title is None:
        title: str = ''
    dataframe: DataFrame = DataFrame(data=data[1:], columns=data[0])
    width: int = defaults['width']
    if 'min_width_per' in defaults and defaults['min_width_per'] > 0:
        target_width = defaults['side_width'] + defaults['min_width_per']*2*len(dataframe['Sample'].unique())
        if width < target_width:
            width = target_width
    return Graph(
            config=defaults['config'], 
            id=graph_id,
            figure=bar(
                dataframe,
                x='Sample',
                y='Count',
                color='Before or after',
                barmode='group',
                title=title,
                height=defaults['height'],
                width=width
            )
        )
