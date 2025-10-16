from pandas import DataFrame, Series
from plotly.express import bar
from dash.dcc import Graph
def make_graph(defaults:dict, before: Series, after: Series, graph_id:str, title: str = None) -> Graph:
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
