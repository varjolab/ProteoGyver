import numpy as np
from pandas import DataFrame
from dash.dcc import Graph
from plotly.graph_objects import Figure, Violin

def make_graph(id_name: str, sets: list, defaults: dict, names: list = None, replicate_colors: dict = None, title: str = None, showbox:bool = False) -> Graph:
    if id_name is None:
        id_name: str = 'comparative-violin-plot'
    if isinstance(names, list):
        assert ((len(sets) == len(names))
                ), 'Length of "sets" should be the same as length of "names"'
    else:
        names: list = []
        for i in range(0, len(sets)):
            names.append(f'Set {i+1}')
    plot_data: list = np.array([])
    plot_legend: list = [[], []]
    for i, data_frame in enumerate(sets):
        for col in data_frame.columns:
            plot_data = np.append(plot_data, data_frame[col].values)
            plot_legend[0].extend([names[i]]*data_frame.shape[0])
            plot_legend[1].extend([f'{col} {names[i]}']*data_frame.shape[0])
    plot_df: DataFrame = DataFrame(
        {
            'Values': plot_data,
            'Column': plot_legend[1],
            'Name': plot_legend[0]
        }
    )


    figure: Figure = Figure()
    for sample_group in plot_df['Name'].unique():
        trace_df: DataFrame = plot_df[plot_df['Name']==sample_group]
        figure.add_trace(
            Violin(
                x=trace_df['Column'],
                y=trace_df['Values'],
                name = sample_group, 
                line_color=replicate_colors['sample groups'][sample_group]
             )
        )
    figure.update_traces(box_visible = showbox, meanline_visible=True)
    figure.update_layout(violinmode='group',
        height=defaults['height'],
        width=defaults['width']
    )
    if title is not None:
        figure.update_layout(title=title)
    return Graph(
        id=id_name,
        config=defaults['config'],
        figure=figure
    )