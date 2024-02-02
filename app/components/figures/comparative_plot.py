import numpy as np
from pandas import DataFrame
from dash.dcc import Graph
from plotly.graph_objects import Figure, Violin, Box
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def make_graph(
        id_name: str, 
        sets: list, 
        defaults: dict, 
        names: list = None, 
        replicate_colors: dict = None, 
        points_visible: str = False, 
        title: str = None, 
        showbox: bool = False, 
        plot_type: str = 'violin') -> Graph:
    if id_name is None:
        id_name: str = 'comparative-violin-plot'
    if isinstance(names, list):
        assert ((len(sets) == len(names))
                ), 'Length of "sets" should be the same as length of "names"'
    else:
        names: list = []
        for i in range(0, len(sets)):
            names.append(f'Set {i+1}')
    plot_data: np.array = np.array([])
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
    trace_args: dict = dict()
    layout_args: dict = {
        'height': defaults['height'],
        'width': defaults['width']
    }
    if title is not None:
        layout_args['title'] = title
    if plot_type == 'violin': 
        plot_func = Violin
        trace_args['box_visible'] = showbox
        trace_args['meanline_visible'] = True
        trace_args['points'] = points_visible
        layout_args['violinmode'] = 'group'
    elif plot_type == 'box':
        plot_func = Box
        trace_args['boxpoints'] = points_visible
        trace_args['boxmean'] = True
        trace_args['boxpoints'] = points_visible
        layout_args['boxmode'] = 'group'
    figure: Figure = Figure()
    for sample_group in plot_df['Name'].unique():
        trace_df: DataFrame = plot_df[plot_df['Name'] == sample_group]
        figure.add_trace(
            plot_func(
                x=trace_df['Column'],
                y=trace_df['Values'],
                name=sample_group,
                line_color=replicate_colors['sample groups'][sample_group]
            )
        )
    figure.update_traces(**trace_args)
    figure.update_layout(**layout_args)
    
    logger.warning(
        f'returning graph: {datetime.now()}')
    previous_time = datetime.now()
    return Graph(
        id=id_name,
        config=defaults['config'],
        figure=figure
    )
