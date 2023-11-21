import numpy as np
from pandas import DataFrame
from dash.dcc import Graph
from plotly.graph_objects import Figure, Violin
from datetime import datetime
import logging
logger = logging.getLogger(__name__)


def make_graph(id_name: str, sets: list, defaults: dict, names: list = None, replicate_colors: dict = None, points_visible: str = False, title: str = None, showbox: bool = False) -> Graph:
    start_time = datetime.now()
    logger.debug(
        f'started: {start_time}')
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
    logger.debug(
        f'things initialized: {datetime.now() - start_time}')
    previous_time = datetime.now()
    for i, data_frame in enumerate(sets):
        for col in data_frame.columns:
            plot_data = np.append(plot_data, data_frame[col].values)
            plot_legend[0].extend([names[i]]*data_frame.shape[0])
            plot_legend[1].extend([f'{col} {names[i]}']*data_frame.shape[0])
    logger.debug(
        f'plot data done: {datetime.now() - previous_time}')
    logger.debug(
        f'plot data {plot_data.shape}')
    logger.debug(
        f'plot legend {len(plot_legend[0])} {len(plot_legend[1])}')
    previous_time = datetime.now()
    plot_df: DataFrame = DataFrame(
        {
            'Values': plot_data,
            'Column': plot_legend[1],
            'Name': plot_legend[0]
        }
    )
    logger.debug(
        f'only plot left: {datetime.now()-previous_time}')
    logger.debug(
        f'plot df {plot_df.shape}')
    previous_time = datetime.now()

    figure: Figure = Figure()
    for sample_group in plot_df['Name'].unique():
        trace_df: DataFrame = plot_df[plot_df['Name'] == sample_group]
        figure.add_trace(
            Violin(
                x=trace_df['Column'],
                y=trace_df['Values'],
                name=sample_group,
                line_color=replicate_colors['sample groups'][sample_group]
            )
        )
    logger.debug(
        f'updating traces: {datetime.now()-previous_time}')
    previous_time = datetime.now()
    figure.update_traces(
        box_visible=showbox,
        points=points_visible,
        meanline_visible=True)
    logger.debug(
        f'updating layout: {datetime.now()-previous_time}')
    previous_time = datetime.now()
    figure.update_layout(
        violinmode='group',
        height=defaults['height'],
        width=defaults['width']
    )
    if title is not None:
        figure.update_layout(title=title)
    logger.debug(
        f'returning graph: {datetime.now()-previous_time}')
    previous_time = datetime.now()
    return Graph(
        id=id_name,
        config=defaults['config'],
        figure=figure
    )
