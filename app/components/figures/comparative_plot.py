"""
Comparative violin/box plot for grouped data frames.

Takes a list of DataFrames (columns = replicates), stacks values into a
long format and renders either grouped violin or box plots.
"""
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
        points_visible: str = 'outliers', 
        title: str = None, 
        showbox: bool = False, 
        plot_type: str = 'violin') -> Graph:
    """Create a comparative violin/box plot from multiple DataFrames.

    :param id_name: Component ID for the ``Graph``; default name used if ``None``.
    :param sets: List of DataFrames; each column is treated as a replicate.
    :param defaults: Dictionary with ``config``, ``height``, ``width`` and related settings.
    :param names: Optional list of labels for each DataFrame in ``sets``.
    :param replicate_colors: Mapping with ``'sample groups'`` color strings per name.
    :param points_visible: Violin/box points visibility (e.g., ``'outliers'`` or ``False``).
    :param title: Optional title.
    :param showbox: For violin, whether to show the internal box.
    :param plot_type: ``'violin'`` or ``'box'``.
    :returns: Dash ``Graph`` instance with the figure.
    """
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
    width: int = defaults['width']
    if 'min_width_per' in defaults and defaults['min_width_per'] > 0:
        target_width = defaults['side_width'] + defaults['min_width_per']*len(plot_df['Column'].unique())
        if width < target_width:
            width = target_width
    trace_args: dict = dict()
    layout_args: dict = {
        'height': defaults['height'],
        'width': width
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
    
    logger.info(
        f'returning graph: {datetime.now()}')
    return Graph(
        id=id_name,
        config=defaults['config'],
        figure=figure
    )
