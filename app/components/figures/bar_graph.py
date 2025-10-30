"""
Bar graph utilities using Plotly Express and Dash.

Provides a configurable bar plot builder and a Dash ``Graph`` factory.
"""
from plotly import express as px
import pandas as pd
from dash.dcc import Graph


def bar_plot(
        defaults: dict,
        value_df: pd.DataFrame,
        title: str,
        x_name: str = None,
        x_label: str = None,
        sort_x: bool = None,
        y_name: str = None,
        y_label: str = None,
        y_idx: int = 0,
        barmode: str = 'relative',
        color: bool = True,
        color_col: str = None,
        hide_legend=False,
        color_discrete_map=False,
        color_discrete_map_dict: dict = None,
        width: int|None = None,
        height: int|None = None) -> px.bar:
    """Draw a bar plot from the given input.

    :param defaults: Dictionary of default values for the figure.
    :param value_df: DataFrame containing the plot data.
    :param title: Title for the figure.
    :param x_name: Column to use for x-axis; if ``None``, index is reset and the new column is used.
    :param x_label: Axis label to use for X regardless of ``x_name``.
    :param sort_x: If ``True`` ascending, ``False`` descending, ``None`` leaves default order.
    :param y_name: Column to use for y-axis; if ``None``, use ``y_idx``.
    :param y_label: Axis label to use for Y regardless of ``y_name``/``y_idx``.
    :param y_idx: Index of the column to use for y-axis when ``y_name`` is ``None``.
    :param barmode: Plotly Express bar ``barmode``.
    :param color: If ``True``, use column ``Color`` unless ``color_col`` is provided.
    :param color_col: Explicit name of color column.
    :param hide_legend: If ``True``, hides the legend.
    :param color_discrete_map: If ``True``, use ``'identity'`` mapping or a provided dict.
    :param color_discrete_map_dict: Explicit map for colors.
    :param width: Figure width; if ``None``, derived from defaults and min width policy.
    :param height: Figure height; if ``None``, derived from defaults.
    :returns: A Plotly Express bar figure.
    """
    colorval: str
    if color_col is not None:
        colorval = color_col
    elif color:
        colorval = 'Color'
    else:
        colorval = None

    cdm_val: dict = None
    if color_discrete_map_dict is not None:
        cdm_val = color_discrete_map_dict
    else:
        if color_discrete_map:
            cdm_val = 'identity'
    if y_name is None:
        y_name: str = value_df.columns[y_idx]
    if x_name is None:
        before: set = set(value_df.columns)
        value_df = value_df.reset_index()
        # Pick out the name of the new column
        x_name = [c for c in value_df.columns if c not in before][0]
    if height is None:
        height: int = defaults['height']
    if width is None:
        width: int = defaults['width']
        if 'min_width_per' in defaults and defaults['min_width_per'] > 0:
            target_width = defaults['side_width'] + (defaults['min_width_per']*len(value_df[x_name].unique()))
            if width < target_width:
                width = target_width
    cat_ord: dict = {}
    if sort_x is not None:
        cat_ord[x_name] = sorted(
            list(value_df[x_name].values), reverse=(not sort_x))
    figure: px.bar = px.bar(
        value_df,
        x=x_name,  # 'Sample name',
        y=y_name,
        category_orders=cat_ord,
        title=title,
        color=colorval,
        barmode=barmode,
        color_discrete_map=cdm_val,
        height=height,
        width=width
    )
    if x_label is not None:
        figure.update_layout(
            xaxis_title=x_label
        )
    if y_label is not None:
        figure.update_layout(
            yaxis_title=y_label
        )
    figure.update_xaxes(type='category')
    if hide_legend:
        figure.update_layout(showlegend=False)
    return figure


def make_graph(graph_id: str, defaults: dict, *args, **kwargs) -> None:
    """Create a Dash ``Graph`` configured with a bar plot.

    :param graph_id: Component ID for the ``Graph``.
    :param defaults: Dictionary with ``config``, ``height``, ``width`` and related settings.
    :param args: Positional arguments forwarded to ``bar_plot``.
    :param kwargs: Keyword arguments forwarded to ``bar_plot``.
    :returns: Dash ``Graph`` instance.
    """
    config=defaults['config']
    figure = bar_plot(
            defaults,
            *args,
            **kwargs
        )
    figure.update_layout(hovermode='closest', dragmode=False)
    return Graph(
        id=graph_id,
        config=config,
        figure=figure
    )
