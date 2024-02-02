
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
        width: int = None,
        height: int = None) -> px.bar:
    """Draws a bar plot from the given input.

    Parameters:
    :param: defaults: dictionary of default values for the figure.
    :param: value_df: dataframe containing the plot data
    :param: title: title for the figure
    :param: x_name: name of the column to use for x-axis values
    :param: x_label: label to use for X axis irregardless of what x_name is
    :param: sort_x: Sort values by x-axis name. If True, sort ascending, if False, sort descending. If None, default sorting will be used.
    :param: y_name: name of the column to use for y-axis values
    :param: y_idx: index of the column to use for y-axis values
    :param: barmode: see https://plotly.com/python-api-reference/generated/plotly.express.bar
    :param: color: True(default) if a column called "Color" contains color values for the plot
    :param: color_col: name of color information containing column, see px.bar reference
    :param: hide_legend: True, if legend should be hidden
    :param: color_discrete_map: if True, color_discrete_map='identity' will be used with the plotly function.
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
    return Graph(
        id=graph_id,
        config=defaults['config'],
        figure=bar_plot(
            defaults,
            *args,
            **kwargs
        )
    )
