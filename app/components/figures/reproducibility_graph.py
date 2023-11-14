import pandas as pd
from dash.dcc import Graph
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def get_reproducibility_dataframe(data_table: pd.DataFrame, sample_groups: dict) -> pd.DataFrame:
    repro_data: dict = {}
    for sample_group, sample_columns in sample_groups.items():
        sgroup_data_table: pd.DataFrame = data_table[sample_columns]
        mean: pd.Series = sgroup_data_table.mean(axis=1)
        for col in sgroup_data_table.columns:
            if sample_group not in repro_data:
                repro_data[sample_group] = {}
            coldata: pd.Series = sgroup_data_table[col].dropna()
            coldata = coldata-mean.loc[coldata.index]
            repro_data[sample_group][col] = list(coldata.values)
    return repro_data


def get_max(plot_data: dict, minval=2) -> int:
    maxval: int = minval
    for _, sgroup_data in plot_data.items():
        for __, replicate_values in sgroup_data.items():
            maxval = max([maxval, max(replicate_values)])
    return int(maxval)


def make_graph(graph_id: str, defaults: dict, plot_data: dict, title: str, table_type: str, num_per_row: int = 2) -> Graph:
    sample_groups: list = sorted(list(plot_data.keys()))
    num_plots: int = len(sample_groups)
    rows: int = int(num_plots/num_per_row)
    if num_plots % num_per_row != 0:
        rows += 1
    fig: go.Figure = make_subplots(
        rows=rows,
        cols=num_per_row,
        subplot_titles=sample_groups,
        x_title=f'Value deviation from mean ({table_type})',
        # y_title='Percent of values',
        y_title='Count',
        # 0.05 # 3 riviä = 0.05, 2 = 0.1 5 = 0.025
        vertical_spacing=0.1/max(1, (rows-1))
    )
    xmax: int = get_max(plot_data)
    xsize: int = int((xmax*2)/50)
    xbins: dict = {'start': -xmax, 'end': xmax, 'size': xsize}
    current_row: int = 1
    current_col: int = 1
    for sgroup in sample_groups:
        sgroup_data: dict = plot_data[sgroup]
        for replicate, rep_list in sgroup_data.items():
            fig.add_trace(
                go.Histogram(
                    x=rep_list,
                    name=replicate,
                    showlegend=False,
                    xbins=xbins,
                    # histnorm='percent'
                ),
                row=current_row,
                col=current_col
            )
        current_col += 1
        if current_col > num_per_row:
            current_col = 1
            current_row += 1
    fig.update_layout(
        barmode='overlay',
        title=title,
        height=defaults['height'] * (rows) * 0.5,
        width=defaults['width'],
        hovermode='x unified'
    )

    # xmax = int(xmax*0.25)
    tick_distance: int = max(1, round(xmax/4))
    fig.update_xaxes(range=[-xmax, xmax], dtick=tick_distance)
    fig.update_traces(opacity=0.5)
    return Graph(id=graph_id, figure=fig, config=defaults['config'])
