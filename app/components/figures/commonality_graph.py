
import io
import base64
import matplotlib as mpl
from matplotlib import pyplot as plt
from supervenn import supervenn as svenn
from dash.dcc import Graph
from dash.html import Img
from pandas import DataFrame
from plotly.express import imshow
from numpy import nan as NA


def supervenn(group_sets: dict, id_str: str) -> Img:
    """Draws a super venn plot for the input data table.

    See https://github.com/gecko984/supervenn for details of the plot.
    Parameters:
    data_table: table of samples (columns) and measurements(rows)
    rev_sample_groups: dictionary of {sample_column_name: sample_group_name} containing all sample columns.
    figure_name: name for the figure title, as well as saved file
    save_figure: Path to save the generated figure. if None (default), figure will not be saved.
    save_format: format for the saved figure. default is svg.

    Returns:
    returns html.Img object containing the figure data in png form.
    """

    # Buffer for use
    buffer: io.BytesIO = io.BytesIO()
    fig: mpl.figure
    axes: mpl.Axes
    fig, axes = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)

    plot_sets: list = []
    plot_setnames: list = []
    for set_name, set_proteins in group_sets.items():
        plot_sets.append(set(set_proteins))
        plot_setnames.append(set_name)
    svenn(
        plot_sets,
        plot_setnames,
        ax=axes,
        rotate_col_annotations=True,
        col_annotations_area_height=1.2,
        widths_minmax_ratio=0.1
    )
    plt.xlabel('Shared proteins')
    plt.ylabel('Sample group')
    plt.savefig(buffer, format="png")
    plt.close()
    data: str = base64.b64encode(buffer.getbuffer()).decode("utf8")  # encode to html elements
    buffer.close()
    return Img(id=id_str, src=f'data:image/png;base64,{data}')

def common_heatmap(group_sets: dict, id_str: str, defaults) -> Graph:
    hmdata: list = []
    index: list = list(group_sets.keys())
    done = set()
    for gname in index:
        hmdata.append([])
        for gname2 in index:
            val: float
            if gname == gname2:
                val = NA
            nstr: str = ''.join(sorted([gname,gname2]))
            if nstr in done:
                val = NA
            else:
                val = len(group_sets[gname] & group_sets[gname2])
            hmdata[-1].append(val)
    return Graph(id = id_str, figure = imshow(
        DataFrame(data=hmdata,index=index,columns=index),
        config=defaults['config'],
        height=defaults['height'],
        width=defaults['width']
        ))

def make_graph(group_sets: dict, id_str: str, defaults:dict) -> Img | Graph:
    if len(group_sets.keys()) <= 6:
        return supervenn(group_sets, id_str)
    else:
        return common_heatmap(group_sets, id_str, defaults)