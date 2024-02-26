
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


def supervenn(group_sets: dict, id_str: str) -> tuple:
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
    mpl.use('agg')
    buffer: io.BytesIO = io.BytesIO()
    buffer2: io.BytesIO = io.BytesIO()
    fig: mpl.figure
    axes: mpl.Axes
    fig, axes = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(8)

    plot_sets: list = []
    plot_setnames: list = []
    all_proteins: set = set()
    for set_name, set_proteins in group_sets.items():
        set_proteins = set(set_proteins)
        plot_sets.append(set_proteins)
        all_proteins |= set_proteins
        plot_setnames.append(set_name)
    minwd: int = 1
    widths_minmax_ratio = 0.1
    if len(plot_sets) > 6:
        minwd = int(len(all_proteins) / 50)
        widths_minmax_ratio = None
    svenn(
        plot_sets,
        plot_setnames,
        ax=axes,
        rotate_col_annotations=True,
        col_annotations_area_height=1.2,
        widths_minmax_ratio=widths_minmax_ratio,
        min_width_for_annotation=minwd
    )
    plt.xlabel('Shared proteins')
    plt.ylabel('Sample group')
    plt.savefig(buffer, format="png")
    plt.savefig(buffer2, format="pdf")
    plt.close()
    data: str = base64.b64encode(buffer.getbuffer()).decode(
        "utf8")  # encode to html elements
    pdf_data: str = base64.b64encode(buffer2.getbuffer()).decode(
        "utf8")  # encode to html elements, this one will be used in PDF export later on.
    buffer.close()
    buffer2.close()
    return (
        Img(id=id_str, src=f'data:image/png;base64,{data}'),
        pdf_data
    )


def common_heatmap(group_sets: dict, id_str: str, defaults) -> tuple:
    hmdata: list = []
    index: list = list(group_sets.keys())
    done = set()
    for gname in index:
        hmdata.append([])
        for gname2 in index:
            val: float
            if gname == gname2:
                val = NA
            nstr: str = ''.join(sorted([gname, gname2]))
            if nstr in done:
                val = NA
            else:
                val = len(group_sets[gname] & group_sets[gname2]) / len(group_sets[gname] | group_sets[gname2])
            hmdata[-1].append(val)
    return (
        Graph(
            id=id_str,
            figure=imshow(
                DataFrame(data=hmdata, index=index, columns=index),
                height=defaults['height'],
                width=defaults['width'],
                zmin=0,
                zmax=1,
                color_continuous_scale = 'Blues'
            ),
            config=defaults['config']
        ),
        ''
    )


def make_graph(group_sets: dict, id_str: str, force_svenn: bool, defaults: dict) -> tuple:
    if force_svenn or (len(group_sets.keys()) <= 6):
        return supervenn(group_sets, id_str)
    else:
        return common_heatmap(group_sets, id_str, defaults)
