"""
Commonality visualizations for sample group overlaps.

Provides two representations:
- ``supervenn``: Supervenn diagram for up to ~10 sets
- ``common_heatmap``: Jaccard-like overlap heatmap for larger numbers
"""
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
    """Render a Supervenn plot from named sets and return as a Dash image.

    See `supervenn <https://github.com/gecko984/supervenn>`_.

    :param group_sets: Mapping of group name to a set of identifiers.
    :param id_str: Component ID for the returned image.
    :returns: Tuple ``(dash.html.Img, pdf_data_base64)``.
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
    """Create a heatmap of pairwise overlap ratios between groups.

    The value for groups ``A`` and ``B`` is ``|A∩B| / |A∪B|`` (Jaccard index).

    :param group_sets: Mapping of group name to a set of identifiers.
    :param id_str: Component ID for the ``Graph``.
    :param defaults: Figure defaults (expects ``height``, ``width``, ``config``).
    :returns: Tuple ``(dash.dcc.Graph, '')``.
    """
    hmdata: list = []
    index: list = list(group_sets.keys())
    done = set()
    config = defaults['config'].copy()
    config['toImageButtonOptions'] = config['toImageButtonOptions'].copy()
    config['toImageButtonOptions']['filename'] = 'Shared identifications'
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
            config=config
        ),
        ''
    )


def make_graph(group_sets: dict, id_str: str, use_supervenn: bool, defaults: dict) -> tuple:
    """Choose and build an overlap visualization based on group count.

    :param group_sets: Mapping of group name to a set of identifiers.
    :param id_str: Component ID for the output figure.
    :param use_supervenn: If ``True``, prefer Supervenn when feasible.
    :param defaults: Figure defaults (expects ``height``, ``width``, ``config``).
    :returns: Tuple of (component, aux_data) where aux_data may be a base64 PDF.
    """
    if len(group_sets.keys()) > 10:
        use_supervenn = False
    if use_supervenn:
        return supervenn(group_sets, id_str)
    else:
        return common_heatmap(group_sets, id_str, defaults)
