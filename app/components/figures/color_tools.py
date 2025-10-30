"""
Color utilities for Plotly/Dash figures.

Includes helpers to translate color formats, darken colors, trim overly
light template colors, and assign consistent group/sample colors.
"""
import numpy as np
from matplotlib import pyplot as plt
import plotly.io as pio
import plotly.colors as pc

def rgba_to_hex(rgba: str) -> str:
    """Convert an RGBA color string to hex.

    :param rgba: Input as ``"rgba(r,g,b,a)"``.
    :returns: Hex color string, e.g. ``"#RRGGBB"``.
    """
    return f'#{rgba.split("(")[1].split(")")[0].split(",")[0:3]}'

def remove_unwanted_colors(figure_template: str) -> str:
    """Return a Plotly template with overly light colors darkened.

    The function reads the colorway from the named template and adjusts
    entries with high luminance to improve visibility on light
    backgrounds.

    :param figure_template: Name of a Plotly template (e.g., ``"plotly"``).
    :returns: Modified template object with updated ``layout.colorway``.
    """

    # grab the current template
    tmpl = pio.templates[figure_template]

    # take its colorway or fallback to default
    colorway = list(tmpl.layout.colorway or pio.templates["plotly"].layout.colorway)

    def too_light(rgb):
        r, g, b = pc.hex_to_rgb(rgb)
        # compute perceived brightness 0–255
        luminance = 0.299*r + 0.587*g + 0.114*b
        return luminance > 210   # tweak threshold (230 ≈ very pale)

    # replace any light color with a darker variant
    safe_colors = [
        rgb if not too_light(rgb) else pc.label_rgb(tuple(max(0, c-60) for c in pc.hex_to_rgb(rgb)))
        for rgb in colorway
    ]

    tmpl.layout.colorway = safe_colors
    return tmpl

def get_assigned_colors(sample_group_dict: dict) -> dict:
    """Assign per-group and per-sample colors.

    :param sample_group_dict: Mapping of group name to list of sample names.
    :returns: Tuple ``(colors, colors_with_contaminant)`` where both are dicts
        with keys ``'samples'`` and ``'sample groups'``.
    """
    entry_list: list = list(sample_group_dict.keys())
    colors: list = get_cut_colors(number_of_colors=len(entry_list))
    group_colors: dict = {}
    for i, entry in enumerate(entry_list):
        group_colors[entry] = f'rgba({",".join(str(int(255*x)) for x in colors[i][:3])}, 1)'
    ret: dict = {'samples': {}, 'sample groups': group_colors}
    ret_cont: dict = {}
    for c in 'contaminant','non-contaminant':
        ret_cont[c] = {'samples': {}, 'sample groups': {}}
    for i, (sample_group, sample_list) in enumerate(sample_group_dict.items()):
        ret_cont['non-contaminant']['sample groups'][sample_group] = group_colors[sample_group]
        ret_cont['contaminant']['sample groups'][sample_group] = darken(group_colors[sample_group],20)
        for sample_name in sample_list:
            ret['samples'][sample_name] = group_colors[sample_group]
            ret_cont['non-contaminant']['samples'][sample_name] = group_colors[sample_group]
            ret_cont['contaminant']['samples'][sample_name] = darken(group_colors[sample_group],20)
    return (ret, ret_cont)

def darken(color: str, percent: int) -> str:
    """Darken a given RGB(A) color by a percentage.

    :param color: Color as ``"rgb(r,g,b)"`` or ``"rgba(r,g,b,a)"``.
    :param percent: Percentage to darken (0–100).
    :returns: Darkened color string of the same type.
    """
    tp: str
    col_ints:list
    tp, col_ints = color.split('(')
    col_ints = [int(x) for x in col_ints.split(')')[0].split(',')]
    multiplier: float = ((100-percent)/100)
    col_ints = [str(max(0,int(c*multiplier))) for c in col_ints]
    if len(col_ints) == 4: # Make sure alpha is not 0 (=invisible)
        col_ints[-1] = '1'
    return f'{tp}({",".join(col_ints)})'

def get_cut_colors(colormapname: str = 'gist_ncar', number_of_colors: int = 15,
                cut: float = 0.4) -> list:
    """Return a list of evenly spaced colors from a matplotlib colormap.

    :param colormapname: Matplotlib colormap name.
    :param number_of_colors: Number of colors to sample.
    :param cut: Brightness adjustment mixed with white (0–1).
    :returns: List of RGBA tuples in 0–1 range.
    """
    number_of_colors += 1
    colors: list = (1. - cut) * (plt.get_cmap(colormapname)(np.linspace(0., 1., number_of_colors))) + \
        cut * np.ones((number_of_colors, 4))
    colors = colors[:-1]
    return colors
