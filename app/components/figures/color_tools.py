import numpy as np
from matplotlib import pyplot as plt

def get_assigned_colors(sample_group_dict: dict) -> dict:
    """Returns a dictionary with each sample name from the sample group dict assigned a color\
        corresponding to its sample group.

    :param sample_group_dict: dictionary of sample groups, where one group maps to a list of samples.
    :returns: tuple of (dictionary of samples and sample groups,dictionary of samples and sample groups with contaminant annotation)
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
    """Darkens a given color by a given percentage value).
    :param color: input color as  "rgb(123,321,123)"
    :param percent: percentage for how much to darken
    :returns: new color string in the  "rgb(123,321,123)" format.
    """
    tp: str
    col_ints:list
    tp, col_ints = color.split('(')
    col_ints = [int(x) for x in col_ints.split(')')[0].split(',')]
    multiplier: float = ((100-percent)/100)
    col_ints = [str(int(c*multiplier)) for c in col_ints]
    return f'{tp}({",".join(col_ints)})'

def get_cut_colors(colormapname: str = 'gist_ncar', number_of_colors: int = 15,
                cut: float = 0.4) -> list:
    """Returns cut colors from the given colormapname

    :param colormap: which matplotlib colormap to use
    :param number_of_colors: how many colors to return. Colors will be equally spaced in the map
    :param cut: how much to cut the colors.
    :returns: cut color list
    """
    number_of_colors += 1
    colors: list = (1. - cut) * (plt.get_cmap(colormapname)(np.linspace(0., 1., number_of_colors))) + \
        cut * np.ones((number_of_colors, 4))
    colors = colors[:-1]
    return colors
