from dash_bootstrap_components import Tooltip


def generic_tooltip(target: str, text: str) -> Tooltip:
    return Tooltip(
        children=text,
        target=target,
        placement='top',
        style={'text-transform': 'none'}
    )


def na_tooltip(target='filtering-label') -> Tooltip:
    return generic_tooltip(target, 'Discard proteins that are not present in at least N percent of at least one replicate group. E.g. drop proteins that were only seen in one replicate of one sample.')

def interactomics_select_top_controls_tooltip(target='interactomics-num-controls') -> Tooltip:
    return generic_tooltip(target, 'Limit the number of inbuilt control runs to a specified number of most-similar runs (by euclidean distance). Increasing the number of control runs increases the SAINT running time, sometimes massively, sometimes barely, but does result in lower number of HCIs.')

def force_svenn_tooltip(target = 'sidebar-force-supervenn') -> Tooltip:
    return generic_tooltip(target, 'This option will force the use of supervenn in commonality plot, instead of deciding between supervenn and a heatmap depending on the number of sample groups.')

