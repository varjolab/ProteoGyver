from dash_bootstrap_components import Tooltip


def na_tooltip(target='filtering-label') -> Tooltip:
    return Tooltip(
        children=[
            'Discard proteins that are not present in at least N percent of at least one replicate group. E.g. drop proteins that were only seen in one replicate of one sample.'
        ],
        target=target,
        placement='top',
        style={'text-transform': 'none'}
    )


def interactomics_select_top_controls_tooltip(target='interactomics-num-controls') -> Tooltip:
    return Tooltip(
        children=[
            'Limit the number of inbuilt control runs to a specified number of most-similar runs (by euclidean distance). There is rarely a benefit to using more than 25 control runs in SAINTexpress. Increasing the number of control runs increases the SAINT running time, sometimes massively, sometimes barely.'
        ],
        target=target,
        placement='top',
        style={'text-transform': 'none'}
    )
