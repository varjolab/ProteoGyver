from dash_bootstrap_components import Tooltip

def na_tooltip() -> Tooltip:
    return Tooltip(
        children = [
            'Discard proteins that are not present in at least N percent of at least one replicate group. E.g. drop proteins that were only seen in one replicate of one sample.'
        ],
        target='filtering-label',
        placement='top',
        style={'text-transform': 'none'}
    )