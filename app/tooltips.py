
import dash_bootstrap_components as dbc

def na_tooltip() -> dbc.Tooltip:
    return dbc.Tooltip(
        children = [
            'Discard proteins that are not present in at least N percent of at least one replicate group. E.g. drop proteins that were only seen in one replicate of one sample.'
        ],
        target='filtering-label',
        placement='top',
        style={'text-transform': 'none'}
    )