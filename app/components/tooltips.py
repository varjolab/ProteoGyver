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

def use_svenn_tooltip(target = 'sidebar-force-supervenn') -> Tooltip:
    return generic_tooltip(target, 'This option will force the use of supervenn in commonality plot, instead of deciding between supervenn and a heatmap depending on the number of sample groups.')

def rescue_tooltip(target = 'interactomics-rescue-filtered-out') -> Tooltip:
    return generic_tooltip(target, 'Rescue will let preys pass filter if they pass the filter with any other bait. This means that BFDR values of e.g. 0.9 can pass, if the prey has a BFDR value of 0.0 with some other bait. !!!NOTE!!! This feature works well for small interactomics datasets, and related baits. With large and unrelated bait sets, it will enrich contaminants, that would otherwise be caught by filter in most baits.')

def test_type_tooltip(target = 'proteomics-test-type') -> Tooltip:
    return generic_tooltip(target, "ONLY USE paired IF you know what you're doing and your samples will be in the same order across sample groups, and you can be CERTAIN of it.")