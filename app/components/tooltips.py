"""
Tooltip helpers for the UI.

Provides convenience wrappers for Bootstrap tooltips used throughout the
application to document inputs and options.
"""
from dash_bootstrap_components import Tooltip


def generic_tooltip(target: str, text: str) -> Tooltip:
    """Create a generic tooltip.

    :param target: ID of the target element.
    :param text: Tooltip text content.
    :returns: Bootstrap ``Tooltip`` component.
    """
    return Tooltip(
        children=text,
        target=target,
        placement='top',
        style={'text-transform': 'none','padding-top': '50px'}
    )


def na_tooltip(target='filtering-label') -> Tooltip:
    """Tooltip for NA filtering control.

    :param target: ID of the target element.
    :returns: Bootstrap ``Tooltip`` component.
    """
    return generic_tooltip(target, 'Discard proteins that are not present in at least N percent of at least one replicate group or in at least N percent of the samples. E.g. drop proteins that were only seen in one replicate of one sample.')

def interactomics_select_top_controls_tooltip(target='interactomics-num-controls') -> Tooltip:
    """Tooltip for selecting most similar inbuilt controls.

    :param target: ID of the target element.
    :returns: Bootstrap ``Tooltip`` component.
    """
    return generic_tooltip(target, 'Limit the number of inbuilt control runs to a specified number of most-similar runs (by euclidean distance). Increasing the number of control runs increases the SAINT running time, sometimes massively, sometimes barely, but does result in lower number of HCIs.')

def rescue_tooltip(target = 'interactomics-rescue-filtered-out') -> Tooltip:
    """Tooltip explaining the rescue option for SAINT filtering.

    :param target: ID of the target element.
    :returns: Bootstrap ``Tooltip`` component.
    """
    return generic_tooltip(target, 'Rescue will let preys pass filter if they pass the filter with any other bait. This means that BFDR values of e.g. 0.9 can pass, if the prey has a BFDR value of 0.0 with some other bait. !!!NOTE!!! This feature works well for small interactomics datasets of related baits, e.g. four receptors of the same family. With large and unrelated bait sets, it will enrich contaminants, that would otherwise be caught by filter in most baits. However, that can also highlight loopholes in your filtering strategy, or e.g. contaminant sources not covered by control sets.')

def test_type_tooltip(target = 'proteomics-test-type') -> Tooltip:
    """Tooltip describing test type selection (paired vs independent).

    :param target: ID of the target element.
    :returns: Bootstrap ``Tooltip`` component.
    """
    return generic_tooltip(target, "ONLY USE paired IF you know what you're doing and your samples will be in the same order across sample groups, and you can be CERTAIN of it. And then check your results to make sure they were.")

def nearest_tooltip(target = 'interactomics-nearest-control-filtering') -> Tooltip:
    """Tooltip for nearest-control selection in interactomics.

    :param target: ID of the target element.
    :returns: Bootstrap ``Tooltip`` component.
    """
    return generic_tooltip(target, 'This option will compare inbuilt control runs and choose n that are most similar to your samples. Only use if you are experiencing performance issues with full controls, as more controls do not degrade performance usually. IF you choose to use it, make sure you know which runs were selected for the final controls.')