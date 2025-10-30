"""
Total ion chromatogram (TIC) and base peak figure utilities.

Builds a Plotly figure from precomputed JSON traces with automatic sizing
and hover behavior.
"""
from plotly import graph_objects as go
from dash.dcc import Graph
import json
from plotly import io as pio

def tic_figure(defaults:dict, traces: dict, datatype: str = 'TIC', height: int = None, width: int = None):
    """Create a TIC/trace figure from serialized Plotly traces.

    :param defaults: Dict with ``height`` and ``width`` fallbacks.
    :param traces: Dict with keys for datatypes; each contains ``max_x``, ``max_y``, and a list of ``traces`` as Plotly JSON.
    :param datatype: Which trace set to render (e.g., ``'TIC'``).
    :param height: Optional explicit height.
    :param width: Optional explicit width.
    :returns: Plotly ``Figure`` with added traces and layout.
    """
    if height is None:
        use_height: int = defaults['height']
    else:
        use_height: int = height
    if width is None:
        use_width: int = defaults['width']
    else:
        use_width: int = width
    tic_figure: go.Figure = go.Figure()
    max_x: float = traces[datatype]['max_x']
    max_y: float = traces[datatype]['max_y']
    for trace in traces[datatype]['traces']:
        #print(json.dumps(trace, indent=2))
        tic_figure.add_traces(pio.from_json(json.dumps(trace))['data'][0])
    hmode: str = 'closest'
    if len(traces[datatype]['traces']) < 10:
        hmode = 'x unified' # Gets cluttered with too many traces
    tic_figure.update_layout(
        height=use_height,
        width=use_width,
        xaxis_range=[0,max_x],
        yaxis_range=[0,max_y],
        margin=dict(l=5, r=5, t=20, b=5),
        hovermode = hmode
    )
    return tic_figure