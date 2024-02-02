from plotly import graph_objects as go
from dash.dcc import Graph
import json
from plotly import io as pio

def tic_figure(defaults:dict, traces: dict, datatype: str = 'TIC', height: int = None, width: int = None):
    if height is None:
        height: int = defaults['height']
    if width is None:
        width: int = defaults['width']
    tic_figure: go.Figure = go.Figure()
    max_x: float = traces[datatype.lower()]['max_x']
    max_y: float = traces[datatype.lower()]['max_y']
    for trace in traces[datatype.lower()]['traces']:
        #print(json.dumps(trace, indent=2))
        tic_figure.add_traces(pio.from_json(json.dumps(trace))['data'][0])
    tic_figure.update_layout(
        height=400,
        xaxis_range=[0,max_x],
        yaxis_range=[0,max_y],
        margin=dict(l=5, r=5, t=20, b=5)
    )
    return tic_figure