from plotly import graph_objects as go
from dash.dcc import Graph
import json
from plotly import io as pio

def tic_figure(defaults:dict, traces: dict, datatype: str = 'TIC', height: int = None, width: int = None):
    if height is None:
        use_height: int = defaults['height']
    else:
        use_height: int = height
    if width is None:
        use_width: int = defaults['width']
    else:
        use_width: int = width
    tic_figure: go.Figure = go.Figure()
    max_x: float = traces[datatype.lower()]['max_x']
    max_y: float = traces[datatype.lower()]['max_y']
    for trace in traces[datatype.lower()]['traces']:
        #print(json.dumps(trace, indent=2))
        tic_figure.add_traces(pio.from_json(json.dumps(trace))['data'][0])
    hmode: str = 'closest'
    if len(traces[datatype.lower()]['traces']) < 10:
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