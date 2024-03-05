import plotly.graph_objects as go
from dash.dcc import Graph
import pandas as pd

def make_graph(imputed: pd.DataFrame, sample_groups: dict, replicate_colors: dict, defaults: dict, id_name: str):
    assert imputed.isna().sum().sum() == 0, 'No missing values allowed'
    all_cvs = []
    all_text = []
    all_abundances = []
    colors = []
    for sg, group_cols in sample_groups.items():
        means = imputed[group_cols].mean(axis=1)
        stds = imputed[group_cols].std(axis=1)
        cv_percent = (stds / means) * 100
        all_cvs.extend(list(cv_percent))
        all_abundances.extend(list(means))
        all_text.extend([f'{i} in {sg}' for i in imputed.index.values])
        colors.extend(
            [
                replicate_colors['sample groups'][sg].replace(', 1)',', 0.5)')
            ]*imputed.shape[0]
        )
    x = all_abundances
    y = all_cvs
    styles = {
        'xy1': {'zeroline': False, 'domain': [0,0.85], 'showgrid': False},
        'xy2': {'zeroline': False, 'domain': [0.85,1], 'showgrid': False},
        'histomarker': {'color': 'rgba(100,0,100, 1)'},
        'scattermarker': {'size': 4, 'color': colors},
    }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
            text=all_text, 
            x = x,
            y = y,
            xaxis = 'x',
            yaxis = 'y',
            mode = 'markers',
            marker = styles['scattermarker'],
        ))
    fig.add_trace(go.Histogram(
            y = y,
            xaxis = 'x2',
            marker = styles['histomarker']
        ))
    fig.add_trace(go.Histogram(
            x = x,
            yaxis = 'y2',
            marker = styles['histomarker']
        ))
    fig.update_layout(
        autosize = False,
        xaxis = styles['xy1']|{'title': 'Mean Value'}, yaxis=styles['xy1']|{'title': '%CV'}, xaxis2=styles['xy2'],yaxis2=styles['xy2'],
        height = defaults['height'],
        width = defaults['width'],
        bargap = 0,
        hovermode = 'closest',
        showlegend = False,
    )
    out_data = {'Means': means.to_dict(), 'CV': cv_percent.to_dict(), 'std': stds.to_dict()}
    return (Graph(config=defaults['config'],figure=fig, id=id_name), out_data)