import pandas as pd
from plotly import graph_objects as go
import numpy as np
from dash.dcc import Graph
from plotly import express as px

def generate_msmic_dataframes(saint_data:pd.DataFrame, reference_data: pd.DataFrame, plot_min: int = 0, plot_max: int = 100) -> tuple:

    #baitnorm = []
    baitsumnorm = []

    #db_bait_max = {}
    db_bait_sum= {}
    for b in saint_data['Bait'].unique():
    #    db_bait_max[b] = max(saint_data[saint_data['Bait']==b]['AvgSpec'].values)
        db_bait_sum[b] = sum(saint_data[saint_data['Bait']==b]['AvgSpec'].values)
    for _,row in saint_data.iterrows():
    #    baitnorm.append(row['AvgSpec']/db_bait_max[row['Bait']])
        baitsumnorm.append(row['AvgSpec']/db_bait_sum[row['Bait']])
    #saint_data['Bait norm'] = baitnorm    
    saint_data['Bait sumnorm'] = baitsumnorm
    index = sorted(list(saint_data['Bait'].unique()))
    cols = sorted(list(reference_data['Loc'].unique()))
    #data_rows = []
    data_rows_sum = []
    #loc_data = {c: {} for c in cols}
    loc_data_sum = {c: {} for c in cols}
    for c in cols:
        ld = reference_data[reference_data['Loc']==c]
        for prey in ld['Prey'].unique():
    #        loc_data[c][prey] = ld[ld['Prey']==prey]['Loc_norm'].mean()
            loc_data_sum[c][prey] = ld[ld['Prey']==prey]['Loc_sumnorm'].mean()
    for i in index:
    #    data_rows.append([])
        data_rows_sum.append([])
        for c in cols:
            bait_data = saint_data[saint_data['Bait']==i]
    #        loc_max = 0.0
            loc_sum = 0.0
            for _,row in bait_data.iterrows():
                #if row['Prey'] in loc_data[c]:
    #                loc_max += loc_data[c][row['Prey']]*row['Bait norm']
                if row['Prey'] in loc_data_sum[c]:
                    loc_sum += loc_data_sum[c][row['Prey']]*row['Bait sumnorm']
    #        data_rows[-1].append(loc_max)
            data_rows_sum[-1].append(loc_sum)
    #bd_max = pd.DataFrame(index=index,columns=cols,data=data_rows)
    bd_sum = pd.DataFrame(index=index,columns=cols,data=data_rows_sum).fillna(0)
    #bd_max = bd_max.div(bd_max.max(axis=1),axis=0)*plot_max
    bd_sum = bd_sum.div(bd_sum.max(axis=1),axis=0)*plot_max
    #bd_max = bd_max.apply(round).astype(int)
    bd_sum = bd_sum.apply(round).astype(int)

    #return bd_max
    return bd_sum
    
def tweak_fig_size_hw(height: int, width: int, desired_ratio: float, method='reduce') -> tuple:
    current_ratio: float = height/width
    if method == 'reduce':
        height = height * (desired_ratio/current_ratio)
    elif method == 'inflate':
        width = width * (current_ratio/desired_ratio)    
    return (height, width)

def draw_localization_plot(defaults: dict, datarow: pd.Series, cmap: list = [[255,255,255], [0,0,255]], nsteps: int = 10, plot_min: int = 0, plot_max: int = 100):
    r = list(datarow.values)
    width = int(360/len(r))
    theta = list(range(0,360,width))
    colors = []
    
    color_vals = {}
    target = np.array(cmap[1])
    start = np.array(cmap[0])
    step_v = (target-start)/nsteps
    step_width = (plot_max-plot_min)/nsteps
    color_vals = {
        round(i*step_width,1): (i*step_v) + start for i in range(1, nsteps+1) 
    }
    color_vals = [
        (k, '#' + ''.join(
            [
                hex(int(e))[2:].rjust(2,'0')
                for e in v
            ])
        ) for k, v in color_vals.items()
    ]
    color_vals.sort(key=lambda x: x[0])

    # Tweak figure size to conform to the constraints of the ms-microscopy plot.
    polar_plot_ratio = 0.76
    fig_height, fig_width = tweak_fig_size_hw(defaults['height'], defaults['width'], polar_plot_ratio)

    for dvalue in r:
        for limval, color in color_vals:
            if dvalue <= limval:
                colors.append(color)
                break
    fig = go.Figure(go.Barpolar(
        r=r,
        theta=theta,
        width=width,
        marker_color=colors,
        marker_line_color="black",
        marker_line_width=1,
        opacity=1
    ))

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[plot_min, plot_max], tickvals = list(range(plot_min, plot_max, int(step_width))), showticklabels=False, ticks=''),
            angularaxis = dict(showticklabels=True, ticks='',tickmode='array', tickvals=theta, ticktext = datarow.index)
        ),
        height=fig_height, 
        width = fig_width,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_polars(radialaxis_showline=False)
    return fig

def localization_graph(graph_id: str, defaults: dict, plot_type: str, *args, **kwargs) -> Graph:
    if plot_type == 'polar':
        return Graph(
            id=graph_id,
            config=defaults['config'],
            figure=draw_localization_plot(
                defaults,
                *args,
                **kwargs
            )
        )
    elif plot_type == 'heatmap':
        return Graph(
                id=graph_id,
                config=defaults['config'],
                figure=draw_localization_heatmap(
                    defaults,
                    *args,
                    **kwargs
                )
            )
    return 'NOT A VALID MSMIC PLOT TYPE'

def draw_localization_heatmap(defaults: dict, localization_results: pd.DataFrame) -> go.Figure:
    fig_height: int = defaults['height']
    fig_width: int = defaults['width']
    fig = px.imshow(localization_results, height=fig_height, width = fig_width, aspect='auto', color_continuous_scale='Blues')
    fig.update_xaxes(side="top",tickangle=270)
    fig.update_layout(margin=dict(l=2, r=2, t=2,b=2))
    return fig