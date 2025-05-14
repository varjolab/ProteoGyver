import plotly.graph_objects as go
from dash.dcc import Graph
import pandas as pd

def make_graph(raw_data: pd.DataFrame, sample_groups: dict, replicate_colors: dict, defaults: dict, id_name: str):
    # Dictionary to store CVs for each sample group
    group_cvs = {}
    group_means = {}
    group_stds = {}
    # Calculate CVs separately for each sample group
    for sg, group_cols in sample_groups.items():
        means = raw_data[group_cols].mean(axis=1)
        stds = raw_data[group_cols].std(axis=1)
        cv_percent = (stds / means) * 100

        group_cvs[sg] = cv_percent
        group_means[sg] = means
        group_stds[sg] = stds

    fig = go.Figure()
    max_cv = max(max(cvs) for cvs in group_cvs.values())
    y_max = ((int(max_cv) // 10) + 1) * 10  # Round up to nearest 10
    annotations = []

    for i, sg in enumerate(sample_groups.keys()):
        values = list(group_cvs[sg])
        mean_val = pd.Series(values).mean()

        fig.add_trace(go.Box(
            y=values,
            name=sg,
            boxpoints='outliers',  # No outliers
            marker_color='black',
            line=dict(color='black'),
            fillcolor=replicate_colors['sample groups'][sg],#.replace(', 1)', ', 0.4)'),  # More transparent fill (0.5 -> 0.3)
            line_color='black',#replicate_colors['sample groups'][sg],
            #meanline=dict(visible=True, color='black', width=2),
            #boxmean='sd',  # show mean + standard deviation
            
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=1)
        )

        # Add mean annotation
        annotations.append(dict(
            x=sg,
            y=mean_val*1.2,
            text=f"Mean: {mean_val:.1f}%",
            showarrow=False,
            yshift=10,
            font=dict(color='black')
        ))

    fig.update_layout(
        autosize=False,
        height=defaults['height'],
        width=defaults['width'],
        yaxis=dict(
            title='%CV',
            tickmode='linear',
            tick0=0,
            dtick=10,
            range=[0, y_max]
        ),
        showlegend=False,
        annotations=annotations
    )
    
    out_data = {
        'group_means': {sg: means.to_dict() for sg, means in group_means.items()},
        'group_cvs': {sg: cvs.to_dict() for sg, cvs in group_cvs.items()},
        'group_stds': {sg: stds.to_dict() for sg, stds in group_stds.items()}
    }
    
    return (Graph(config=defaults['config'], figure=fig, id=id_name), out_data)