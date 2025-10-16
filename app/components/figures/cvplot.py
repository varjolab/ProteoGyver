import plotly.graph_objects as go
from dash.dcc import Graph
import pandas as pd

def make_graph(raw_data: pd.DataFrame, sample_groups: dict, replicate_colors: dict, defaults: dict, id_name: str):
    # Dictionary to store CVs for each sample group
    group_cvs = {}
    group_means = {}
    group_stds = {}
    # Calculate CVs separately for each sample group
    # This will drop proteins with only one non-NA value in the group.
    for sg, group_cols in sample_groups.items():
        means = raw_data[group_cols].mean(axis=1)
        stds = raw_data[group_cols].std(axis=1)

        # Drop proteins with only one non-NA value in the group: std cannot be calculated.
        means = means[stds.notna()]
        stds = stds[stds.notna()]

        cv_percent = (stds / means) * 100
        
        group_cvs[sg] = cv_percent
        group_means[sg] = means
        group_stds[sg] = stds

    # Create violin plot
    fig = go.Figure()

    # Calculate max CV to set y-axis range
    max_cv = max([max(cvs) for cvs in group_cvs.values()])
    y_max = ((int(max_cv) // 10) + 1) * 10  # Round up to nearest 10
    annotations = []


    for sg in sample_groups.keys():
        values = list(group_cvs[sg])
        
        mean_val = pd.Series(values).mean()
        fig.add_trace(go.Violin(
            y=values,
            name=sg,
            box_visible=True,
            meanline_visible=True,
            fillcolor=replicate_colors['sample groups'][sg].replace(', 1)', ', 0.4)'),  # More transparent fill (0.5 -> 0.3)
            line_color='black',#replicate_colors['sample groups'][sg],
            line=dict(width=1),  # Add thinner line width
            points=False  # Remove outliers
        ))
        # Add mean annotation
        annotations.append(dict(
            x=sg,
            y=mean_val*1.2,
            text=f"Mean: {mean_val:.1f}%",
            showarrow=False,
            yshift=10,
            font=dict(color='black')
        ))

    width: int = defaults['width']
    if 'min_width_per' in defaults and defaults['min_width_per'] > 0:
        target_width = defaults['side_width'] + defaults['min_width_per']*4*len(sample_groups.keys())
        if width < target_width:
            width = target_width
    fig.update_layout(
        autosize=False,
        height=defaults['height'],
        width=width,
        yaxis=dict(
            title='%CV',
            tickmode='linear',
            tick0=0,
            dtick=10,  # Set tick interval to 10
            range=[0, y_max]  # Set range from 0 to rounded max
        ),
        showlegend=True,
        violingap=0.2,
        violinmode='overlay',
        annotations=annotations
    )
    out_data = {
        'group_means': {sg: means.to_dict() for sg, means in group_means.items()},
        'group_cvs': {sg: cvs.to_dict() for sg, cvs in group_cvs.items()},
        'group_stds': {sg: stds.to_dict() for sg, stds in group_stds.items()}
    }
    
    return (Graph(config=defaults['config'], figure=fig, id=id_name), out_data)