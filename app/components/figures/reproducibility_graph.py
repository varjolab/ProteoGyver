from pandas import DataFrame
from dash.dcc import Graph
from plotly import express as px

def get_reproducibility_dataframe(data_table: DataFrame, sample_groups: dict) -> Graph:
    figure_datapoints:list = []
    for sample_group, sample_columns in sample_groups.items():
        for i, column in enumerate(sample_columns):
            if i == (len(sample_columns)-1):
                break
            for column2 in sample_columns[i+1:]:
                figure_datapoints.extend([[r[column], r[column2], sample_group] for _,r in data_table.iterrows()])
    plot_dataframe: DataFrame = DataFrame(data=figure_datapoints, columns = ['Sample A','Sample B', 'Sample group'])
    plot_dataframe = plot_dataframe.dropna() # No use plotting data points with missing values.
    return plot_dataframe

def make_graph(graph_id: str, defaults:dict, plot_dataframe: DataFrame, title: str) -> Graph:
    plot_dataframe = plot_dataframe.dropna() # No use plotting data points with missing values.
    return Graph(id=graph_id, figure = px.density_heatmap(
        plot_dataframe,
        title=title,
        x='Sample A',
        y='Sample B',
        height=defaults['height']*(len(plot_dataframe['Sample group'].unique())/2),
        width=defaults['width'],
        color_continuous_scale='blues',
        #marginal_x = 'histogram',
        #marginal_y = 'histogram',
        facet_col= 'Sample group',
        facet_col_wrap=2,
        nbinsx=50,
        nbinsy=50,
    ))