import os
from dash import Dash, html, dcc
from dash import dash_table
import plotly.express as px
import pandas as pd
import json
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template


app = Dash(__name__,external_stylesheets=[dbc.themes.SOLAR])
load_figure_template('SOLAR')
server = app.server

parameterfile = 'parameters.json'
with open(parameterfile) as fil:
    parameters = json.load(fil)

data = pd.read_csv(parameters['Run data'],sep='\t',index_col='Index')

#graph = px.bar(data, x='Fruit', y='Amount', color='City', barmode='group')

app.layout = html.Div([
    html.H1('Data upload'),
    dcc.Tabs(id='upload-tabs',value='tab1-data-entry',children=[
        dcc.Tab(label='Data entry',value='tab1-data-entry'),
        dcc.Tab(label='Current data',value='tab2-current-data'),
    ]),
    html.Div(id='data-app-tabs')
])

@app.callback(Output('data-app-tabs','children'),Input('upload-tabs', 'value'))

def render_content(tab):
    if tab == 'tab1-data-entry':
        return html.Div([
            html.H3('Upload data'),
            ['Here a button to upload experimental design', 'And a button to fill out experimental design manually', 'And a button to proceed to analysis']
        ])
    elif tab == 'tab2-current-data':
        return html.Div([
            dash_table.DataTable(
                data.to_dict('records'), [{'name': i, 'id': i} for i in data.columns]
            )
        ])

if __name__ == '__main__':
    app.run(port='8050', debug=True)
    