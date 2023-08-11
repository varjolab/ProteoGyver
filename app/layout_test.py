""" Restructured frontend for proteogyver app"""

import os
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
from components.ui_components import main_sidebar, table_of_contents, main_content_div

# the style arguments for the sidebar. We use position:fixed and a fixed width
import DbEngine


#db = DbEngine()
app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
figure_templates = ['1','2']
implemented_workflows = ['3','4']
@callback(
    Output('toc-div','children'),
    Input('contents-div','children'),
    prevent_initial_call=True
)
def update_toc(contents):
    return table_of_contents(contents)

@callback(
    Output('contents-div', 'children', allow_duplicate=True),
    Input('call1-button','n_clicks'),
    State('contents-div','children'),
    prevent_initial_call=True
)
def call1(_, contents) -> html.Div:
    ret_text = "Nulla ut erat quis enim dignissim egestas. Quisque id porttitor leo. Aliquam vitae euismod massa, vel posuere mi. Curabitur vulputate pretium metus sed placerat. Nulla facilisi. Pellentesque sit amet sapien at leo cursus ullamcorper. Etiam quis sodales felis, hendrerit lobortis neque."
    new_contents = []
    idstr = 'call1-div'
    for element in contents:
        needed = True
        try:
            if element['props']['id']==idstr:
                needed = False
        except KeyError:
            pass
        if needed:
            new_contents.append(element)
    new_contents.append(
        html.Div(id=idstr, children=[
            html.H2(id='h12',children=ret_text[:12]),
            html.H3(id='h13',children=ret_text[24:36]),
            html.H2(id='h13',children=ret_text[36:48]),
            html.H3(id='h13',children=ret_text[48:55]),
            html.H4(id='h13',children=ret_text[48:55]),
            html.H3(id='h13',children=ret_text[48:55]),
            html.H2(id='h13',children=ret_text[55:65]),
            html.H3(id='h13',children=ret_text[55:65]),
            html.P(id='p1',children=ret_text)
        ])
    )     
    return new_contents

app.layout = html.Div([main_sidebar(figure_templates, implemented_workflows), main_content_div()])

if __name__ == '__main__':
    app.run(debug=True)
