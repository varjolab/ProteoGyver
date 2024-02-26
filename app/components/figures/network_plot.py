from dash import dcc, html, callback
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
# Load extra layouts
cyto.load_extra_layouts()

def get_cytoscape_elements_and_ints(interaction_data):
    cy_edges = []
    cy_nodes = []
    keepi = []
    for b in interaction_data['Bait'].unique():
        keepi.extend(list(interaction_data[interaction_data['Bait']==b].head(100).index))
    interaction_data = interaction_data[interaction_data.index.isin(keepi)]
    interactions = {}
    added: set = set()
    for _,row in interaction_data.iterrows():
        bname = row['Bait']
        pname = row['Prey']
        if bname not in interactions:
            interactions[bname] = {}
        interactions[bname][pname] = (row['PreyGene'], row['AvgSpec'])
        cy_edge = {'data': {'source': bname, 'target': pname, 'int': f'{bname}-{pname}'}}
        pnode = {'data': {'id': pname, 'label': row['PreyGene']}, 'classes': 'preyNode'}
        bnode = {'data': {'id': bname, 'label': bname}, 'classes': 'baitNode'}
        if pname not in added:
            added.add(pname)
            cy_nodes.append(pnode)
        if bname not in added:
            added.add(bname)
            cy_nodes.append(bnode)
        cy_edges.append(cy_edge)
    created_elements = cy_nodes#[genesis_node]
    created_elements.extend(cy_edges)

    return (created_elements, interactions)

def get_stylesheet():
    return [
        {"selector": "node", "style": {"opacity": 0.65, "z-index": 9999}},
        {
            "selector": "edge",
            "style": {"curve-style": "bezier", "opacity": 0.45, "z-index": 5000},
        },
        {"selector": ".preyNode", "style": {"font-size": 12, "background-color": "#0074D9", "label": "data(label)"}},
        {
            "selector": ".preyEdge",
            "style": {
                "line-color": "#0074D9",
            },
        },
        {"selector": ".baitNode", "style": {"font-size": 12, "background-color": "#FF4136", "label": "data(label)"}},
        {
            "selector": ".baitEdge",
            "style": {
                "line-color": "#FF4136",
            },
        },
        {
            "selector": ":selected",
            "style": {
                "border-width": 2,
                "border-color": "black",
                "border-opacity": 1,
                "opacity": 1,
                "color": "black",
                "font-size": 12,
                "z-index": 9999,
            },
        },
    ]

def get_cytoscape_container(cyto_elements, full_height): 
    default_stylesheet = get_stylesheet()
    cyto_layouts = [
        "grid",
        "circle",
        "concentric",
        "breadthfirst",
        "cose",
        #"cose-bilkent", # Does not work for now, next version should fix.
        "dagre",
        "cola",
        "klay",
        "spread",
        "euler"
    ]
    return html.Div([
        html.Div([
            html.Div([
                cyto.Cytoscape(
                    id="cytoscape",
                    elements=cyto_elements,
                    stylesheet=default_stylesheet,
                       style={'height': full_height, "width": "100%"},
                )
            ],style={'width': '69%', 'display': 'inline-block'}),
            html.Div([
                html.P(children=f"Layout:", style={"margin-left": "3px"}),
                dcc.Dropdown(
                    id='dropdown-layout',
                    options = [
                        {'label': val.capitalize(), 'value': val} for val in cyto_layouts
                    ],
                    clearable=False,
                    value = cyto_layouts[0]
                ),
                html.Div(id='nodedata-div', style = {'padding-top': '15px'})
            ], style={'vertical-align': 'top', 'width': '30%','display': 'inline-block'})
        ], style={
            'height': full_height,
            #'overflow': 'auto',
            'margin': '5px'
        })
    ])
