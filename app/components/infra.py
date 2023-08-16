"""Infrastructure components for Proteogyver"""

from dash import dcc, html

def data_stores() -> html.Div:
    """Returns all the needed data store components"""
    return html.Div([
        dcc.Store(id='uploaded-data-table-info'),
        dcc.Store(id='uploaded-data-table'),
        dcc.Store(id='uploaded-sample-table-info'),
        dcc.Store(id='uploaded-sample-table'),
        dcc.Store(id='upload-data-store'),
        dcc.Store(id='replicate-colors'),
        dcc.Store(id='discard-samples'),
        dcc.Store(id='count-data-store'),
        dcc.Store(id='coverage-data-store'),
        dcc.Store(id='reproducibility-data-store'),
        dcc.Store(id='missing-data-store'),
        dcc.Store(id='sum-data-store'),
        dcc.Store(id='mean-data-store'),
        dcc.Store(id='distribution-data-store'),
        dcc.Store(id='commonality-data-store'),
    ])

def notifiers() -> html.Div:
    """Returns divs used for various callbacks only."""
    return html.Div(
        id='notifiers-div',
        children=[
            html.Div(id='qc-done-notifier'),
            html.Div(id='workflow-done-notifier')
        ],
        hidden=True
    )