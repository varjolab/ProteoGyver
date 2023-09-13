"""Infrastructure components for Proteogyver"""

from dash import dcc, html

DATA_STORE_IDS: list = [
    'uploaded-data-table-info',
    'uploaded-data-table',
    'uploaded-sample-table-info',
    'uploaded-sample-table',
    'upload-data-store',
    'replicate-colors',
    'replicate-colors-with-contaminants',
    'discard-samples',
    'count-data-store',
    'coverage-data-store',
    'reproducibility-data-store',
    'missing-data-store',
    'sum-data-store',
    'mean-data-store',
    'distribution-data-store',
    'commonality-data-store',
    'proteomics-na-filtered-data-store',
    'proteomics-normalization-data-store',
    'proteomics-imputation-data-store',
    'proteomics-distribution-data-store',
    'proteomics-pca-data-store',
    'proteomics-clustermap-data-store',
    'proteomics-volcano-data-store',
    'interactomics-saint-input-data-store',
    'interactomics-saint-crapome-data-store',
    'interactomics-saint-output-data-store',
    'interactomics-saint-final-output-data-store',
    'interactomics-saint-filtered-output-data-store',
]


def data_stores() -> html.Div:
    """Returns all the needed data store components"""
    return html.Div(
        id='dcc-stores',
        children=[
            dcc.Store(id=ID_STR) for ID_STR in DATA_STORE_IDS
        ]
    )

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