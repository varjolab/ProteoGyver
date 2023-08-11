"""Styles for interface elements"""


SIDEBAR_STYLE: dict = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20%",
    #"padding": "1% 1%",
    #"background-color": "#f8f9fa",
  #  'display': 'inline-block',
}
# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE: dict = {
    "margin-left": "22%",
    "margin-right": "2%",
    "padding": "1% 1%",
    'width': '60%'
  #  "display": "inline-block",
    #'overflow': 'scroll'
}
UPLOAD_A_STYLE: dict = {
    'color': '#1EAEDB',
    'cursor': 'pointer',
    'text-decoration': 'underline',
}
UPLOAD_STYLE: dict = {
    'width': '100%',
 #   'display': 'inline-block', 
    'height': '60px',
    'lineHeight': '20px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '2px',
    'textAlign': 'center',
    'alignContent': 'center',
    'margin': 'auto',
    #'float': 'right'
}
UPLOAD_BUTTON_STYLE: dict = {
    'display': 'inline-block',
    'margin': '2%',
    'width':'96%',
}
SIDEBAR_LIST_STYLES: dict = {
    1: {
        'font-size': '1.2rem',
        'padding-left': '2%'
    },
    2: {
        'font-size': '1.1rem',
        'padding-left': '2%'
    },
    3: {
        'font-size': '1.0rem',
        'padding-left': '2%'
    },
    4: {
        'font-size': '0.9rem',
        'padding-left': '2%'
    },
    5: {
        'font-size': '0.8rem',
        'padding-left': '2%'
    },
    6: {
        'font-size': '0.7rem',
        'padding-left': '2%'
    },
}


UPLOAD_INDICATOR_STYLE: dict = {
                    'background-color': 'gray',
                    'opacity': '50%',
                    'width': '23%',
                    'height': '62px',
                    'border':'2px black solid',
                    'float': 'right',
                    'borderRadius': '15px',
                    'display': 'inline-block'
                }