"""Dash app for inspection and analysis of MS performance based on TIC graphs"""

import os
from pathlib import Path
import dash
from dash import dcc, html
from element_styles import GENERIC_PAGE
import logging

logger = logging.getLogger(__name__)
dash.register_page(__name__, path=f'/user_guide')
logger.info(f'{__name__} loading')

def announcements():
    """Render announcements markdown page.

    :returns: Div containing announcements markdown content.
    """
    announcements:str = ''
    umstyle:dict = GENERIC_PAGE.copy()
    umstyle['max-width']='800px'
    umstyle['paddingTop'] = 80
    root_dir = Path(__file__).resolve().parents[1]
    doc_path = os.path.join(root_dir, 'data','announcements.md')
    with open(doc_path) as fil:
        announcements = fil.read()
    return html.Div([
        dcc.Markdown(
            announcements,
            style={'paddingRight': '4%', 'paddingLeft': '2%'},
            className='md-table')
            
    ], style=umstyle)

def other_tools():
    """Render other tools markdown page.

    :returns: Div containing other tools markdown content.
    """
    manual_contents:str = ''
    umstyle:dict = GENERIC_PAGE.copy()
    umstyle['max-width']='800px'
    umstyle['paddingTop'] = 80
    root_dir = Path(__file__).resolve().parents[1]
    doc_path = os.path.join(root_dir, 'data','other_tools.md')
    with open(doc_path) as fil:
        manual_contents = fil.read()
    return html.Div([
        dcc.Markdown(
            manual_contents,
            style={'paddingRight': '4%', 'paddingLeft': '2%'},
            className='md-table')
    ], style=umstyle)


def user_manual():
    """Render the user manual markdown page.

    :returns: Div containing user guide markdown content.
    """
    manual_contents:str = ''
    umstyle:dict = GENERIC_PAGE.copy()
    umstyle['max-width']='800px'
    umstyle['paddingTop'] = 80
    root_dir = Path(__file__).resolve().parents[1]
    doc_path = os.path.join(root_dir, 'data','user_guide.md')
    with open(doc_path) as fil:
        manual_contents = fil.read()
    return html.Div([
        dcc.Markdown(
            manual_contents,
            style={'paddingRight': '4%', 'paddingLeft': '2%'},
            className='md-table')
    ], style=umstyle)

layout = [user_manual(), other_tools()]