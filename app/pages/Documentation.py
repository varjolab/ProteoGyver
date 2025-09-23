"""Dash app for inspection and analysis of MS performance based on TIC graphs"""

import os
import dash
from dash import dcc, html
from element_styles import GENERIC_PAGE
import logging

logger = logging.getLogger(__name__)
dash.register_page(__name__, path=f'/user_guide')
logger.info(f'{__name__} loading')

def announcements():
    announcements:str = ''
    umstyle:dict = GENERIC_PAGE.copy()
    umstyle['max-width']='800px'
    umstyle['paddingTop'] = 80
    with open(os.path.join('data','announcements.md')) as fil:
        announcements = fil.read()
    return html.Div([
        dcc.Markdown(
            announcements,
            style={'paddingRight': '4%', 'paddingLeft': '2%'},
            className='md-table')
            
    ], style=umstyle)

def other_tools():
    manual_contents:str = ''
    umstyle:dict = GENERIC_PAGE.copy()
    umstyle['max-width']='800px'
    umstyle['paddingTop'] = 80
    with open(os.path.join('data','other_tools.md')) as fil:
        manual_contents = fil.read()
    return html.Div([
        dcc.Markdown(
            manual_contents,
            style={'paddingRight': '4%', 'paddingLeft': '2%'},
            className='md-table')
    ], style=umstyle)


def user_manual():
    manual_contents:str = ''
    umstyle:dict = GENERIC_PAGE.copy()
    umstyle['max-width']='800px'
    umstyle['paddingTop'] = 80
    with open(os.path.join('data','user_guide.md')) as fil:
        manual_contents = fil.read()
    return html.Div([
        dcc.Markdown(
            manual_contents,
            style={'paddingRight': '4%', 'paddingLeft': '2%'},
            className='md-table')
    ], style=umstyle)

layout = [user_manual(), other_tools()]