"""Module for creating embedded page files from a list of websites.

This module reads a text file containing website names and URLs, then generates
Dash pages that embed these websites using html.Embed.

Limitations:
- Not all sites can be embedded due to content security policies (CSP)
- HTTPS support is untested but may help with embedding restrictions
- Successfully tested only with:
  - Sites served from the same server
  - www.proteomics.fi
- All testing has been done without HTTPS
"""

import os
from typing import Tuple, List

def parse_embed_file(filename: str) -> List[Tuple[str, str]]:
    """Parse the embed configuration file containing website names and URLs.

    :param filename: Path to the text file containing site information.
    :returns: List of (site_name, url) tuples.
    """
    sites = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            if line.strip() and not line.startswith('#'):
                name, url = line.strip().split('\t')
                sites.append((name.strip(), url.strip()))
    return sites

def create_page_file(output_dir: str, site_name: str, url: str) -> None:
    """Create a new Dash page file for embedding a website.

    :param output_dir: Directory where the page file should be created.
    :param site_name: Name of the website (used for the page title and route).
    :param url: URL of the website to embed.
    :returns: None.
    """
    # Create sanitized filename from site name
    filename = f"{site_name.lower().replace(' ', '_').replace('.', '_')}.py"
    filepath = os.path.join(output_dir, filename)
    print('Creating page file:', filepath)
    # Page template
    page_content = f'''"""Embedded page for {site_name}."""

import dash
from dash import html

dash.register_page(__name__, path='/{site_name.lower().replace(" ", "-").replace(".", "-")}')

layout = html.Div([
    html.Embed(
        src="{url}",
        style={{
            'position': 'fixed',
            'top': '85px',  # Matches navbar height
            'left': '0',
            'bottom': '0',
            'right': '0',
            'width': '100%',
            'height': 'calc(100% - 85px)',  # Subtract navbar height
            'border': 'none',
            'margin': '0',
            'padding': '0',
            'overflow': 'hidden',
            'z-index': '999999'
        }}
    )
])
'''
    
    with open(filepath, 'w') as f:
        f.write(page_content)

def update_pages(output_dir: str, embed_file: str) -> None:
    """Update embedded pages based on the configuration file.

    :param output_dir: Directory where page files should be created.
    :param embed_file: Path to the text file containing site information.
    :returns: None.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Parse embed file and create pages
    sites = parse_embed_file(embed_file)
    for site_name, url in sites:
        create_page_file(output_dir, site_name, url) 

if __name__ == "__main__":
    update_pages(os.path.join(os.path.dirname(__file__), 'pages'), 'embed_pages.tsv')