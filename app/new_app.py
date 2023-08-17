
from dash import Dash, html, page_registry, page_container  # dcc,DiskcacheManager, CeleryManager
from components.ui_components import navbar
from dash_bootstrap_components.themes import FLATLY


app = Dash(external_stylesheets=[FLATLY],suppress_callback_exceptions=True)
app: Dash = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        FLATLY
    ],
    suppress_callback_exceptions=True
)

app.title = 'Data analysis alpha version'
app.enable_dev_tools(debug=True)
#app.config.from_pyfile('app_config.py')
server = app.server

print('Initialize app.')
print('Site pages:')
navbar_pages: list = []
for page in page_registry.values():
    if 'TIC' in page['name']:
        continue
    navbar_pages.append((page['name'], page['relative_path']))
    print(page['name'])
navbar_pages.append(('JupyterHub','pg-23.biocenter.helsinki.fi:8090/'))

app.layout = html.Div([
    navbar(navbar_pages),
    page_container,
])
print('End app.')
if __name__ == '__main__':
    app.run(debug=True)
# <iframe src="https://www.chat.openai.com/chat" title="ChatGPT Embed"></iframe>
