class Styles:


    @property
    def upload_style(self) -> dict:
        return self._upload_style
    @property
    def upload_a_style(self) -> dict:
        return self._upload_a_style

    def __init__(self) -> None:
        self._upload_a_style = {
            'color': '#1EAEDB',
            'cursor': 'pointer',
            'text-decoration': 'underline'
        }
        self._upload_style = {
                            'width': '40%',
                            'height': '60px',
                            #'lineHeight': '20px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'alignContent': 'center',
                            'margin': 'auto',
                            #'float': 'right'
        }