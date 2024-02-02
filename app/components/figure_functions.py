import pandas as pd
from scipy.ndimage import gaussian_filter1d
import numpy as np

def improve_text_position(data_frame: pd.DataFrame) -> list:
    """Returns a list of text positions in alternating pattern."""
    
    positions: list = ['top left','top right','top center','middle left','middle right','middle center','bottom left','bottom right','bottom center']
    return [positions[i % len(positions)] for i in range(data_frame.shape[0])]
