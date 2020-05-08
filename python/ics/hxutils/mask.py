from importlib import reload

import numpy as np
import fitsio

def load(path):
    return fitsio.read(path).astype('bool')
