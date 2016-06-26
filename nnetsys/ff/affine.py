"""
Module with affine changes
"""
from random import randint, choice
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

def render_2d(data):
    """Render a RGB 2D image"""
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    plt.imshow(data)
    plt.savefig('image.png')

def shift(data, d3):
    if d3:
        random_vector = (0, randint(-3, 3), randint(-3, 3))
    else:
        random_vector = (randint(-3, 3), randint(-3, 3))

    return scipy.ndimage.shift(data, 
                               random_vector,
                               mode='reflect')

def flip(data, d3):
    if d3:
        return np.swapaxes(np.fliplr(np.swapaxes(data,1,2)),1,2)
    else:
        return np.fliplr(data)

def random_change(data):
    """
    Perform a random, dimension and arity preserving transformation
    """
    
    d3 = len(data.shape) == 3   # is this a RGB image?
    
    ops = choice([shift, flip])
    
    a = ops(data, d3)
    return a
