import numpy as np
from cross_correlation_2d import cross_correlation_2d

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    rotated_filter_ = np.rot90(np.rot90(kernel))

    result = cross_correlation_2d(img,rotated_filter_)

    return result