import numpy as np

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    xx,yy = np.meshgrid(np.linspace(start=-(width//2),stop=width//2,num=width),
                        np.linspace(start=-(height//2),stop=(height//2),num=height))

    ex = np.exp(-((xx**2) + (yy**2)) / (2*(sigma**2)))

    g_filter = (ex / np.sum(ex))

    g_filter = g_filter.astype(np.float64)

    return g_filter