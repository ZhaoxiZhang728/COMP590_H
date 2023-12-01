from gaussian_blur_kernel_2d import gaussian_blur_kernel_2d

from convolve_2d import convolve_2d


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    gaussian_kernel = gaussian_blur_kernel_2d(sigma = sigma,height=size[0],width=size[1])
    result = convolve_2d(img,gaussian_kernel)

    return result
