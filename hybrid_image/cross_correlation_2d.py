import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    index_h = k_h // 2
    index_w = k_w // 2


    padding_img = np.pad(array=img,pad_width = ((index_h,index_h),(index_w,index_w),(0,0)),mode = 'constant')

    h,w,c = padding_img.shape

    output_img = np.zeros((h-k_h+1,w-k_w+1,c))
    for channel in range(c):
        for width in range(index_w,w-index_w):
            for height in range(index_h,h-index_h):

                F = padding_img[height-index_h:height+index_h+1,width-index_w:width+index_w+1,channel]


                output_img[height-index_h,width-index_w,channel] = np.sum(F * kernel)

    return output_img