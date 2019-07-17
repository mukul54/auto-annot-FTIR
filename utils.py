import skimage
import matplotlib.pyplot as plt
import numpy as np

def blend(img1, img2, alpha=0.5):
    
    """
    Alpha blend two images.

    Parameters
    ----------
    img1, img2 : numpy.ndarray
                Images to blend.
    alpha : float
            Blending factor.

    Returns
    -------
    result : numpy.ndarray
            Blended image
    """
    img1 = skimage.img_as_float(img1)
    img2 = skimage.img_as_float(img2)
    return img1*alpha+(1-alpha)*img2


def imshowpair_python(img1, img2, output_size = (16,16)):
    """ 
    Python version of matlab's imshowpair 
    Shows images as an overlapping pair with image 1 in red and image 2 in green
    
    Parameters
    ----------
    img1, img2 : numpy.ndarray
                Images to blend.
    img1: first image to display. Will be shown in red.

    img2: second image to display. Will be shown in green.

    ax: axis to plot on. If none, current.

    return:
    --------
    None
    """
    if ax is None:
        ax = plt.gca()

    # Create zero image
    zero_pad = np.zeros(shape=img_1.shape)
    # Normalize images
    img1 = norm_img(img1)
    img2 = norm_img(img2)
    # Ensure images have same mean
    img2 *= (img1.mean()/img2.mean())
    # Concatenate image
    show_img = np.stack((img1, img2, zero_pad), axis=2)
    # Plot
    ax.imshow(show_img)

def norm_img(img):

    """
    Normalizes an image to between 0 and 1
    
    Parameters
    ----------
    img: ndarray
    image to normalize

    return 
    --------- 
    nomralized image
    """
    return (img - img.min())/(img.max() - img.min())

