from PIL import Image
import numpy as np
from scipy.ndimage import filters


class calcGradient(object):
    """Apply sobel operators in x and y direction on the given PIL.Image.
    """

    def __init__(self):
        pass

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: channel1: grayscale.
                       channel2: x gradient
                       channel3: y gradient
        """
        w, h = img.size
        ret_img = np.zeros((3, h, w))
        img = np.array(img.convert('LA'))

        ret_img[:, :, 0] = np.asarray(img)
        filters.sobel(img, 1, ret_img[:, :, 1])
        filters.sobel(img, 0, ret_img[:, :, 2])

        img = Image.fromarray(ret_img)

        return img

    def randomize_parameters(self):
        pass