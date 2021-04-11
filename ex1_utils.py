"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
import cv2
import numpy as np

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 111111111


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    image = cv2.imread(filename)
    if representation == LOAD_RGB:
        imageColor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        imageColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # make normalization
    image = imageColor
    image = (image - image.min()) / (image.max() - image.min())
    return image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    image = imReadAndConvert(filename, representation)
    if representation == LOAD_GRAY_SCALE:
        plt.gray()
    plt.imshow(image)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    yiq_transform = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.275, -0.321],
                              [0.212, -0.523, 0.311]])
    return np.dot(imgRGB, yiq_transform.T.copy())


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_transform = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.275, -0.321],
                              [0.212, -0.523, 0.311]])
    return np.tensordot(imgYIQ, np.linalg.inv(yiq_transform).copy(), axes=(2, 1))


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    is_rgb = False
    # RGB image procedure should only operate on the Y chanel
    if len(imgOrig.shape) == 3:
        is_rgb = True
        yiq_image = transformRGB2YIQ(np.copy(imgOrig))
        imgOrig = yiq_image[:, :, 0]
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')
    histOrg = np.histogram(imgOrig.flatten(), 256)[0]
    cumsum = np.cumsum(histOrg)
    imEq = cumsum[imgOrig]
    imEq = cv2.normalize(imEq, None, 0, 255, cv2.NORM_MINMAX)
    histEQ = np.histogram(imEq.flatten(), 256)[0]
    if is_rgb:
        # get 3 channels
        yiq_image[:, :, 0] = imEq / (imEq.max() - imEq.min())
        imEq = transformYIQ2RGB(np.copy(yiq_image))
    return imEq, histOrg, histEQ


def finding_z(z, q: np.array):
    for i in range(1, len(q)):
        z[i] = (q[i - 1] + q[i]) / 2
    return z

def finding_q(q, z, hist):
    for i in range(len(q)):
        # cast z to closest integer
        start, end = np.rint(z[i]).astype(np.int64), np.rint(z[i + 1] + 1).astype(np.int64)
        # use weighted mean
        g = np.arange(start, end)
        h = hist[start:end]
        q[i] = np.sum(g * h) / np.sum(h)
    return q


# Function to aplly & return quantized image.
def quantization(z, q, image):
    img = np.copy(image)
    for i in range(len(q)):
        low = image >= (z[i] / 255)
        if i == len(q) - 1:
            high = image <= z[i + 1] / 255
        else:
            high = image < z[i + 1] / 255
        # replacing pixels to quantized value:
        img[(np.logical_and(low, high))] = q[i] / 255
    return img




def init_z(cumsum, nQuant):
    '''
     function to initialize z by initial division such that each segment will contain approximately
     the same number of pixels., q-means of each k
    based on cum sum  of image histogram'''
    z = np.zeros((nQuant + 1,))
    z[nQuant] = 255
    total_sum = cumsum[len(cumsum) - 1]
    i = j = 1
    while (j < nQuant):
        # iterate over the list slices until we hit the "middle"
        while cumsum[i] < j * (total_sum / nQuant):
            # make sure  we won't go further
            if cumsum[i + 1] >= j * (total_sum / nQuant):
                break
            i += 1
        z[j] = i
        j += 1
    return z


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    error_list = []
    quant_image_list = []
    image = np.copy(imOrig)
    is_rgb = False
    if len(imOrig.shape) == 3:
        is_rgb = True
        yiq_image = transformRGB2YIQ(imOrig)
        image = yiq_image[:, :, 0].copy()
    histOrg = np.histogram(image.flatten(), 256)[0]
    cumsum = np.cumsum(histOrg)
    # divide based on histogram cumsum
    z = init_z(cumsum, nQuant)
    q = []
    # init q based on z
    for i in range(1, nQuant):
        q = (z[1:] + z[:-1]) / 2

    for it in range(nIter):
        z_prev = np.copy(z)
        z = finding_z(z, q)
        q = finding_q(q, z, histOrg)
        pic = quantization(z, q, image)
        # compute error
        error = np.sqrt(np.sum((pic - image) ** 2)) / image.size
        if is_rgb:
            yiq_image[:, :, 0] = pic
            pic = transformYIQ2RGB(np.copy(yiq_image))
        error_list.insert(it, error)
        # normalization
        pic = (pic - pic.min()) / (pic.max() - pic.min())
        quant_image_list.insert(it, pic)
        # check if already converged
        if np.array_equal(z, z_prev):
            break

    return quant_image_list, error_list
