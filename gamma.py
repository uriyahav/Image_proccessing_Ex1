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
import cv2
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE , LOAD_RGB


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    def trackAndDisplay(i):
        img = cv2.imread(img_path)
        if rep == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gamma = cv2.getTrackbarPos(trackbar, title)
        #maybe/100 for brighter
        gamma = float(gamma) / 50
        # Calculate new picture & cast to uint8 to represent the pic
        img = np.uint8(255*np.power((img.copy() / 255), gamma))
        cv2.imshow(title, img)
        pass
    title = 'Gamma Correction'
    trackbar = 'Gamma :'
    cv2.namedWindow(title)
    cv2.createTrackbar(trackbar, title, 0 , 200, trackAndDisplay )
    trackAndDisplay(0)
    #wait untill change
    cv2.waitKey()

def main():
    gammaDisplay('beach.jpg', LOAD_GRAY_SCALE)
    gammaDisplay('beach.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
