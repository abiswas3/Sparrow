HEADER = '\033[95m'
BLUE = '\033[94m'
GREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

import pickle
import scipy.misc
import cv2

def unpickle(fileName):

    with open(fileName, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def write_image_to_file(image, image_path):    
    return scipy.misc.imsave(image_path, image)

def read_image_from_file(fName):

    return cv2.imread(fName)
