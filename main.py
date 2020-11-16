import glob
import cv2
import numpy as np


def load_images(directory):
	# get a list of all the picture filenames
	jpgs = glob.glob(directory + '/*.jpeg')
	# load a greyscale version of each image
	imgs = np.array([cv2.imread(i, 1).flatten() for i in jpgs])
	return imgs