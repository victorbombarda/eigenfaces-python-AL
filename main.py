import glob
import cv2
import numpy as np


def load_images(directory):
	# get a list of all the picture filenames
	jpgs = glob.glob(directory + '/*.jpeg')
	# load a greyscale version of each image
	imgs = np.array([cv2.imread(i, 1).flatten() for i in jpgs])
	return imgs

#Carregamos as fotos em uma variável
images = load_images(r"faces_jpeg")

#Criando o vetor média
mean = np.zeros(images[0].shape[0])
for i in images:
    mean = mean + i
mean = mean * 1/100

#Diminuímos cada coluna da matriz do vetor média
images_transf = images
for i in range(len(images)):
    images_transf[i] = images[i] - mean

