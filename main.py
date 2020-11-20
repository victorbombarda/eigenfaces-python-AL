import glob
import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from PIL import Image


def load_images(directory):
	# get a list of all the picture filenames
	jpgs = glob.glob(directory + '/*.jpeg')
	# load a greyscale version of each image
	imgs = np.array([cv2.imread(i, 0).flatten() for i in jpgs])
	return imgs

#Carregamos as fotos em uma variável
images = load_images(r"faces_jpeg")
images_teste = load_images(r"faces_jpeg")

#Criando o vetor média
mean = np.zeros(images[0].shape[0])
for i in images:
    mean = mean + i
mean = mean * 1/100
#Transformamos do float64 para  grayscale
mean = mean.astype(np.uint8)

#Visualizamos a imagem média
mean_img = mean.reshape(983,800)
Image.fromarray(mean_img)

#Diminuímos cada coluna da matriz do vetor média
for i in range(len(images)):
    images_teste[i] = images[i] - mean

#Conseguimos visualizar as imagens normalizadas
teste_img = images_teste[0].reshape(983,800)
Image.fromarray(teste_img)

#Cria matriz quadrada 100x100
quadrada = np.matrix(images_teste) * np.matrix(images_teste.transpose())
quadrada = quadrada * 1/100	

#Calculando os autovalores e autovetores 
autovalores, autovetores = np.linalg.eig(quadrada)
autovalores = autovalores[autovalores.argsort()[::-1]]
autovetores = autovetores[:,autovalores.argsort()[::-1]]


