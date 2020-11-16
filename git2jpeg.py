import glob
import cv2 
import re


def gifs2jpeg(directory):
    files = glob.glob(directory + '/*.gif')
    for i in files:
        gif = cv2.VideoCapture(i)
        ret, frame = gif.read()
        image = i[i.find('/'): i.find('.')]
        cv2.imwrite(f"jpeg{image}.jpeg", frame)
        
   
    

	