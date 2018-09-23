import cv2
import sys
import os.path
from os import listdir, mkdir

cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
for name in ['2']:
    try:os.mkdir('dataset/'+name)
    except FileExistsError: pass
    for i, fn in enumerate(listdir('images-'+name)):
        print('images-'+name+'/'+fn)
        img = cv2.imread('images-'+name+'/'+fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (24, 24))
        for (j, (x, y, w, h)) in enumerate(faces):
            face = img[y:y+h,x:x+w]
            face = cv2.resize(face, (128, 128))
            print('dataset/%s/%d-%d.jpg' % (name, i, j))
            cv2.imwrite('dataset/%s/%d-%d.jpg' % (name, i, j), face)
