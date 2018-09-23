import sys
import cv2
from keras.models import load_model
import numpy as np
from os import listdir
from sklearn.externals import joblib
from keras import applications
import json


fw = fh = 128

model_final = load_model("model.h5")

(scaler, pca, classifier) = joblib.load('classifier.pkl')
#classifier = joblib.load('classifier.pkl')

def features(img):
    img = cv2.resize(img, (197, 197))
    img = np.reshape(img, (1, 197, 197, 3))
    f = np.reshape(model_final.predict(img)[0], (1, -1))
    f = scaler.transform(f)
    f = pca.transform(f)
    #print(f)
    return f

cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')

categories = json.load(open("categories.txt","r"))
img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
    
faces = cascade.detectMultiScale(gray,
            # detector options
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (24, 24))

scale = 1.2
for (x, y, w, h) in faces:
    x -= w*(scale-1)/2
    w *= scale
    y -= h*(scale-1)/2
    h *= scale
    x, y, w, h = int(max(x,0)), int(max(y,0)), int(w), int(h)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    face = img[y:y+h, x:x+w]

    f = features(face)
    n = classifier.predict_proba(f)[0]
    label = np.argmax(n)
    name=categories[label]
    cv2.putText(img,'%s (%.2f)'%(name, n[label]*100),(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)

cv2.imwrite(sys.argv[1].split('/')[-1], img)
cv2.imshow('img', img)
cv2.waitKey()
