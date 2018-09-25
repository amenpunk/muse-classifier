import sys
import cv2
from keras.models import load_model
import numpy as np
from os import listdir
from sklearn.externals import joblib
import json
from subprocess import check_output

model_final = load_model("model.h5")
classifier = joblib.load('classifier.pkl')
categories = json.load(open("categories.txt","r"))
size = 197
def features(img):
    img = cv2.resize(img, (size, size))
    img = np.reshape(img, (1, size, size, 3))
    f = np.reshape(model_final.predict(img)[0], (1, -1))
    return f

img = cv2.imread(sys.argv[1])
faces = json.loads(check_output(['ruby', 'detect.rb', sys.argv[1]]))
scale = 1.4
for face in faces:
    print(face['likelihood'])

    x, y, w, h = face['face']['x'], face['face']['y'], face['face']['width'], face['face']['height']
    x -= w*(scale-1)/2
    w *= scale
    y -= h*(scale-1)/2
    h *= scale
    x, y, w, h = int(max(x,0)), int(max(y,0)), int(w), int(h)
    
    fimg = img[y:y+h,x:x+w]
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    f = features(fimg)
    n = classifier.predict_proba(f)[0]
    label = np.argmax(n)
    name=categories[label]
    cv2.putText(img,'%s (%.2f)'%(name, n[label]*100),(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()
