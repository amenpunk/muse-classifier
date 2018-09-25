import numpy as np
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Reshape,GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import optimizers
from os import listdir
import cv2
import json 

import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, confusion_matrix

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    # https://gist.github.com/zachguo/10296432
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


categories = ['eri', 'hanayo', 'honoka', 'kotori', 'maki', 'nico', 'nozomi', 'rin', 'umi']
with open("categories.txt", "w") as f:
    f.write(json.dumps(categories))

try:
    X, y = joblib.load('features.pkl')
except:
    size = 197
    model = applications.ResNet50(input_shape=(size, size, 3), weights='imagenet', include_top=False)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    model_final = Model(input=model.input, output=x)
    model_final.summary()
    model_final.save('model.h5')

    def features(img):
        img = cv2.resize(img, (size, size))
        img = np.reshape(img, (1, size, size, 3))
        f = model_final.predict(img)[0]
        #print(f)
        return f

    X, y = [], []
    for i, cat in enumerate(categories):
        for fn in listdir('dataset/'+cat):
            path = 'dataset/%s/%s' % (cat, fn)
            print(path)
            img = cv2.imread(path)
            X.append(np.squeeze(features(img)))
            y.append(i)

n_img = 0
for cat in categories:
    n_img += len(listdir('dataset/'+cat))

test_size = 0.05
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=0)

scaler = sklearn.preprocessing.StandardScaler()
pca = PCA(whiten=True, random_state=0)
pipeline=Pipeline(steps=[('scaler', scaler), ('pca', pca), ('svc', SVC(probability=True))])

#{'pca__n_components': 384, 'svc__C': 100, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}
#{'pca__n_components': 384, 'svc__C': 1, 'svc__kernel': 'linear'}
C=[1, 10, 100]
n_components = [128, 256, 384, 480, 512]
#n_components = [320, 384, 416, 448, 480]
params = [
    {
        'pca__n_components': n_components,
        'svc__C': C,
        'svc__kernel': ['linear']
    },
    {
        'pca__n_components': n_components,
        'svc__C': C,
        'svc__gamma': [0.001, 0.0001],
        'svc__kernel': ['rbf']
    },
]
clf = GridSearchCV(pipeline, params,
                   n_jobs=11, verbose=3)
clf.fit(X_train, y_train)

# View the best parameters for the model found using grid search
print('Params:',clf.best_params_)

classifier = clf.best_estimator_

y_pred = classifier.predict(X_test)
train_score, validation_score = classifier.score(X_train, y_train), accuracy_score(y_test,y_pred)
print("Score: %.2f%%" % (train_score*100))
print("Accuracy: %.2f%%" % (validation_score*100))
print("Total accuracy: %.2f%%" % (((train_score*n_img*(1-test_size)+validation_score*n_img*test_size)/n_img)*100))
print_cm(confusion_matrix(y_test, y_pred), labels=categories)

joblib.dump((X, y), 'features.pkl')
joblib.dump(classifier, 'classifier.pkl')
