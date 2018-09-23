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
from sklearn import grid_search
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA

categories = ['eri', 'hanayo', 'honoka', 'kotori', 'maki', 'nico', 'nozomi', 'rin', 'umi']
with open("categories.txt", "w") as f:
    f.write(json.dumps(categories))

# mobilenet+linearsvm without pca,scaling
# globalaveragePooling2D
# -13 *: 96.1%/71.0%
# -14:   98.2%/68.6%*
# -15:   96.9%/74.3%*
# -16 *: 89.6%/73.9%
# -17:   96.8%/70.0%*
# -20 *: 98.4%/74.3%*
# -26 *: 98.4%/70.0%*
# C=1e5: 97.4%/75.7%*
# C=1e6: 97.1%/75.7%*
# C=5e6: 97.9%/78.6%*
# C=6e6: 97.6%/77.1%*

# pca @ 512
# resnet50: 98.9%/87.0%
# resnet50 (@452): 98.9%/88.4%
# xception:  99.1%/56.2%
# inceptionV3: 99.1%/51.8%


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

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)

# Apply standard scaler to output from resnet50
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Take PCA to reduce feature space dimensionality
pca = PCA(n_components=452, svd_solver='full', whiten=True, random_state=0)
pca = pca.fit(X_train)
print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print("fitting...")

C=[1, 10, 100, 1000]
params = [
    {
        'C': C,
        'kernel': ['linear']
    },
    {
        'C': C,
        'gamma': [0.001, 0.0001],
        'kernel': ['rbf']
    },
]
clf = GridSearchCV(estimator=SVC(probability=True), param_grid=params, n_jobs=-1)
clf.fit(X_train, y_train)

# View the best parameters for the model found using grid search
print('Params:',clf.best_params_)

classifier = clf.best_estimator_

y_pred = classifier.predict(X_test)
print("Score: {0:0.1f}%".format(classifier.score(X_train, y_train)*100))
print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))

joblib.dump((X, y), 'features.pkl')
joblib.dump((scaler, pca, classifier), 'classifier.pkl')
#joblib.dump(classifier, 'classifier.pkl')
