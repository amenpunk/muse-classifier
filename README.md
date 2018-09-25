# muse-classifier

Classifies characters from the idol group *muse* from the franchise *Love Live!*.

Unlike methods from [here](https://github.com/freedomofkeima/transfer-learning-anime) and [here](http://christina.hatenablog.com/entry/2015/01/23/212541), this classifier extracts features from pretrained models and classifies characters based on those features using an SVM. This leads to significantly faster training time (~1m39s on Quadro K2000+Xeon E5-2640) at the cost of accuracy.

![preview](preview.jpeg "Preview")

## Dataset

For each of the 9 characters in *muse*, I collected 77 images with their respective tags from yande.re and konachan.com. Faces were extracted from the images using [lbp-cascade](https://github.com/nagadomi/lbpcascade_animeface)/[anime-face2009](https://github.com/nagadomi/anime-face2009) and hand-filtered so as to remove noise, other characters and non-conforming art styles (ie. chibi versions). They were then all scaled to 128x128px ( **note** : some are in 96x96px resolution ).

The dataset is comprised of 693 files, 10% of them is used for validation.

## Pipeline

For the model, deep, high-level features is extracted from each image of every character. Then, the features are normalized using a stddev scaler. Using PCA, the feature space of X dimensions is reduced to Y components. The post-reduced features are finally classified using a support vector machine (SVM).

ResNet50 is used for the model. The pipeline used has parameters of `{'pca__n_components': 384, 'svc__C': 100, 'svc__gamma': 0.0001, 'svc__kernel': 'rbf'}`

With the current dataset it can classify 98.39% of the training set correctly and 84.29% of the validation set correctly (total of 96.98%).

If 5% of the dataset as validation is used, it can classify 98.33% of the training set and 77.14% of the validation set correctly (total of 97.27%).

Confusion matrix for 5% validation:

```
              eri hanayo honoka kotori   maki   nico nozomi    rin    umi
       eri    4.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
    hanayo    0.0    2.0    0.0    0.0    0.0    0.0    0.0    1.0    0.0
    honoka    0.0    0.0    1.0    0.0    0.0    0.0    0.0    0.0    0.0
    kotori    0.0    0.0    0.0    1.0    0.0    0.0    0.0    1.0    0.0
      maki    0.0    0.0    0.0    0.0    4.0    0.0    0.0    0.0    0.0
      nico    0.0    0.0    0.0    0.0    0.0    7.0    0.0    0.0    0.0
    nozomi    1.0    0.0    0.0    1.0    0.0    1.0    2.0    0.0    0.0
       rin    1.0    0.0    2.0    0.0    0.0    0.0    0.0    2.0    0.0
       umi    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    4.0
```

## Notes

* It can't classify faces with closed eyes very well. This is probably due to the dataset used.
* It can't classify faces with different lighting conditions well. Might be good to normalize/scale image or use data augmentation.
