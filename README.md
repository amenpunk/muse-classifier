# muse-classifier

Classifies characters from the idol group *muse* from the franchise *Love Live!*.

Unlike methods from [here](https://github.com/freedomofkeima/transfer-learning-anime) and [here](http://christina.hatenablog.com/entry/2015/01/23/212541), this classifier extracts features from pretrained models and classifies characters based on those features using an SVM. This leads to significantly faster training time (~1m39s on Quadro K2000+Xeon E5-2640) at the cost of accuracy.

![preview](preview.jpeg "Preview")

## Dataset

For each of the 9 characters in *muse*, I collected give-or-take 77 images with their respective tags from yande.re and konachan.com. Faces were extracted from the images using lbp-cascade and hand-filtered so as to remove noise, other characters and non-conforming art styles (ie. chibi versions). They were then all scaled to 128x128px.

The dataset is comprised of 684 files, 10% of them is used for validation.

## Pipeline

For the model, deep, high-level features is extracted from each image of every character. Then, the features are normalized using a stddev scaler. Using PCA, the feature space of X dimensions is reduced to Y components. The post-reduced features are finally classified using a support vector machine (SVM).

Here's a comparison of models used to extract deep features.

|  Model                      | X  | Y |  Training accuracy |  Validation accuracy |
|-----------------------------|----|---|--------------------|----------------------|
| MobileNet (13th conv layer) |1024| - | 96.1%              | 71.0%                |
| MobileNet (12th conv layer) |512 | - | 89.6%              | 73.9%                |
| ResNet50                    |2048|452| 98.9%              | 88.4%                |
| Xception                    |2048|512| 98.7%              | 43.5%                |
| InceptionV3                 |2048|512| 99.1%              | 51.8%                |

ResNet50 and Y=452 is used for the model. Y was finely tuned from a baseline of 512.

The SVM used is `SVM(C=10, gamma=0.0001, kernel="rbf")`
