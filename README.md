# Kaggle-DigitRecognizer-CNN

## Overview
This is a kaggle competition project coding in `Python`. In this project, the goal is to correctly identify the handwritten single digit by training a dataset of tens of thousands of handwritten images. More information about the competition can be found [here](https://www.kaggle.com/c/digit-recognizer#description).

## Details
This program utilizes a Convolutional Neural Network([LeNet](http://yann.lecun.com/exdb/lenet/)) to achieve the goal. In this CNN, hiden layers contain two convolution layers, two max pooling layers and two fully connected layers.

![LeNet](https://raw.githubusercontent.com/Siboooo/imgForMD/master/DigitRecognizer/LeNet.jpg)
<p style="text-align: center;">Figure 1: LeNet Structure from [ujjwalkarn.me](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)</p>


Training dataset and test dataset can be downloaded [here](https://www.kaggle.com/c/digit-recognizer/data).

## Performance
Two versions were submitted (Score up to 98.8%).

![submission](https://raw.githubusercontent.com/Siboooo/imgForMD/master/DigitRecognizer/DR-CNN-sub.png)

## Dependencies
* [NumPy](http://www.numpy.org)
* [Pandas](http://pandas.pydata.org)
* [TensorFlow](https://www.tensorflow.org)