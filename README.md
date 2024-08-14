---
title: Potato
emoji: 📈
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Here is the updated README with badges:

**Potato Disease Detection: PlantVillage Insights 🥔**
=====================================================

**Author:** Phuong Phan
**Date:** 14 July 2024
**Dataset:** Potato Plant Diseases

**Welcome! 🌱**
---------------

We're excited to share our project on potato disease detection using machine learning and computer vision. This project aims to revolutionize the field of precision agriculture by providing a accurate and efficient way to detect potato diseases.

**The Problem 🌪️**
------------------

Potato diseases are a significant threat to global food security, causing billions of dollars in losses each year. However, accurate detection of these diseases remains a challenge. This project aims to develop a machine learning model that can detect potato diseases using image data, enabling farmers and researchers to take targeted action to prevent the spread of disease.

**Methodology 🔍**
-----------------

Our approach involves the following steps:

1. **Data Augmentation 🔄**: We apply random transformations to the images to artificially increase the size of the training set.
2. **Evaluation 📊**: We performance test the accuracy and loss of four pre-trained models: RESNET, Inception, Xception & NASnetMobile on the test dataset.
3. **Fine-tuning 🔩**: We fine-tune the models by unfreezing the weights and training it on the augmented training data.
4. **Hyperparameter Tuning 🔧**: We define a new model architecture with a custom classification head on top of a pre-trained base model, and compile it with Adam optimizer and sparse categorical cross-entropy loss.

**Installing Required Libraries 📚**
-----------------------------------

To get started, you'll need to install the following libraries:

```bash
!pip install kaggle -q
!kaggle datasets download -d hafiznouman786/potato-plant-diseases-data
! unzip potato-plant-diseases-data.zip
```

**Dataset Content 📁**
---------------------

Our dataset contains 2152 images categorized into three classes: early blight, late blight, and healthy.

**Model Architectures 🤖**
-------------------------

We use four pre-trained models:

1. **Xception 🤩**: A deep learning model that uses a novel architecture to achieve state-of-the-art results on image classification tasks.
2. **ResNet50 📈**: A popular deep learning model that uses residual connections to ease the training process.
3. **Inception 🤔**: A deep learning model that uses multiple parallel branches to capture features at different scales.
4. **NASNetMobile 📱**: A deep learning model that uses neural architecture search to find the optimal architecture for mobile devices.

**Model Comparison 📊**
----------------------

We compare the performance of the four models on the test dataset.

**Results 📈**
--------------

Our results show that the Xception model performs best on the test dataset, achieving an accuracy of 97.36%.

**Predicting 🔮**
----------------

We use the Xception model to predict the class of new images.

**GIF 🎥**
----------

We create a GIF to visualize the predictions.

**Conclusion 🌟**
----------------

This project demonstrates the potential of machine learning in detecting potato diseases using image data. We hope that our work will inspire further research and development in precision agriculture.

**Badges 🏆**
-------------

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.4-orange.svg)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Potato%20Plant%20Diseases-green.svg)](https://www.kaggle.com/hafiznouman786/potato-plant-diseases-data)

**Future Work 🌱**
-----------------

There are many ways to improve this project, including:

* Collecting more data to increase the accuracy of the model
* Using transfer learning to adapt the model to other crops
* Developing a mobile app to enable farmers to use the model in the field

**Get Involved 🤝**
------------------

We're always looking for collaborators and contributors to help us improve this project. If you're interested in getting involved, please don't hesitate to reach out.

I added the following badges:

* License: MIT
* Python: 3.8
* TensorFlow: 2.4
* Keras: 2.4
* Dataset: Potato Plant Diseases

These badges provide a quick and easy way to see the key technologies and dependencies

Here's another version of the README:

**Potato Disease Detection: A Deep Learning Approach 🥔**
=====================================================

**Overview 🌱**
---------------

This project uses deep learning techniques to detect potato diseases from images. The goal is to develop a model that can accurately identify diseases such as early blight, late blight, and healthy plants.

**Motivation 🌪️**
-----------------

Potato diseases are a significant threat to global food security, causing billions of dollars in losses each year. Traditional methods of disease detection are time-consuming and require expertise. This project aims to develop a machine learning model that can detect potato diseases quickly and accurately, enabling farmers and researchers to take targeted action to prevent the spread of disease.

**Dataset 📁**
-------------

The dataset used in this project consists of 2152 images of potato plants, categorized into three classes: early blight, late blight, and healthy.

**Methodology 🔍**
-----------------

Our approach involves the following steps:

1. **Data Preprocessing 🔄**: We apply random transformations to the images to artificially increase the size of the training set.
2. **Model Selection 🤔**: We compare the performance of four pre-trained models: RESNET, Inception, Xception & NASnetMobile on the test dataset.
3. **Fine-tuning 🔩**: We fine-tune the models by unfreezing the weights and training it on the augmented training data.
4. **Hyperparameter Tuning 🔧**: We define a new model architecture with a custom classification head on top of a pre-trained base model, and compile it with Adam optimizer and sparse categorical cross-entropy loss.

**Results 📈**
--------------

Our results show that the Xception model performs best on the test dataset, achieving an accuracy of 97.


Here is a well-structured README for your repository:

**Potato Disease Detection: PlantVillage Insights**
=====================================================

**Author:** Phuong Phan
**Date:** 14.July.2024
**Dataset:** Potato Plant Diseases

**Inspiration**
---------------

The devastating impact of potato diseases on global food security, with losses estimated to be in the billions, has long been a pressing concern for agricultural researchers and farmers alike. However, accurate detection of these diseases remains a significant challenge, hindering our ability to develop effective management strategies and ensure a stable food supply. This project embarks on a journey to develop a machine learning model capable of detecting potato diseases using image data, leveraging the power of computer vision and deep learning to revolutionize the field of precision agriculture, as inspired by the innovative approaches of PlantVillage Insights.

**Problem Statement**
-------------------

The potato disease dataset is a comprehensive collection of images meticulously categorized into three distinct classes: early blight, late blight, and healthy. Each class represents a specific condition affecting potato crops, enabling researchers and agricultural experts to delve into the intricacies of disease identification, progression, and management.

**Dataset**
------------

The dataset consists of 2152 images belonging to 3 classes.

**Methodology**
--------------

1. **Data Augmentation**: Apply random transformations to the images to artificially increase the size of the training set using TF-keras pre-processing layers.
2. **Evaluation**: Performance test accuracy and loss of 4 Models: RESNET, Inception, Xception & NASnetMobile on the test dataset. Visualizing curves over epochs using matplotlib to identify potential overfitting or underfitting issues.
3. **Fine-tuning**: 4 Models model is specifically fine-tuned further by unfreezing the weights and training it on the augmented training data. With learning rate reduction and early stopping callbacks to prevent overfitting, and its test accuracy is evaluated.
4. **Hyperparameter Tuning**: The code defines a new model architecture with a custom classification head on top of a pre-trained base model, and compiles it with Adam optimizer and sparse categorical cross-entropy loss.
5. **Prediction**: Made on a random sample of images from the validation set. Incorrect predictions are also identified and visualized.

**Installing Required Libraries**
---------------------------------

```bash
!pip install kaggle -q
!kaggle datasets download -d hafiznouman786/potato-plant-diseases-data
!unzip plant-pathology-2021resized-512-x-512.zip
```

**Importing Libraries**
----------------------

```python
import numpy as np
import pandas as pd
import os
import sys
import json
import shutil
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, UpSampling2D, Dropout, Flatten, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications import MobileNetV3Large, EfficientNetV2B0, ResNet50V2, Xception, InceptionV3, MobileNetV3Small, NASNetMobile, InceptionResNetV2
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Javascript, Image, YouTubeVideo
```

**Model Architectures**
----------------------

1. **Xception**: Xception Modelling
2. **ResNet**: ResNet Modelling
3. **Inception**: Inception Modelling
4. **NASNetMobile**: NASNetMobile Modelling

**Model Comparison**
-------------------

| Model | Valid Dataset Accuracy | Test Dataset Accuracy |
| --- | --- | --- |
| Xception | 0.9736 | 0.97 |
| ResNet50 | 0.9950 | 0.99 |
| Inception | 0.9577 | 0.95 |
| NASNetMobile | 0.8999 | 0.92 |

**Predicting**
--------------

```python
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
```

**GIF**
------

I hope you found it informative and engaging. If you enjoyed the content and found it helpful. Your feedback is valuable and encourages me to create more content like this in the future.
