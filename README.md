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
