**Potato Disease Detection: A Deep Learning Approach 🥔**
=====================================================

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-green.svg)](https://opensource.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Potato%20Plant%20Diseases-green.svg)](https://www.kaggle.com/hafiznouman786/potato-plant-diseases-data)


A Machine Learning Tool to Prevent the Spread of Potato Diseases


## `Inspiration` 🌪️


> The devastating impact of potato diseases on global food security, with losses estimated to be in the billions, has long been a pressing concern for agricultural researchers and farmers alike.

> However, accurate detection of these diseases remains a significant challenge, hindering our ability to develop effective management strategies and ensure a stable food supply.


## `Solution ` 🌱

* Develop a machine learning model capable of detecting potato diseases using image data, leveraging the power of computer vision and deep learning to revolutionize the field of precision agriculture, as inspired by the innovative approaches of PlantVillage Insights.

* Using a combination of convolutional neural networks (CNNs) and transfer learning to achieve high accuracy.


## `Dataset` 📊

Potato Plant Diseases: is a comprehensive collection of images meticulously categorized into three distinct classes: early blight, late blight, and healthy. 

Each class represents a specific condition affecting potato crops, enabling researchers and agricultural experts to delve into the intricacies of disease identification, progression, and management.


## `Methodology` 🔍

1. **Data Augmentation**: We've applied random transformations to our images to artificially increase the size of our training dataset.
2. **Model Selection**: We've evaluated the performance of four different models: Xception, ResNet50, Inception, and NASNetMobile.
3. **Fine-tuning**: We've fine-tuned our models using learning rate reduction and early stopping callbacks to prevent overfitting.
4. **Hyperparameter Tuning**: We've optimized our model's hyperparameters using a grid search approach.

**Methodology**
--------------

1. **Data Augmentation**: Apply random transformations to the images to artificially increase the size of the training set using TF-keras pre-processing layers.
2. **Evaluation**: Performance test accuracy and loss of 4 Models: RESNET, Inception, Xception & NASnetMobile on the test dataset. Visualizing curves over epochs using matplotlib to identify potential overfitting or underfitting issues.
3. **Fine-tuning**: 4 Models model is specifically fine-tuned further by unfreezing the weights and training it on the augmented training data. With learning rate reduction and early stopping callbacks to prevent overfitting, and its test accuracy is evaluated.
4. **Hyperparameter Tuning**: The code defines a new model architecture with a custom classification head on top of a pre-trained base model, and compiles it with Adam optimizer and sparse categorical cross-entropy loss.
5. **Prediction**: Made on a random sample of images from the validation set. Incorrect predictions are also identified and visualized.

**Methodology 🔍**
-----------------

Our approach involves the following steps:

1. **Data Preprocessing 🔄**: We apply random transformations to the images to artificially increase the size of the training set.
2. **Model Selection 🤔**: We compare the performance of four pre-trained models: RESNET, Inception, Xception & NASnetMobile on the test dataset.
3. **Fine-tuning 🔩**: We fine-tune the models by unfreezing the weights and training it on the augmented training data.
4. **Hyperparameter Tuning 🔧**: We define a new model architecture with a custom classification head on top of a pre-trained base model, and compile it with Adam optimizer and sparse categorical cross-entropy loss.



## `Model Architectures` 🤖


1. **Xception**: A deep learning model that uses a novel architecture to achieve state-of-the-art results on image classification tasks.
2. **ResNet50**: A popular deep learning model that uses residual connections to ease the training process.
3. **Inception**: A deep learning model that uses multiple parallel branches to capture features at different scales.
4. **NASNetMobile**: A deep learning model that uses neural architecture search to find the optimal architecture for mobile devices.


> ### Model Performance 📊 

| Model | Valid Dataset Accuracy | Test Dataset Accuracy |
| --- | --- | --- |
| Xception | 0.9736 | 0.97 |
| ResNet50 | 0.9950 | 0.99 |
| Inception | 0.9577 | 0.95 |
| NASNetMobile | 0.8999 | 0.92 |

**Predicting**
--------------

We've implemented a prediction function that takes an image as input and returns the predicted class label and confidence score.
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


**Get Involved 🤝**
------------------

We're always looking for collaborators and contributors to help us improve this project. If you're interested in getting involved, please don't hesitate to reach out.


**Conclusion 🌟**
----------------

This project demonstrates the potential of machine learning in detecting potato diseases using image data. We hope that our work will inspire further research and development in precision agriculture.
