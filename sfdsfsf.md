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

* Potato Plant Diseases: is a comprehensive collection of images meticulously categorized into three distinct classes: early blight, late blight, and healthy. 

* Each class represents a specific condition affecting potato crops, enabling researchers and agricultural experts to delve into the intricacies of disease identification, progression, and management.


## `Methodology` 🔍

1. **Data Preprocessing 🔄**: Applying random transformations to the images to artificially increase the size of the training set using TF-keras pre-processing layers.
2. **Model Selection 🤔**: Performance test accuracy and loss of 4 pre-trained models: RESNET, Inception, Xception & NASnetMobile 
3. **Fine-tuning 🔩**: Unfreezing the weights and training models on the augmented training data. Using learning rate reduction and early stopping callbacks to prevent overfitting.
4. **Hyperparameter Tuning 🔧**: Using a grid search approach, Adam optimizer and sparse categorical cross-entropy loss.5.
5. **Prediction**: Made on a random sample of images from the validation set. Incorrect predictions are also identified and visualized.



## `Model Architectures` 🤖

* **Xception**: A deep learning model that uses a novel architecture to achieve state-of-the-art results on image classification tasks.
* **ResNet50**: A popular deep learning model that uses residual connections to ease the training process.
* **Inception**: A deep learning model that uses multiple parallel branches to capture features at different scales.
* **NASNetMobile**: A deep learning model that uses neural architecture search to find the optimal architecture for mobile devices.


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
