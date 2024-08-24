**Potato Disease Detection: A Deep Learning Approach 🥔**
=====================================================

**Overview 🌱**
---------------

This project uses deep learning techniques to detect potato diseases from images. 
This project is inspired by the innovative approaches of PlantVillage Insights, and we hope to contribute to the development of precision agriculture.
The goal is to develop a model that can accurately identify diseases such as early blight, late blight, and healthy plants.


**Motivation 🌪️**
-----------------

Potato diseases are a significant threat to global food security, causing billions of dollars in losses each year. Traditional methods of disease detection are time-consuming and require expertise. This project aims to develop a machine learning model that can detect potato diseases quickly and accurately, enabling farmers and researchers to take targeted action to prevent the spread of disease.

**Inspiration**
---------------

The devastating impact of potato diseases on global food security, with losses estimated to be in the billions, has long been a pressing concern for agricultural researchers and farmers alike. However, accurate detection of these diseases remains a significant challenge, hindering our ability to develop effective management strategies and ensure a stable food supply. This project embarks on a journey to develop a machine learning model capable of detecting potato diseases using image data, leveraging the power of computer vision and deep learning to revolutionize the field of precision agriculture, as inspired by the innovative approaches of PlantVillage Insights.

**Problem Statement**
-------------------

The potato disease dataset is a comprehensive collection of images meticulously categorized into three distinct classes: early blight, late blight, and healthy. Each class represents a specific condition affecting potato crops, enabling researchers and agricultural experts to delve into the intricacies of disease identification, progression, and management.

** Solution 💡**

A machine learning model that can detect potato diseases from images. Using a combination of convolutional neural networks (CNNs) and transfer learning to achieve high accuracy.


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


