# Orangelib

Oranges_classifier is the github repository hosting orangelib. Orangelib is a library built to simplify the implementation of computer vision in real problems. It is a library for classifying oranges into two classes: ripe and uripe. 

The model for classifying oranges is trained with MobilleNetV2. Both the trained model and the dataset used in training the model are available as releases in this repository.

Install Orangelib with:

**pip install orangelib**

## The code to implement the classification of oranges with orangelib:
```from orangelib.model import OrangeClassifier

classifier = OrangeClassifier("trained_model.h5")

fruit_name, confidence = classifier.predict("path_to_image")

print(" Fruit Name: ",fruit_name)
print("Prediction Confidence: ",confidence)
```
Looking into each line of code: 
```from orangelib.model import OrangeClassifier
```
*We import in the class for classifying oranges from orangelib*
```classifier = OrangeClassifier("trained_model.h5")
```
The path to model used for classifying oranges is loaded.
```fruit_name, confidence = classifier.predict("path_to_image")
```
*The path to image to be predicted is loaded*
```print(" Fruit Name: ",fruit_name)
print("Prediction Confidence: ",confidence)
```
*The fruit name and the confidence of class predicted are printed out*

We shall test the library by performing inference on eight images:
*sample1*
![alt_test](photos/sample1.jpg)
fruit_name, confidence = classifier.predict("sample1.jpg")
