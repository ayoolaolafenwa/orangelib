# Orangelib

Oranges_classifier is the github repository hosting orangelib. Orangelib is a library built to simplify the implementation of computer vision in real problems. It is a library for classifying oranges into two classes: ripe and uripe. 

The model for classifying oranges is trained with MobilleNetV2. Both the trained model and the dataset used in training the model are available as releases in this repository.

Install Orangelib with:

**pip install orangelib**

###### The code to implement the classification of oranges with orangelib:
```from orangelib.model import OrangeClassifier

classifier = OrangeClassifier("trained_model.h5")

fruit_name, confidence = classifier.predict("path_to_image")

print(" Fruit Name: ",fruit_name)
print("Prediction Confidence: ",confidence)```
    


