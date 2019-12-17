Oranges_classifier is the github repository hosting orangelib. Orangelib is a library built to simplify the implementation of computer vision in real problems. It is a library for classifying oranges into two classes: ripe and uripe. 

The model for classifying oranges is trained with MobilleNetV2. Both the trained model and the dataset used in training the model are available as releases in this repository.

Install Orangelib with:

pip install orangelib

Using few lines of code you can easily classify an orange:

from orangelib.model import OrangeClassifier

classifier = OrangeClassifier("path_to_trained_model")
fruit_name, confidence = classifier.predict("path_to_image")
print(" Fruit Name: ",fruit_name)
print("Prediction Confidence: ", confidence)
    

