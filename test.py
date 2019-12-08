# Testing Your Library
#Having installed your library in the step above. Create a new file below anywhere on your computer and write the following code to predict the image of a fruit.


from orangelib.model import OrangeClassifier

classifier = FruitClassifier("path_to_trained_model")

fruit_name, confidence = classifier.predict("path_to_image")

print(" Fruit Name: ",fruit_name)
print("Prediction Confidence: ",confidence)
