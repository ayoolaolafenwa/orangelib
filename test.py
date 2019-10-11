# Testing Your Library
#Having installed your library in the step above. Create a new file below anywhere on your computer and write the following code to predict the image of a fruit.


from orangelib.model import FruitClassifier

classifier = FruitClassifier("orange_model_96.h5")

fruit_name, confidence = classifier.predict("sample.jpg")

print(" Fruit Name: ",fruit_name)
print("Prediction Confidence: ",confidence)