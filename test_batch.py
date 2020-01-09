from orangelib.model import OrangeClassifier

classifier = OrangeClassifier("path_to_trained_model.h5")


fruit_names_list, confidence_list = classifier.predictBatch(["path_to_image1", "path_to_image2", "path_to_image3"])

for fruit_names, confidence in zip(fruit_names_list,confidence_list):
    print("Fruit Name: ",fruit_names)
    print("Prediction Confidence: ", confidence)
