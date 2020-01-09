import tensorflow 
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os
from tensorflow.keras.preprocessing import image
from .net import MobileNetV2
import numpy as np
from tensorflow.keras import backend as K


class OrangeClassifier():
    def __init__(self, model_path):
        self.model = MobileNetV2(input_shape=(224, 224, 3), num_classes=2)
        self.model.load_weights(model_path)

        self.class_map = {0:"ripe orange",1:"unripe orange"}


    def preprocess_input(self,x):

        x *= (1. / 255)

        return x

    def predict(self,image_path):

        image_to_predict = image.load_img(image_path, target_size=(
        224, 224))
        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
        image_to_predict = np.expand_dims(image_to_predict, axis=0)

        image_to_predict = self.preprocess_input(image_to_predict)

        prediction = self.model.predict(image_to_predict)

        predicted_class = prediction.argmax()
        prediction_confidence = prediction.max() * 100

        image_class = self.class_map[predicted_class]

        return image_class, prediction_confidence

    def predictBatch(self,image_paths):

        #create an array to store all processed images
        images_array = []

        #loop over the batch of images sent
        for image_path in image_paths:
            image_to_predict = image.load_img(image_path, target_size=(224, 224))
            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
            image_to_predict = np.expand_dims(image_to_predict, axis=0)

            image_to_predict = self.preprocess_input(image_to_predict)

            #append the processed images to the array
            images_array.append(image_to_predict)
        
        #merge all the images together as one array
        images = np.concatenate(images_array)

        predictions = self.model.predict(images)

        #use axis=1 to compute the argmax and max
        predicted_classes = predictions.argmax(axis=1)
        prediction_confidence = predictions.max(axis=1) * 100

        #create an array to store the names of the classes
        predicted_class_names = []

        #loop over all the predictions and convert indexes to class names
        for predicted_index in predicted_classes:
            class_name = self.class_map[predicted_index]
            #append the class name to the array
            predicted_class_names.append(class_name)

        #return the class name list and the confidences
        return predicted_class_names,prediction_confidence    

class BananaClassifier():
    def __init__(self, model_path):
        self.model = MobileNetV2(input_shape=(224, 224, 3), num_classes=2)
        self.model.load_weights(model_path)

        self.class_map = {0:"ripe banana",1:"unripe banana"}


    def preprocess_input(self,x):

        x *= (1. / 255)

        return x

    def predict(self,image_path):

        image_to_predict = image.load_img(image_path, target_size=(
        224, 224))
        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
        image_to_predict = np.expand_dims(image_to_predict, axis=0)

        image_to_predict = self.preprocess_input(image_to_predict)

        prediction = self.model.predict(image_to_predict)

        predicted_class = prediction.argmax()
        prediction_confidence = prediction.max() * 100

        image_class = self.class_map[predicted_class]

        return image_class, prediction_confidence  

    def predictBatch(self,image_paths):

        #create an array to store all processed images
        images_array = []

        #loop over the batch of images sent
        for image_path in image_paths:
            image_to_predict = image.load_img(image_path, target_size=(224, 224))
            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
            image_to_predict = np.expand_dims(image_to_predict, axis=0)

            image_to_predict = self.preprocess_input(image_to_predict)

            #append the processed images to the array
            images_array.append(image_to_predict)
        
        #merge all the images together as one array
        images = np.concatenate(images_array)

        predictions = self.model.predict(images)

        #use axis=1 to compute the argmax and max
        predicted_classes = predictions.argmax(axis=1)
        prediction_confidence = predictions.max(axis=1) * 100

        #create an array to store the names of the classes
        predicted_class_names = []

        #loop over all the predictions and convert indexes to class names
        for predicted_index in predicted_classes:
            class_name = self.class_map[predicted_index]
            #append the class name to the array
            predicted_class_names.append(class_name)

        #return the class name list and the confidences
        return predicted_class_names,prediction_confidence          

class AppleClassifier():
    def __init__(self, model_path):
        self.model = MobileNetV2(input_shape=(224, 224, 3), num_classes=2)
        self.model.load_weights(model_path)

        self.class_map = {0:"green apple",1:"red apple"}


    def preprocess_input(self,x):

        x *= (1. / 255)

        return x

    def predict(self,image_path):

        image_to_predict = image.load_img(image_path, target_size=(
        224, 224))
        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
        image_to_predict = np.expand_dims(image_to_predict, axis=0)

        image_to_predict = self.preprocess_input(image_to_predict)

        prediction = self.model.predict(image_to_predict)

        predicted_class = prediction.argmax()
        prediction_confidence = prediction.max() * 100

        image_class = self.class_map[predicted_class]

        return image_class, prediction_confidence


    def predictBatch(self,image_paths):

        #create an array to store all processed images
        images_array = []

        #loop over the batch of images sent
        for image_path in image_paths:
            image_to_predict = image.load_img(image_path, target_size=(224, 224))
            image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
            image_to_predict = np.expand_dims(image_to_predict, axis=0)

            image_to_predict = self.preprocess_input(image_to_predict)

            #append the processed images to the array
            images_array.append(image_to_predict)
        
        #merge all the images together as one array
        images = np.concatenate(images_array)

        predictions = self.model.predict(images)

        #use axis=1 to compute the argmax and max
        predicted_classes = predictions.argmax(axis=1)
        prediction_confidence = predictions.max(axis=1) * 100

        #create an array to store the names of the classes
        predicted_class_names = []

        #loop over all the predictions and convert indexes to class names
        for predicted_index in predicted_classes:
            class_name = self.class_map[predicted_index]
            #append the class name to the array
            predicted_class_names.append(class_name)

        #return the class name list and the confidences
        return predicted_class_names,prediction_confidence     
