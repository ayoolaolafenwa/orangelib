# EDIT THE model.py file
#This is where we define the class that users of our library are going to use for fruit class prediction. It will import the MobileNetV2 model from the **net.py** file.


import keras
from keras.layers import *
from keras.models import *
import os
from keras.preprocessing import image
#import MobileNetV2 from the net.py file
from .net import MobileNetV2

class FruitClassifier():
    def __init__(self, model_path):
        self.model = MobileNetV2(input_shape=(224, 224, 3), num_classes=2)
        self.model.load_weights(model_path)

        self.class_map = {0:"ripe",1:"unripe"}


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
        prediction_confidence = prediction.max()

        image_class = self.class_map[predicted_class]

        return image_class, prediction_confidence