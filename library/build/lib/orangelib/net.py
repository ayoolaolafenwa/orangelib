# EDIT THE net.py file
#Add the following to the net.py file


import keras
from keras.layers import *
from keras.models import *
import os
from keras.preprocessing import image

#The Linear Bottleneck increases the number of channels going into the depthwise convs
def LinearBottleNeck(x,in_channels,out_channels,stride,expansion):

    #Expand the input channels
    out = Conv2D(in_channels*expansion,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    out = BatchNormalization()(out)
    out = Activation(relu6)(out)

    #perform 3 x 3 depthwise conv
    out = DepthwiseConv2D(kernel_size=3,strides=stride,padding="same",use_bias=False)(out)
    out = BatchNormalization()(out)
    out = Activation(relu6)(out)

    #Reduce the output channels to conserve computation
    out = Conv2D(out_channels,kernel_size=1,strides=1,padding="same",use_bias=False)(out)
    out = BatchNormalization()(out)

    #Perform resnet-like addition if input image and output image are same dimesions
    if stride == 1 and in_channels == out_channels:
        out = add([out,x])
    return out





    #Relu6 is the standard relu with the maximum thresholded to 6
def relu6(x):
    return K.relu(x,max_value=6)


def MobileNetV2(input_shape,num_classes=1000,multiplier=1.0):

    images = Input(shape=input_shape)

    net = Conv2D(int(32*multiplier),kernel_size=3,strides=2,padding="same",use_bias=False)(images)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    #First block with 16 * multplier output with stride of 1
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(16 * multiplier), stride=1, expansion=1)

    #Second block with 24 * multplier output with first stride of 2
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(24 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(24 * multiplier), out_channels=int(24 * multiplier), stride=1, expansion=6)

    #Third block with 32 * multplier output with first stride of 2
    net = LinearBottleNeck(net, in_channels=int(24 * multiplier), out_channels=int(32 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(32 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(32 * multiplier), stride=1, expansion=6)

    #Fourth block with 64 * multplier output with first stride of 2
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(64 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(64 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(64 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(64 * multiplier), stride=1, expansion=6)

    #Fifth block with 96 * multplier output with first stride of 1
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(96 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(96 * multiplier), out_channels=int(96 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(96 * multiplier), out_channels=int(96 * multiplier), stride=1, expansion=6)

    #Sixth block with 160 * multplier output with first stride of 2
    net = LinearBottleNeck(net, in_channels=int(96 * multiplier), out_channels=int(160 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(160 * multiplier), out_channels=int(160 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(160 * multiplier), out_channels=int(160 * multiplier), stride=1, expansion=6)

    #Seventh block with 320 * multplier output with stride of 1
    net = LinearBottleNeck(net, in_channels=int(160 * multiplier), out_channels=int(320 * multiplier), stride=1, expansion=6)


    #Final number of channels must be at least 1280

    if multiplier > 1.0:
        final_channels = int(1280 * multiplier)
    else:
        final_channels = 1280

    #Expand the output channels
    net = Conv2D(final_channels,kernel_size=1,padding="same",use_bias=False)(net)
    net = BatchNormalization()(net)
    net = Activation(relu6)(net)
    net = Dropout(0.3)(net)

    #Final Classification is by 1 x 1 Conv
    net = AveragePooling2D(pool_size=(7,7))(net)
    net = Conv2D(num_classes,kernel_size=1,use_bias=False)(net)
    net = Flatten()(net)
    net = Activation("softmax")(net)

    return Model(inputs=images,outputs=net)