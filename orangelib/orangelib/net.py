import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K



def LinearBottleNeck(x,in_channels,out_channels,stride,expansion):

    
    out = Conv2D(in_channels*expansion,kernel_size=1,strides=1,padding="same",use_bias=False)(x)
    out = BatchNormalization()(out)
    out = Activation(relu6)(out)

    
    out = DepthwiseConv2D(kernel_size=3,strides=stride,padding="same",use_bias=False)(out)
    out = BatchNormalization()(out)
    out = Activation(relu6)(out)

    
    out = Conv2D(out_channels,kernel_size=1,strides=1,padding="same",use_bias=False)(out)
    out = BatchNormalization()(out)

    
    if stride == 1 and in_channels == out_channels:
        out = add([out,x])
    return out


    
def relu6(x):
    return K.relu(x,max_value=6)


def MobileNetV2(input_shape,num_classes=1000,multiplier=1.0):

    images = Input(shape=input_shape)

    net = Conv2D(int(32*multiplier),kernel_size=3,strides=2,padding="same",use_bias=False)(images)
    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(16 * multiplier), stride=1, expansion=1)

    
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(24 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(24 * multiplier), out_channels=int(24 * multiplier), stride=1, expansion=6)

    
    net = LinearBottleNeck(net, in_channels=int(24 * multiplier), out_channels=int(32 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(32 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(32 * multiplier), stride=1, expansion=6)

    
    net = LinearBottleNeck(net, in_channels=int(32 * multiplier), out_channels=int(64 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(64 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(64 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(64 * multiplier), stride=1, expansion=6)

    
    net = LinearBottleNeck(net, in_channels=int(64 * multiplier), out_channels=int(96 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(96 * multiplier), out_channels=int(96 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(96 * multiplier), out_channels=int(96 * multiplier), stride=1, expansion=6)

    
    net = LinearBottleNeck(net, in_channels=int(96 * multiplier), out_channels=int(160 * multiplier), stride=2, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(160 * multiplier), out_channels=int(160 * multiplier), stride=1, expansion=6)
    net = LinearBottleNeck(net, in_channels=int(160 * multiplier), out_channels=int(160 * multiplier), stride=1, expansion=6)

    
    net = LinearBottleNeck(net, in_channels=int(160 * multiplier), out_channels=int(320 * multiplier), stride=1, expansion=6)


    

    if multiplier > 1.0:
        final_channels = int(1280 * multiplier)
    else:
        final_channels = 1280

    
    net = Conv2D(final_channels,kernel_size=1,padding="same",use_bias=False)(net)
    net = BatchNormalization()(net)
    net = Activation(relu6)(net)
    net = Dropout(0.3)(net)

    
    net = AveragePooling2D(pool_size=(7,7))(net)
    net = Conv2D(num_classes,kernel_size=1,use_bias=False)(net)
    net = Flatten()(net)
    net = Activation("softmax")(net)

    return Model(inputs=images,outputs=net)