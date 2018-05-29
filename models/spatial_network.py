
import random
import cv2, numpy as np
import pickle
import csv


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, RepeatVector, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras import regularizers
from keras.callbacks import History 
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.layers import Input

def getVggModel(input_shape, printmod=1):

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.get_layer('block5_pool').output   # collect outputs from hidden layer, Block 5
    # stitch layers to the VGG16 layers
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    predictions = Dense(101, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # freeze original VGG16 layers
    for i, layer in enumerate(model.layers):
        if 'block1' in layer.name or 'block2' in layer.name or 'block3' in layer.name or \
        'block4' in layer.name or 'block5' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    if (printmod==1 ):
        model.summary()
    return model

def getResNet50Model(input_shape, printmod=1):

    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.get_layer('avg_pool').output   # collect outputs from hidden layer, Block 5
    # stitch layers to the VGG16 layers
    
    x = Flatten()(x)
    # add fully-connected & dropout layers
    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='fc-2')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(101, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # freeze ResNet layers
    for layer in model.layers[:-6]:
        layer.trainable = False
            
    if (printmod==1 ):
        model.summary()
    return model

def getVggModel2(input_shape, printmod=1):

    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base_model.get_layer('block5_pool').output   # collect outputs from hidden layer, Block 5
    # stitch layers to the VGG16 layers
    
    x = Flatten()(x)
    
    x = Dense(512, activation='relu',name='fc-1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu',name='fc-2')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(101, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # freeze original VGG16 layers
    for i, layer in enumerate(model.layers):
        if 'block1' in layer.name or 'block2' in layer.name or 'block3' in layer.name or \
        'block4' in layer.name or 'block5' in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True
            
    if (printmod==1 ):
        model.summary()
    return model

def getVggBottleNeckModel(input_shape, printmod=1):
    model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    
    if (printmod==1 ):
        model.summary()
    
    return model